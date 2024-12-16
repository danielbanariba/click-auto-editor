import os
import PIL
import eyed3
from pydub import AudioSegment
import shutil
from PIL import Image
import io
import stat
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import quote

def clean_url(url):
    """
    Limpia la URL eliminando parámetros innecesarios y localizaciones
    """
    if not url or url.startswith('/search'):
        return None
        
    # Lista de parámetros a eliminar
    params_to_remove = ['hl', 'lang', 'locale', 'ref', 'utm_source', 'utm_medium', 'utm_campaign']
    
    # Eliminar parámetros de lenguaje y tracking
    if '?' in url:
        base_url = url.split('?')[0]
        return base_url
    
    return url

def search_music_links(band_name, album_name):
    links = {
        'bandcamp': None,
        'spotify': None,
        'apple_music': None,
        'deezer': None,
        'amazon': None,
        'youtube_music': None,
        'facebook': None,
        'instagram': None,
        'metal_archives': None,
        'youtube': None,
        'tiktok': None,
        'twitter': None,
        'spirit_of_metal': None
    }
    
    # Queries específicas para cada plataforma
    music_queries = [
        f"{band_name} {album_name} bandcamp",
        f"{band_name} {album_name} spotify",
        f"{band_name} {album_name} apple music",
        f"{band_name} {album_name} deezer",
        f"{band_name} {album_name} amazon music",
        f"{band_name} {album_name} youtube music"
    ]
    
    social_queries = [
        f"{band_name} official facebook",
        f"{band_name} official instagram",
        f"{band_name} official youtube channel",
        f"{band_name} official tiktok",
        f"{band_name} official twitter",
    ]
    
    metal_database_queries = [
        f"site:metal-archives.com/bands {band_name}",
        f"site:spirit-of-metal.com/en/band {band_name}"
    ]
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Búsqueda de servicios de música
        for query in music_queries:
            encoded_query = quote(query)
            url = f"https://www.google.com/search?q={encoded_query}"
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if not href.startswith('/search'):
                    if 'bandcamp.com' in href and not links['bandcamp']:
                        links['bandcamp'] = clean_url(href)
                    elif 'spotify.com' in href and not links['spotify']:
                        links['spotify'] = clean_url(href)
                    elif 'music.apple.com' in href and not links['apple_music']:
                        links['apple_music'] = clean_url(href)
                    elif 'deezer.com' in href and not links['deezer']:
                        links['deezer'] = clean_url(href)
                    elif 'amazon.com' in href and not links['amazon']:
                        links['amazon'] = clean_url(href)
                    elif ('music.youtube.com' in href or 'youtube.com/music' in href) and not links['youtube_music']:
                        links['youtube_music'] = clean_url(href)
            
            time.sleep(2)
        
        # Búsqueda de redes sociales
        for query in social_queries:
            encoded_query = quote(query)
            url = f"https://www.google.com/search?q={encoded_query}"
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if not href.startswith('/search'):
                    # Facebook
                    if 'facebook.com' in href and not links['facebook']:
                        if '/pages/' in href or '/groups/' not in href:
                            links['facebook'] = clean_url(href)
                    # Instagram
                    elif 'instagram.com' in href and not links['instagram']:
                        if '/p/' not in href:
                            links['instagram'] = clean_url(href)
                    # YouTube
                    elif 'youtube.com' in href and not links['youtube']:
                        if '/channel/' in href or '/c/' in href or '/user/' in href:
                            links['youtube'] = clean_url(href)
                    # TikTok
                    elif 'tiktok.com' in href and not links['tiktok']:
                        if '@' in href:
                            links['tiktok'] = clean_url(href)
                    # Twitter/X
                    elif ('twitter.com' in href or 'x.com' in href) and not links['twitter']:
                        if '/status/' not in href and not href.startswith('/search'):
                            links['twitter'] = clean_url(href)
            
            time.sleep(2)
        
        # Búsqueda de bases de datos de metal
        for query in metal_database_queries:
            encoded_query = quote(query)
            url = f"https://www.google.com/search?q={encoded_query}"
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a'):
                href = link.get('href', '')
                # Metal Archives
                if 'metal-archives.com/bands/' in href and band_name.lower() in href.lower() and not links['metal_archives']:
                    links['metal_archives'] = clean_url(href)
                # Spirit of Metal
                elif 'spirit-of-metal.com' in href and '/band/' in href and not links['spirit_of_metal']:
                    links['spirit_of_metal'] = clean_url(href)
            
            time.sleep(2)
        
        # Eliminar cualquier link que sea None o comience con /search
        for key in links:
            if not links[key] or links[key].startswith('/search'):
                links[key] = None
        
        return links
    except Exception as e:
        print(f"Error buscando links: {e}")
        return links

# Exportamos el directorio de FFmpeg para poder exportar el archivo final
AudioSegment.converter = "C:\\Program Files\\FFmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffmpeg = "C:\\Program Files\\FFmpeg\\bin\\ffprobe.exe"

# Definir la ruta de la carpeta que contiene los audios
main_dir_path = "E:\\01_edicion_automatizada\\02_juntar_audios"

# Recorre todos los directorios en la ruta principal
for folder_name in os.listdir(main_dir_path):
    folder_path = os.path.join(main_dir_path, folder_name)

    if os.path.isdir(folder_path):
        new_folder_path = folder_path.replace('–', '-')

        # Mueve el directorio si el nombre ha cambiado
        if folder_path != new_folder_path:
            shutil.move(folder_path, new_folder_path)
            folder_path = new_folder_path

        # Crear una lista vacía para almacenar datos de los audios
        audios = []
        audio_names = []
        audio_years = []
        audio_genres = []
        audio_durations = []
        audio_paths = []
        band_names = []  
        album_names = []  

        # Si existe un archivo de audio en la carpeta, ignora la carpeta
        audio_files = [file_name for file_name in os.listdir(folder_path) if file_name.endswith((".MP3",".Mp3", ".mp3", ".flac", ".wav", ".wma", ".m4a"))]
        if len(audio_files) <= 1:
            continue  
        
        for file_name in audio_files:
            # Verificar si el archivo es un audio
            if file_name.endswith((".MP3",".Mp3", ".mp3", ".flac", ".wav", ".wma", ".m4a")):
                # Abrir el audio y agregarlo a la lista
                audio_path = os.path.join(folder_path, file_name)
                audio = AudioSegment.from_file(audio_path)
                audios.append(audio)
                audio_paths.append(audio_path)
                # Extraer el nombre del audio y su duración
                audio_file = eyed3.load(audio_path)
                if audio_file is not None and audio_file.info is not None:
                    title = audio_file.tag.title if audio_file.tag and audio_file.tag.title else "Unknown"
                    audio_names.append(title)
                    duration = audio_file.info.time_secs
                    audio_durations.append(duration)
                else:
                    audio_names.append("Unknown")
                    audio_durations.append(0)
                
                # Extraer el año y el género solo de la primera canción
                if not audio_years and not audio_genres and not band_names and not album_names and audio_file is not None and audio_file.tag is not None:
                    year = audio_file.tag.getBestDate() if audio_file.tag.getBestDate() else "Unknown"
                    audio_years.append(year)
                    genre = audio_file.tag.genre if audio_file.tag.genre else "Unknown"
                    audio_genres.append(genre)
                    band = audio_file.tag.artist if audio_file.tag.artist else "Unknown"
                    band_names.append(band)
                    album = audio_file.tag.album if audio_file.tag.album else "Unknown"
                    album_names.append(album)

        # Verificar si existe alguna imagen en el directorio
        image_formats = [".png", ".jpg", ".jpeg", ".jfif", ".gif"]
        has_image = any(file_name.endswith(tuple(image_formats)) for file_name in os.listdir(folder_path))
        
        # Si no existe ninguna imagen, recorrer la lista de audios
        if not has_image:
            for audio_path in audio_paths:
                audio_file = eyed3.load(audio_path)
                if audio_file is not None and audio_file.tag is not None:
                    # Extraer las imágenes del archivo de audio
                    for image in audio_file.tag.images:
                        # Guardar la primera imagen como "Cover.jpg" y terminar el bucle
                        try:
                            img = Image.open(io.BytesIO(image.image_data))
                            if img.mode == 'RGBA':
                                img = img.convert('RGB')
                            cover_path = os.path.join(folder_path, "Cover.jpg")
                            img.save(cover_path, 'JPEG')
                            os.chmod(cover_path, stat.S_IWRITE)
                            break
                        except PIL.UnidentifiedImageError:
                            print(f"Cannot identify image in file: {audio_path}")
                        except PermissionError:
                            print(f"Permission denied: '{cover_path}'. The file might be open, read-only or the script might not have the necessary permissions.")
                        except FileNotFoundError:
                            print(f"File not found: '{cover_path}'. The file might not have been created.")
                        else:
                            continue
                        break

        # Juntar los audios
        combined = sum(audios, AudioSegment.empty())

        # Guardar el audio resultante
        combined.export(os.path.join(folder_path, f"{folder_name}.mp3"), format="mp3", bitrate="320k")

        # Eliminar los archivos de audio originales
        for audio_path in audio_paths:
            try:
                os.chmod(audio_path, stat.S_IWRITE) # Cambiar los permisos del archivo para poder eliminarlo
                os.remove(audio_path)
            except PermissionError:
                print(f"Permission denied: '{audio_path}'. The file might be open, read-only or the script might not have the necessary permissions.")

    else:
        print(f"'{folder_path}' no es un directorio válido.")
            
    #TODO Cambiar el (Full album) por el ep, compilacion o cualquiera segun sea la epoca
    if audio_genres and audio_years and band_names and album_names: 
        # Buscar enlaces
        links = search_music_links(band_names[0], album_names[0])
            
        text = f"{band_names[0]} - {album_names[0]} (Full Album)\n\n"
        text += f"Genre: {audio_genres[0]}\nYear: {audio_years[0]}\n\n"
        
        # Agregar enlaces de streaming
        text += "Stream/Download:\n"
        if links['bandcamp']:
            text += f"Bandcamp: {links['bandcamp']}\n"
        if links['spotify']:
            text += f"Spotify: {links['spotify']}\n"
        if links['youtube_music']:
            text += f"YouTube Music: {links['youtube_music']}\n"
        if links['apple_music']:
            text += f"Apple Music: {links['apple_music']}\n"
        if links['deezer']:
            text += f"Deezer: {links['deezer']}\n"
        if links['amazon']:
            text += f"Amazon Music: {links['amazon']}\n"

        # Agregar redes sociales
        text += "\nFollow:\n"
        if links['facebook']:
            text += f"Facebook: {links['facebook']}\n"
        if links['instagram']:
            text += f"Instagram: {links['instagram']}\n"
        if links['youtube']:
            text += f"YouTube: {links['youtube']}\n"
        if links['tiktok']:
            text += f"TikTok: {links['tiktok']}\n"
        if links['twitter']:
            text += f"X/Twitter: {links['twitter']}\n"
        if links['metal_archives']:
            text += f"Metal Archives: {links['metal_archives']}\n"
        if links['spirit_of_metal']:
            text += f"Spirit of Metal: {links['spirit_of_metal']}\n"
        
        text += "\nTracklist:\n\n"
    else:
        text = "Unknown - Unknown\nGenre: Unknown\nYear: Unknown\n\n"
    
    # Agregar "Intro (00:00)" al inicio del tracklist
    text += "0 - Intro (00:00)\n"
    
    total_duration = 0

    # Agrega el nombre de cada audio y su duración al archivo de texto
    for i, audio in enumerate(audios):
        # Agrega 8 segundos solo al inicio de la primera canción por el intro del video 
        if i == 0:
            total_duration += 8
        minutes, seconds = divmod(total_duration, 60)
        if i < len(audios):
            total_duration += audio_durations[i]
            text += f"{i+1} - {audio_names[i]} ({int(minutes):02d}:{int(seconds):02d})\n"

    # Guardar la información recolectada y crea en un archivo de texto
    with open(os.path.join(folder_path, f"{folder_name}.txt"), "w", encoding='utf-8') as f:
        f.write(text)

    print(folder_name, "completado")

print("************************************************************************************************************************************************")
print("----------------------------------------------------SE TERMINO CON EXITO---------------------------------------------------------------------")
print("************************************************************************************************************************************************")