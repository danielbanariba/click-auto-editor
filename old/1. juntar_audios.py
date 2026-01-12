import os
import PIL
import eyed3
from pydub import AudioSegment
import shutil
from PIL import Image
import io
import stat
import time
from playwright.sync_api import sync_playwright
from urllib.parse import quote, unquote
import random
from pathlib import Path
import sys

def clean_url(url):
    """
    Limpia la URL eliminando parámetros innecesarios y localizaciones
    """
    if not url:
        return None
        
    # Decodificar la URL si está codificada
    url = unquote(url)
    
    # Limpiar por plataforma
    if 'spotify.com' in url:
        if '/album/' in url or '/artist/' in url:
            return url.split('?')[0].split('#')[0]
    elif 'youtube.com' in url:
        if 'watch?v=' in url:
            video_id = url.split('watch?v=')[1].split('&')[0]
            return f"https://www.youtube.com/watch?v={video_id}"
        elif '/channel/' in url or '/c/' in url or '/user/' in url:
            return url.split('?')[0].split('#')[0]
            
    return url.split('?')[0].split('#')[0]

def search_music_links(band_name, album_name):
    """
    Busca enlaces de música y redes sociales usando Playwright
    """
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
    
    queries = {
        'music': [
            (f"{band_name} official bandcamp", ['bandcamp.com']),
            (f"{band_name} {album_name} spotify", ['spotify.com']),
            (f"{band_name} {album_name} apple music", ['music.apple.com']),
            (f"{band_name} {album_name} deezer", ['deezer.com']),
            (f"{band_name} {album_name} amazon music", ['amazon.com/music']),
            (f"{band_name} {album_name} youtube music", ['music.youtube.com'])
        ],
        'social': [
            (f"{band_name} official profile facebook", ['facebook.com']),
            (f"{band_name} official profile instagram", ['instagram.com']),
            (f"{band_name} official youtube channel", ['youtube.com/channel', 'youtube.com/c']),
            (f"{band_name} official tiktok account", ['tiktok.com/@']),
            (f"{band_name} official twitter account", ['twitter.com', 'x.com'])
        ],
        'metal': [
            (f"site:metal-archives.com/bands {band_name}", ['metal-archives.com/bands']),
            (f"site:spirit-of-metal.com/en/band {band_name}", ['spirit-of-metal.com/en/band'])
        ]
    }

    try:
        with sync_playwright() as p:
            browser = p.firefox.launch(
                headless=True,
                firefox_user_prefs={
                    "media.autoplay.default": 2,
                    "media.autoplay.blocking_policy": 2
                }
            )
            
            context = browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            
            page = context.new_page()
            page.set_default_timeout(30000)
            page.set_default_navigation_timeout(30000)
            
            try:
                for category, category_queries in queries.items():
                    for query, domains in category_queries:
                        try:
                            encoded_query = quote(query)
                            url = f"https://duckduckgo.com/html/?q={encoded_query}&kl=wt-wt&kp=-2"
                            
                            page.goto(url, wait_until="networkidle")
                            page.wait_for_selector('.result__body', state="attached")
                            
                            page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")
                            time.sleep(0.5)
                            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                            time.sleep(0.5)
                            
                            links_elements = page.query_selector_all('.result__url')
                            additional_links = page.query_selector_all('.result__extras__url')
                            all_links = links_elements + additional_links
                            
                            for element in all_links:
                                href = element.get_attribute('href')
                                if not href:
                                    text_content = element.text_content()
                                    if text_content:
                                        href = text_content
                                
                                if href:
                                    href = clean_url(href)
                                    
                                    if 'bandcamp.com' in href and not links['bandcamp']:
                                        if f"{band_name.lower().replace(' ', '')}.bandcamp.com" in href.lower():
                                            links['bandcamp'] = href
                                    elif 'spotify.com' in href and not links['spotify']:
                                        links['spotify'] = href
                                    elif 'music.apple.com' in href and not links['apple_music']:
                                        links['apple_music'] = href
                                    elif 'deezer.com' in href and not links['deezer']:
                                        links['deezer'] = href
                                    elif 'amazon.com/music' in href and not links['amazon']:
                                        links['amazon'] = href
                                    elif 'music.youtube.com' in href and not links['youtube_music']:
                                        links['youtube_music'] = href
                                    elif 'facebook.com' in href and not links['facebook']:
                                        links['facebook'] = href
                                    elif 'instagram.com' in href and not links['instagram']:
                                        if '/p/' not in href and '/reel/' not in href:
                                            links['instagram'] = href
                                    elif 'youtube.com' in href and not links['youtube']:
                                        if '/channel/' in href or '/c/' in href:
                                            links['youtube'] = href
                                    elif 'tiktok.com' in href and not links['tiktok']:
                                        if '@' in href:
                                            links['tiktok'] = href
                                    elif ('twitter.com' in href or 'x.com' in href) and not links['twitter']:
                                        links['twitter'] = href
                                    elif 'metal-archives.com/bands' in href and not links['metal_archives']:
                                        links['metal_archives'] = href
                                    elif 'spirit-of-metal.com' in href and not links['spirit_of_metal']:
                                        links['spirit_of_metal'] = href
                            
                            time.sleep(random.uniform(2, 4))
                            
                        except Exception as e:
                            print(f"Error en búsqueda {query}: {str(e)}")
                            continue
                
            except Exception as e:
                print(f"Error general: {str(e)}")
            
            finally:
                browser.close()
        
    except Exception as e:
        print(f"Error al iniciar Playwright: {e}")
    
    return links

# Importar configuración
sys.path.append(str(Path(__file__).parent))
from config import DIR_JUNTAR_AUDIOS, AUDIO_FORMATS

# FFmpeg está en PATH en Linux, no necesitamos especificar rutas
# AudioSegment.converter y AudioSegment.ffmpeg se detectan automáticamente

# Definir la ruta de la carpeta que contiene los audios
main_dir_path = str(DIR_JUNTAR_AUDIOS)

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
        audio_files = [file_name for file_name in os.listdir(folder_path) if file_name.endswith(AUDIO_FORMATS)]
        if len(audio_files) <= 1:
            continue

        for file_name in audio_files:
            # Verificar si el archivo es un audio
            if file_name.endswith(AUDIO_FORMATS):
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
        print(f"\nBuscando enlaces para: {band_names[0]} - {album_names[0]}")
        # Buscar enlaces usando la nueva función con Playwright
        links = search_music_links(band_names[0], album_names[0])
            
        text = f"{band_names[0]} - {album_names[0]} (Full EP)\n\n"
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

    # Guardar la información recolectada y crear el archivo de texto
    with open(os.path.join(folder_path, f"{folder_name}.txt"), "w", encoding='utf-8') as f:
        f.write(text)

    print(folder_name, "completado")

print("************************************************************************************************************************************************")
print("----------------------------------------------------SE TERMINO CON EXITO---------------------------------------------------------------------")
print("************************************************************************************************************************************************")