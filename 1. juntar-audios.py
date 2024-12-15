import os
import PIL
import eyed3
from pydub import AudioSegment
import shutil
from PIL import Image
import io
import stat

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
                    audio_durations.append(0)  # Add a default duration if audio_file is None
                
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
                            os.chmod(cover_path, stat.S_IWRITE)  # Cambiar los permisos del archivo
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

        # Guardar el audio resultante con calidad 320 kbps y el nombre de la carpeta en la misma carpeta
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
        text = f"{band_names[0]} - {album_names[0]} (Full Album)\n\nGenre: {audio_genres[0]}\nYear: {audio_years[0]}\n\nTracklist:\n\n"
    else:
        text = "Unknown - Unknown\nGenre: Unknown\nYear: Unknown\n\n"
    
    # Agregar "Intro (00:00)" al inicio del tracklist
    text += "0. Intro (00:00)\n"
    
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