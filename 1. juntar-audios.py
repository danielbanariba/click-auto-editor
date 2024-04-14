import os
import eyed3
from pydub import AudioSegment
import shutil
from PIL import Image #pip install pillow
import io
import stat
from multiprocessing import Pool, cpu_count

# Exportamos el directorio de FFmpeg para poder exportar el archivo final
AudioSegment.converter = "C:\\Program Files\\FFmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffmpeg = "C:\\Program Files\\FFmpeg\\bin\\ffprobe.exe"

# Definir la ruta de la carpeta que contiene los audios
main_dir_path = "D:\\01_edicion_automatizada\\01_limpieza_de_impurezas"

# Recorre todos los directorios en la ruta principal
def process_folders(folder_name):
    for folder_name in os.listdir(main_dir_path):
        folder_path = os.path.join(main_dir_path, folder_name)

        if os.path.isdir(folder_path):
            new_folder_path = folder_path.replace('–', '-')

            # Mueve el directorio si el nombre ha cambiado
            if folder_path != new_folder_path:
                shutil.move(folder_path, new_folder_path)
                folder_path = new_folder_path

            # Crear una lista vacía para almacenar los audios
            audios = []
            audio_names = []
            audio_years = []
            audio_genres = []
            # Crear una lista vacía para almacenar las duraciones de los audios
            audio_durations = []
            # Crear una lista vacía para almacenar las rutas de los audios
            audio_paths = []
            band_names = []  
            album_names = []  

            # Recorrer la carpeta
            for file_name in os.listdir(folder_path):
                # Verificar si el archivo es un audio
                if file_name.endswith(".mp3"):
                    # Abrir el audio y agregarlo a la lista
                    audio_path = os.path.join(folder_path, file_name)
                    audio = AudioSegment.from_mp3(audio_path)
                    audios.append(audio)
                    audio_paths.append(audio_path)
                    # Extraer el nombre del audio y su duración
                    audio_file = eyed3.load(audio_path)
                    if audio_file.tag is not None:
                        title = audio_file.tag.title if audio_file.tag.title else "Unknown"
                    else:
                        title = "Unknown"
                    audio_names.append(title)
                    duration = audio_file.info.time_secs
                    audio_durations.append(duration)

                    # Extraer el año y el género solo de la primera canción
                    if not audio_years and not audio_genres and not band_names and not album_names:
                        year = audio_file.tag.getBestDate() if audio_file.tag.getBestDate() else "Unknown"
                        audio_years.append(year)
                        genre = audio_file.tag.genre if audio_file.tag.genre else "Unknown"
                        audio_genres.append(genre)
                        band = audio_file.tag.artist if audio_file.tag.artist else "Unknown"  # Extract band name
                        band_names.append(band)
                        album = audio_file.tag.album if audio_file.tag.album else "Unknown"  # Extract album name
                        album_names.append(album)

            # Verificar si hay alguna imagen en el directorio
            image_formats = [".png", ".jpg", ".jpeg", ".jfif"]
            has_image = any(file_name.endswith(tuple(image_formats)) for file_name in os.listdir(folder_path))

            # Si no hay ninguna imagen, recorrer la lista de audios
            if not has_image:
                for audio_path in audio_paths:
                    audio_file = eyed3.load(audio_path)
                    # Extraer las imágenes del archivo de audio
                    for image in audio_file.tag.images:
                        # Guardar la primera imagen como "Cover.jpg" y terminar el bucle
                        img = Image.open(io.BytesIO(image.image_data))
                        img.save(os.path.join(folder_path, "Cover.jpg"))
                        break
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
                    os.chmod(audio_path, stat.S_IWRITE)  # Change the file attributes to writable
                    os.remove(audio_path)
                except PermissionError:
                    print(f"Permission denied: '{audio_path}'. The file might be open, read-only or the script might not have the necessary permissions.")

        else:
            print(f"'{folder_path}' no es un directorio válido.")

        #! Cambiar el (Full album) por el ep, compilacion o cualquiera segun sea la epoca
        if audio_genres and audio_years and band_names and album_names:
            text = f"{band_names[0]} - {album_names[0]} (Full Album)\n\nGenre: {audio_genres[0]}\nYear: {audio_years[0]}\n\n"
        else:
            text = "Unknown - Unknown\nGenre: Unknown\nYear: Unknown\n\n"
        total_duration = 0

        for i, audio in enumerate(audios):
            minutes, seconds = divmod(total_duration, 60)
            if i < len(audios):
                total_duration += audio_durations[i]
                text += f"{i+1} - {audio_names[i]} ({int(minutes):02d}:{int(seconds):02d})\n"

        with open(os.path.join(folder_path, f"{folder_name}.txt"), "w", encoding='utf-8') as f:
            f.write(text)

        print(folder_name, "completado")

# Recorre todos los directorios en la ruta principal        
folder_names = os.listdir(main_dir_path)

# Crea un pool de procesos
with Pool(cpu_count()) as p:
    p.map(process_folders, folder_names)