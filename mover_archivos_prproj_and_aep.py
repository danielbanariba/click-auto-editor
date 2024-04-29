import os
import shutil
import difflib

# Definir las rutas de las carpetas
video_folder = "D:\\01_edicion_automatizada\\upload_video"
ae_folder = "C:\\Users\\banar\\Desktop\\save_after_effects"
pr_folder = "C:\\Users\\banar\\Desktop\\regresar_save_after_effects"

# Obtener la lista de archivos en cada carpeta
video_files = os.listdir(video_folder)
ae_files = os.listdir(ae_folder)
pr_files = os.listdir(pr_folder)

# Para cada archivo de video
for video_file in video_files:
    print(f"Procesando el archivo de video: {video_file}")
    # Comparar el nombre del archivo de video con los nombres de los archivos de After Effects y Premiere Pro
    for ae_file in ae_files:
        # Ignorar los últimos 2 caracteres del nombre del archivo .aep
        ae_file_name = ae_file[:-2]
        print(f"Comparando {video_file} con {ae_file_name}")
        match_ratio = difflib.SequenceMatcher(None, video_file, ae_file_name).ratio()
        print(f"Coincidencia: {match_ratio}")
        if match_ratio >= 0.8:
            print(f"Encontrada coincidencia con el archivo de After Effects: {ae_file} (coincidencia: {match_ratio})")
            # Verificar si el archivo ya existe en la carpeta de destino
            if os.path.exists(os.path.join(video_folder, ae_file)):
                # Si es así, eliminarlo
                os.remove(os.path.join(video_folder, ae_file))
            # Mover el archivo de After Effects a la carpeta de videos
            shutil.move(os.path.join(ae_folder, ae_file), video_folder)
    for pr_file in pr_files:
        # Ignorar los últimos 3 caracteres del nombre del archivo .prproj
        pr_file_name = pr_file[:-3]
        print(f"Comparando {video_file} con {pr_file_name}")
        match_ratio = difflib.SequenceMatcher(None, video_file, pr_file_name).ratio()
        print(f"Coincidencia: {match_ratio}")
        if match_ratio >= 0.8:
            print(f"Encontrada coincidencia con el archivo de Premiere Pro: {pr_file} (coincidencia: {match_ratio})")
            # Verificar si el archivo ya existe en la carpeta de destino
            if os.path.exists(os.path.join(video_folder, pr_file)):
                # Si es así, eliminarlo
                os.remove(os.path.join(video_folder, pr_file))
            # Mover el archivo de Premiere Pro a la carpeta de videos
            shutil.move(os.path.join(pr_folder, pr_file), video_folder)