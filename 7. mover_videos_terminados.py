import os
import shutil

# Define las rutas de las carpetas
folder_path = "E:\\01_edicion_automatizada\\audio_scripts"
save_premier_pro_path = "C:\\Users\\banar\\Desktop\\save_premier_pro"
save_after_effects_path = "C:\\Users\\banar\\Desktop\\save_after_effects"
upload_video_path = "E:\\01_edicion_automatizada\\upload_video"

# Recorre todas las subcarpetas y archivos en la carpeta de origen
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Comprueba si la extensión del archivo es .mp4
        if file.endswith(".mp4"):
            # Obtiene la parte del nombre del archivo antes del punto
            name_before_dot = file.split('.')[0]
            # Elimina los últimos dos caracteres
            shortened_name = name_before_dot[:-2]

            # Recorre todos los archivos en las carpetas de destino
            for save_folder in [save_premier_pro_path, save_after_effects_path]:
                for save_file in os.listdir(save_folder):
                    # Comprueba si el nombre del archivo coincide con el nombre del archivo en la carpeta de destino
                    # Quita diferentes cantidades de caracteres para los nombres de archivos en diferentes carpetas
                    if save_folder == save_premier_pro_path:
                        if save_file.split('.')[0][:-4] == shortened_name:
                            # Mueve el archivo a la carpeta correspondiente
                            shutil.move(os.path.join(save_folder, save_file), os.path.join(root, save_file))
                    elif save_folder == save_after_effects_path:
                        if save_file.split('.')[0][:-2] == shortened_name:
                            # Mueve el archivo a la carpeta correspondiente
                            shutil.move(os.path.join(save_folder, save_file), os.path.join(root, save_file))

            # Mueve la carpeta que contiene el video a la carpeta de carga de video
            shutil.move(root, upload_video_path)