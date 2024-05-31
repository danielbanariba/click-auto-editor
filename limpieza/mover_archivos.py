import os
import shutil

def mover_archivos(main_dir_path):
    # Definir la carpeta de destino
    dest_dir_path = "E:\\01_edicion_automatizada\\no_tienen_carpetas"

    # Listar todos los archivos en el directorio principal
    for file in os.listdir(main_dir_path):
        # Construir la ruta completa al archivo
        file_path = os.path.join(main_dir_path, file)
        destination_path = os.path.join(dest_dir_path, file)

        # Solo mover el archivo si es un archivo, no un directorio
        if os.path.isfile(file_path):
            # Si el archivo ya existe en el destino, eliminarlo
            if os.path.exists(destination_path):
                os.chmod(destination_path, 0o777)  # Cambiar los permisos del archivo
                os.remove(destination_path)

            # Mover el archivo a la carpeta de destino
            shutil.move(file_path, destination_path)