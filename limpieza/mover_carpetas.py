import os
import shutil

def mover_carpetas(root_origen, root_destino):
    # Lista todos los archivos en el directorio de origen
    files = os.listdir(root_origen)

    # Mueve cada archivo al directorio de destino
    for file in files:
        shutil.move(os.path.join(root_origen, file), root_destino)