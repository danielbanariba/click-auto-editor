#ESTE SCRIPT ES UNA BASURA, lo que se me ocurre hacer es dividirlo en pedacitos mas pequenno, osea una que eliminar url, otro que elimine carpetas, y todos esos lo llamen con una funcion para yo solo invocarlos y ya

import os
from unidecode import unidecode


def eliminar_archivos_rar(root_dir):
    compression_extensions = ['.zip', '.rar', '.7z', '.tar', '.gz', '.001', '.zipx', 'zip1']

    # Directorio raíz desde donde comenzar la búsqueda
    root_dir = root_dir

    # Eliminar archivos comprimidos
    for filename in os.listdir(root_dir):
        if os.path.splitext(filename)[1] in compression_extensions:
            file_path = os.path.join(root_dir, filename)
            os.remove(file_path)
            print(f"Deleted {file_path}")