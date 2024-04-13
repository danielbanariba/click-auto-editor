import os
import shutil

def delete_subfolders(root_dir):
    # Recorrer cada carpeta en el directorio raíz
    for foldername, subfolders, _ in os.walk(root_dir):
        # Para cada subcarpeta en cada carpeta
        for subfolder in subfolders:
            # Construir la ruta completa a la subcarpeta
            subfolder_path = os.path.join(foldername, subfolder)
            # Si la subcarpeta está en una carpeta de primer nivel (es decir, su carpeta padre es el directorio raíz)
            if os.path.dirname(subfolder_path) != root_dir:
                # Eliminar la subcarpeta
                try:
                    shutil.rmtree(subfolder_path)
                    print(f"Deleted {subfolder_path}")
                except PermissionError:
                    print(f"Permission denied when trying to delete {subfolder_path}. The folder might be open in another program, read-only, or the script might not have the necessary permissions.")