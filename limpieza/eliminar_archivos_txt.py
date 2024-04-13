import os

# Extensiones de archivos para eliminar
extensions_to_delete = [".pdf", ".txt", ".log", ".url"]

def eliminar_archivos_txt(root_dir):
    # Variable para rastrear si se ha eliminado algún archivo
    file_deleted = False

    # Recorrer cada carpeta en el directorio raíz
    for foldername, _, filenames in os.walk(root_dir):
        # Para cada archivo en cada carpeta
        for filename in filenames:
            # Si la extensión del archivo está en la lista de extensiones para eliminar
            if os.path.splitext(filename)[1] in extensions_to_delete:
                # Construir la ruta completa al archivo
                file_path = os.path.join(foldername, filename)
                # Eliminar el archivo
                try:
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
                    file_deleted = True
                except PermissionError:
                    print(f"Permission denied when trying to delete {file_path}. The file might be open in another program, read-only, or the script might not have the necessary permissions.")

    # Si no se ha eliminado ningún archivo, imprimir un mensaje
    if not file_deleted:
        print("No files found to delete.")