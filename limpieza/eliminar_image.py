import os
import stat

# Extensiones de archivos de imagen
image_extensions = [".jpg", ".jpeg", ".png", ".gif"]

def delete_images(root_dir):
    # Recorrer cada carpeta en el directorio raíz
    for foldername, _, filenames in os.walk(root_dir):
        # Lista para almacenar los nombres de los archivos de imagen
        image_files = []

        # Para cada archivo en cada carpeta
        for filename in filenames:
            if os.path.splitext(filename)[1] in image_extensions:
                # Si el archivo es una imagen, agregar su nombre a la lista
                image_files.append(filename)

        # Si hay más de un archivo de imagen en la carpeta
        if len(image_files) > 1:
            # Si hay un archivo cuyo nombre comienza con 'cover' (ignorando mayúsculas y minúsculas) y tiene una extensión de imagen, conservarlo y eliminar los demás
            cover_files = [file for file in image_files if file.lower().startswith('cover') and os.path.splitext(file)[1] in image_extensions]
            if cover_files:
                image_files = [file for file in image_files if file not in cover_files]
            for image_file in list(image_files):  # Crear una copia de la lista antes de recorrerla
                file_path = os.path.join(foldername, image_file)
                if os.path.exists(file_path):  # Verificar si el archivo existe antes de intentar eliminarlo
                    try:
                        os.chmod(file_path, stat.S_IWRITE)  # Change the file permissions
                        os.remove(file_path)
                        print(f"Deleted {file_path}")
                        image_files.remove(image_file)  # Actualizar la lista después de cada eliminación
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")