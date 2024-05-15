import os

# Nombres base de archivos para eliminar
files_to_delete = ["DeathGrindClub - The Most Brutal Music In The World!", 
                   "back.jpg", "cd.jpg", "cover's cd.jpg", "DeathGrindClub", 
                   "DeathGrind.Club", "TechnicalDeathMetal", 
                   "DeathGrindClub • The Most Brutal Music In The World!", 
                   "DeathGrindClub", "DeathGrindClub ⋅ The Most Brutal Music In The World!", 
                   ".DS_Store", "desktop.ini", "Thumbs.db", "DeathGrindClub ⋅ The Most Brutal Music In The World!", 
                   "DeathGrindClub ⋅ The Most Brutal Music In The World!.url"]

def delete_specific_files(root_dir):
    # Recorrer cada carpeta en el directorio raíz
    for foldername, _, filenames in os.walk(root_dir):
        # Para cada archivo en cada carpeta
        for filename in filenames:
            # Si el nombre base del archivo está en la lista de archivos para eliminar
            if os.path.splitext(filename)[0] in files_to_delete:
                # Construir la ruta completa al archivo
                file_path = os.path.join(foldername, filename)
                # Eliminar el archivo
                try:
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
                except PermissionError:
                    print(f"Permission denied when trying to delete {file_path}. The file might be open in another program, read-only, or the script might not have the necessary permissions.")