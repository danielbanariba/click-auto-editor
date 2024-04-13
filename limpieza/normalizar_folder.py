import os
import unicodedata

def normalize_folder_names(directory):
    for filename in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, filename)):
            normalized_name = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode()
            try:
                os.rename(os.path.join(directory, filename), os.path.join(directory, normalized_name))
            except Exception as e:
                print(f"Error renaming {filename} to {normalized_name}: {e}")