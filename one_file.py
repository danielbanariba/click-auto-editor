# Elimina carpetas que contienen un solo archivo, descomnetar lo que esta comentado para poder eliminarlo

import os
import shutil

import time

# Define the directory to analyze
directory = "E:\\01_edicion_automatizada\\audio_scripts"

# List to hold the names of folders with exactly one file
folders_with_one_file = []


# Walk through the directory
for folder_name, subfolders, files in os.walk(directory):
	# Check if the current folder contains exactly one file
	if len(files) == 1:
		folders_with_one_file.append(folder_name)
		print(folder_name)  # Mover esta línea aquí imprime solo las carpetas con un archivo

# Descomentar para eliminar las carpetas con un solo archivo
# Function to safely remove a folder, catching PermissionError
def safe_remove_folder(folder_path):
	try:
		shutil.rmtree(folder_path)
		print(f"Removed folder: {folder_path}")
	except PermissionError as e:
		print(f"PermissionError: Could not remove {folder_path}. File may be in use. Error: {e}")

# Walk through the directory
for folder_name, subfolders, files in os.walk(directory):
	# Check if the current folder contains exactly one file
	if len(files) == 1:
		# Attempt to remove the folder and its contents
		safe_remove_folder(folder_name)
