import os
import shutil
import time
import psutil

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

# Function to close open files in a folder
def close_open_files(folder_path):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for item in proc.open_files():
                if folder_path in item.path:
                    proc.terminate()
                    proc.wait()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

# Function to safely remove a folder, catching PermissionError
def safe_remove_folder(folder_path, retries=3, delay=5):
    for attempt in range(retries):
        try:
            close_open_files(folder_path)
            shutil.rmtree(folder_path)
            print(f"Removed folder: {folder_path}")
            break
        except PermissionError as e:
            print(f"PermissionError: Could not remove {folder_path}. File may be in use. Error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to remove {folder_path} after {retries} attempts.")

# Walk through the directory
for folder_name, subfolders, files in os.walk(directory):
    # Check if the current folder contains exactly one file
    if len(files) == 1:
        # Attempt to remove the folder and its contents
        safe_remove_folder(folder_name)