import os
import shutil

root_dir = "D:\\01_edicion_automatizada\\audio_scripts"
upload_dir = "D:\\01_edicion_automatizada\\upload_video"
after_effects_dir = "C:\\Users\\banar\\Desktop\\save_after_effects"
premier_pro_dir = "C:\\Users\\banar\\Desktop\\save_premier_pro"

# Process each subdirectory in the root directory
for dirpath, dirnames, filenames in os.walk(root_dir):
    for dirname in dirnames:
        subdir_path = os.path.join(dirpath, dirname)
        subdir_files = os.listdir(subdir_path)
        
        # Check if the subdirectory contains at least one .mp4 file
        if any(filename.endswith('.mp4') for filename in subdir_files):
            # Print the subdirectory name and the names of the files in the subdirectory
            print(f"\nSubdirectory: {dirname}")
            for filename in subdir_files:
                print(f"  - {filename}")
            
            # Ask the user to confirm the move
            confirm = input("\nType 'YES' to move this subdirectory to the upload directory: ")
            if confirm.upper() == 'YES':
                # Move the subdirectory to the upload directory
                shutil.move(subdir_path, os.path.join(upload_dir, dirname))

# Process After Effects directory
for filename in os.listdir(after_effects_dir):
    if filename.endswith('.aep'):
        # Remove the extension and the last two characters from the filename
        filename_without_extension = filename[:-6]
        if filename_without_extension == os.path.basename(after_effects_dir):
            print(f"\nFile: {filename}")
            confirm = input("\nType 'YES' to move this file to the upload directory: ")
            if confirm.upper() == 'YES':
                shutil.move(os.path.join(after_effects_dir, filename), os.path.join(upload_dir, filename))

# Process Premier Pro directory
for filename in os.listdir(premier_pro_dir):
    if filename.endswith('.prproj'):
        # Remove the extension and the last two characters from the filename
        filename_without_extension = os.path.splitext(filename)[0]
        if filename_without_extension == os.path.basename(premier_pro_dir):
            print(f"\nFile: {filename}")
            confirm = input("\nType 'YES' to move this file to the upload directory: ")
            if confirm.upper() == 'YES':
                shutil.move(os.path.join(premier_pro_dir, filename), os.path.join(upload_dir, filename))