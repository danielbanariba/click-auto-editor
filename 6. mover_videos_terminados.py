import os
import shutil

root_dir = "D:\\01_edicion_automatizada\\audio_scripts"
upload_dir = "D:\\01_edicion_automatizada\\upload_video"

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