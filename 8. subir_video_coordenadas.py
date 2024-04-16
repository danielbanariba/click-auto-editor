import os
import pyautogui
import time
import random
import webbrowser
import shutil

root_dir = "D:\\01_edicion_automatizada\\audio_scripts"
upload_dir = "D:\\01_edicion_automatizada\\upload_video"
url_upload = 'https://www.youtube.com/upload'

# Create a list of directories that contain at least one .mp4 file
dirs_with_videos = [dirpath for dirpath, dirnames, filenames in os.walk(root_dir) if any(filename.endswith('.mp4') for filename in filenames)]

# Primera parte: Abrir 7 ventanas en blanco en el navegador
for _ in range(7):
    webbrowser.open_new_tab(url_upload)

# Process only 3 videos
for _ in range(3):
    # Select a random directory
    dirpath = random.choice(dirs_with_videos)
    filenames = os.listdir(dirpath)

    # Move the entire directory to the upload directory
    shutil.move(dirpath, os.path.join(upload_dir, os.path.basename(dirpath)))

    # Update the dirpath to the new location
    dirpath = os.path.join(upload_dir, os.path.basename(dirpath))
    filenames = os.listdir(dirpath)  # Update the filenames list to reflect the new directory location

    for filename in filenames:
        if filename.endswith('.mp4'):
            print(os.path.join(dirpath, filename))
            for txt_filename in filenames:
                if txt_filename.endswith('.txt'):
                    with open(os.path.join(dirpath, txt_filename), 'r', encoding='utf-8') as f:
                        content = f.read()

                    paragraphs = content.split('\n\n')
                    titulo_video = paragraphs[0]
                    descripcion_video = '\n\n'.join(paragraphs[1:])

                    # Segunda parte: Abrir Una nueva pestanna
                    pyautogui.click(3594, 18)
                    time.sleep(5)
                    pyautogui.click(3231, 64)
                    pyautogui.write(url_upload)
                    pyautogui.press('enter')
                    time.sleep(5)
                
                    #Tercera parte: Subir el video
                    pyautogui.click(2881, 505)
                    time.sleep(5)
                    # Update the path to the new location of the video file
                    pyautogui.write(os.path.join(dirpath, filename))
                    pyautogui.press('enter')
                    time.sleep(5)
                    
                    # Cuarta parte: Poner el titulo al video
                    pyautogui.click(2959, 369)
                    pyautogui.hotkey('ctrl', 'a')
                    pyautogui.write(titulo_video)
                    time.sleep(1)
                    pyautogui.click(2713, 527)
                    pyautogui.write(descripcion_video)
                    time.sleep(2)
                    
                    #Quita parte: Publicar el video
                    #No se como putas voy hacer aqui, pero aja xd
        