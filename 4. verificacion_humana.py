import os
import shutil
import webbrowser
import random
import pyautogui
import time
import io

def get_folder_names(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def get_first_paragraph(file_path):
    with io.open(file_path, 'r', encoding='utf8') as f:
        return f.readline().strip()

folder_names = get_folder_names('E:\\01_edicion_automatizada\\verificacion')
random.shuffle(folder_names)  # Mezcla las carpetas de manera aleatoria

for folder_name in folder_names:
    folder_path = os.path.join('E:\\01_edicion_automatizada\\verificacion', folder_name)
    files = os.listdir(folder_path)
    for file in files:
        if file.endswith('.txt'):
            search_query = get_first_paragraph(os.path.join(folder_path, file)).replace(' ', '+')
            url = f"https://www.youtube.com/results?search_query={search_query}"
            webbrowser.open_new_tab(url)

            # Espera un poco para que el navegador se abra
            time.sleep(2)

            # Mueve el cursor a la posición de la ventana de la consola y haz clic en ella
            pyautogui.moveTo(973, 901) 
            pyautogui.click()

            response = input("Esta en Youtube? (y/N/d) ")
            if response.lower() == 'y':
                shutil.rmtree(folder_path)
            elif response.lower() == 'd':
                shutil.move(folder_path, os.path.join('E:\\01_edicion_automatizada\\no_tienen_descripcion', folder_name))
            else:
                shutil.move(folder_path, os.path.join('E:\\01_edicion_automatizada\\audio_scripts', folder_name))