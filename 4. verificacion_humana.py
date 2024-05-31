import os
import shutil
import webbrowser
import random
import pyautogui
import time

def get_folder_names(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

folder_names = get_folder_names('E:\\01_edicion_automatizada\\verificacion')
random.shuffle(folder_names)  # Mezcla las carpetas de manera aleatoria

for folder_name in folder_names:
    search_query = folder_name.replace(' ', '+')
    url = f"https://www.youtube.com/results?search_query={search_query}"
    webbrowser.open_new_tab(url)

    # Espera un poco para que el navegador se abra
    time.sleep(2)

    # Mueve el cursor a la posición de la ventana de la consola y haz clic en ella
    pyautogui.moveTo(973, 901) 
    pyautogui.click()

    response = input("Esta en Youtube? (y/N) ")
    if response.lower() == 'y':
        shutil.rmtree(os.path.join('E:\\01_edicion_automatizada\\verificacion', folder_name))
    else:
        shutil.move(os.path.join('E:\\01_edicion_automatizada\\verificacion', folder_name), os.path.join('E:\\01_edicion_automatizada\\audio_scripts', folder_name))