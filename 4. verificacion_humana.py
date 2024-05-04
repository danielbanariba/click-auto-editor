import os
import shutil
import webbrowser
import random
import pyautogui
import time

def get_folder_names(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

folder_names = get_folder_names('D:\\01_edicion_automatizada\\audio_scripts')

for i in range(len(folder_names)):
    folder_name = folder_names[i]
    search_query = folder_name.replace(' ', '+')
    url = f"https://www.youtube.com/results?search_query={search_query}"
    webbrowser.open_new_tab(url)

    # Espera un poco para que el navegador se abra
    time.sleep(2)

    # Mueve el cursor a la posición de la ventana de la consola y haz clic en ella
    # Deberás ajustar las coordenadas (x, y) según la posición de tu ventana de consola
    pyautogui.moveTo(973, 901) 
    pyautogui.click()

    response = input("Esta en Youtube? ")
    if response.lower() == 'yes':
        shutil.rmtree(os.path.join('D:\\01_edicion_automatizada\\audio_scripts', folder_name))
    elif response.lower() == 'no':
        shutil.move(os.path.join('D:\\01_edicion_automatizada\\audio_scripts', folder_name), 'D:\\01_edicion_automatizada\\audio_scripts')