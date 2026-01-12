"""
NOTA: Este script usa pyautogui con coordenadas específicas de Windows.
En Linux, las coordenadas serán diferentes y deberás ajustarlas manualmente.
Considera usar verificación manual sin clicks automáticos en Linux.
"""

import os
import shutil
import webbrowser
import random
import time
import io
from pathlib import Path
import sys

# Importar configuración
sys.path.append(str(Path(__file__).parent))
from config import DIR_VERIFICACION, DIR_NO_TIENEN_DESCRIPCION, DIR_AUDIO_SCRIPTS

def get_folder_names(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def get_first_paragraph(file_path):
    with io.open(file_path, 'r', encoding='utf8') as f:
        return f.readline().strip()

folder_names = get_folder_names(str(DIR_VERIFICACION))
random.shuffle(folder_names)  # Mezcla las carpetas de manera aleatoria

for folder_name in folder_names:
    folder_path = os.path.join(str(DIR_VERIFICACION), folder_name)
    files = os.listdir(folder_path)
    for file in files:
        if file.endswith('.txt'):
            search_query = get_first_paragraph(os.path.join(folder_path, file)).replace(' ', '+')
            search_query = search_query.replace('\0', '')  # Remove null characters
            url = f"https://www.youtube.com/results?search_query={search_query}"
            webbrowser.open_new_tab(url)

            # Espera un poco para que el navegador se abra
            time.sleep(2)

            # En Linux, no usamos pyautogui para clicks (comentado)
            # pyautogui.moveTo(973, 901)
            # pyautogui.click()

            response = input("Esta en Youtube? (y/N/d) ")
            if response.lower() == 'y':
                shutil.rmtree(folder_path)
            elif response.lower() == 'd':
                shutil.move(folder_path, os.path.join(str(DIR_NO_TIENEN_DESCRIPCION), folder_name))
            else:
                shutil.move(folder_path, os.path.join(str(DIR_AUDIO_SCRIPTS), folder_name))