import os
import pyautogui
import time
import random
import webbrowser
from datetime import datetime, timedelta
import locale

# Set the locale to Spanish
locale.setlocale(locale.LC_TIME, "es_ES.UTF-8")

# Genera una fecha aleatoria dentro del próximo mes
current_date = datetime.now()
random_days = random.randint(1, 30)  # Genera un número aleatorio entre 1 y 30
publish_date = current_date + timedelta(days=random_days)
day = publish_date.strftime("%d")
month = publish_date.strftime("%b").replace('.', '')  # Remove the period
year = publish_date.strftime("%Y")
publish_date_string = f"{day} {month} {year}"  # Convertir a formato "16 abr 2025"

# Genera una hora aleatoria en intervalos de 15 minutos
random_hour = random.randint(0, 23)  # Genera un número aleatorio entre 0 y 23
random_minute = random.choice([0, 15, 30, 45])  # Elige un número aleatorio de la lista [0, 15, 30, 45]
publish_time_string = f"{random_hour:02d}:{random_minute:02d}"  # Convertir a formato "14:30"

root_dir = "D:\\01_edicion_automatizada\\upload_video"
url_upload = 'https://www.youtube.com/upload'
pleca = ' '

# Create a list of directories that contain at least one .mp4 file
dirs_with_videos = [dirpath for dirpath, dirnames, filenames in os.walk(root_dir) if any(filename.endswith('.mp4') for filename in filenames)]

# Primera parte: Abrir 7 ventanas en blanco en el navegador
for _ in range(7):
    webbrowser.open_new_tab(pleca)
time.sleep(30)

# Process only 3 videos
for _ in range(3):
    # Select a random directory
    dirpath = random.choice(dirs_with_videos)
    filenames = os.listdir(dirpath)

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
                    time.sleep(15)
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
                    time.sleep(5)
                    
                    #Quita parte: Publicar el video
pyautogui.click(3296, 1001) #Click en siguiente
time.sleep(1)
pyautogui.click(3262, 520) #Añadir pantalla Final
time.sleep(1)
pyautogui.click(2546, 382) #Añadir elemento
time.sleep(1)
pyautogui.click(3285, 226) #Guardar
time.sleep(5)
pyautogui.click(3296, 1001) #Click en siguiente
pyautogui.click(3296, 1001) #Click en siguiente
time.sleep(1)
pyautogui.click(2942, 767) #Programar
pyautogui.click(2671, 603) #Fecha
pyautogui.click(2671, 603) 
pyautogui.hotkey('ctrl', 'a')
time.sleep(1)
pyautogui.write(publish_date_string)#Ponemos cualquier fecha aleatoria en el intervalo de 30 dias
time.sleep(1)
pyautogui.press('enter')
time.sleep(1)
pyautogui.click(2795, 604)
pyautogui.hotkey('ctrl', 'a')
pyautogui.write(publish_time_string)#Ponemos cualquier hora aleatoria con intervalo de 15 minutos
pyautogui.press('enter')
pyautogui.click(3294, 1005) #Programar