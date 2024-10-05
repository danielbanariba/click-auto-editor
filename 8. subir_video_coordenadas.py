import os
import pyautogui
import time
import random
import webbrowser
from datetime import datetime, timedelta
import locale
import shutil  # Import the shutil module

root_dir = "E:\\01_edicion_automatizada\\upload_video"
uploading_dir = "E:\\01_edicion_automatizada\\upload_video\\00. videos_que_se_estan_subiendo"  # Define the uploading directory
url_upload = 'https://www.youtube.com/upload'

# Create a list of directories that contain at least one .mp4 file
dirs_with_videos = [dirpath for dirpath, dirnames, filenames in os.walk(root_dir) 
                    if any(filename.endswith('.mp4') for filename in filenames) 
                    and "00. videos_que_se_estan_subiendo" not in dirpath]  # Exclude the uploading directory

num = int(input("Cuantas videos desea subir hoy? "))

# Randomly select 7 directories
selected_dirs = random.sample(dirs_with_videos, num)

# Move the selected directories to the uploading directory and update selected_dirs
for i, dirpath in enumerate(selected_dirs):
    new_dirpath = shutil.move(dirpath, uploading_dir)
    selected_dirs[i] = new_dirpath  # Update the path in selected_dirs

# Sort selected_dirs in normal order
selected_dirs.sort()

# Now, change the root directory to the uploading directory
root_dir = uploading_dir

# Primera parte: Abrir 7 ventanas en blanco en el navegador
# for _ in range(7):
#     webbrowser.open_new_tab(url_upload)
# time.sleep(30)

# Process all videos
# Select a random directory
for dirpath in selected_dirs:
    filenames = os.listdir(dirpath)
    for filename in filenames:
        if filename.endswith('.mp4'):
            print(os.path.join(dirpath, filename))
            for txt_filename in filenames:
                if txt_filename.endswith('.txt'):
                    with open(os.path.join(dirpath, txt_filename), 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Set the locale to Spanish
                    locale.setlocale(locale.LC_TIME, "es_ES.UTF-8")
                    # Establece una fecha base en el año 2025
                    base_date = datetime(2026, 8, 27)  # Fecha específica: 29 de Junio de 2026
                    day = base_date.strftime("%d")
                    month = base_date.strftime("%b").replace('.', '')  # Remove the period
                    year = base_date.strftime("%Y")
                    publish_date_string = f"{day} {month} {year}"  # Convertir a formato "16 abr 2025"
                    # Genera una hora aleatoria en intervalos de 15 minutos
                    random_hour = random.randint(0, 23)  # Genera un número aleatorio entre 0 y 23
                    random_minute = random.choice([0, 15, 30, 45])  # Elige un número aleatorio de la lista [0, 15, 30, 45]
                    publish_time_string = f"{random_hour:02d}:{random_minute:02d}"  # Convertir a formato "14:30"
                    paragraphs = content.split('\n\n')
                    titulo_video = paragraphs[0]
                    descripcion_video = '\n\n'.join(paragraphs[1:])
                    
                    # Agrega el título del video a la lista
                    with open("bandas-subidas-al-canal.txt", "a", encoding='utf-8') as f:
                        f.write((titulo_video)[:-13] + "\n")
                    # Segunda parte: Abrir Una nueva pestanna
                    pyautogui.click(3618, 19)
                    time.sleep(7)
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
                    time.sleep(7)
                    
                    #Quita parte: Publicar el video
                    pyautogui.click(3296, 1001) #Click en siguiente
                    time.sleep(1)
                    # COMENTAR ABAJO SI ME QUITAN LA MONETIZACION
                    #Activamos la monetizacion
                    pyautogui.click(2821, 445)
                    time.sleep(1)
                    pyautogui.click(2483, 443)
                    pyautogui.click(2794, 548) #Hecho
                    time.sleep(1)
                    pyautogui.click(2555, 642) #Revisar Colocacion de anuncios
                    time.sleep(1)
                    pyautogui.click(2754, 279) #Colocar anuncios automaticamente
                    time.sleep(1)
                    pyautogui.click(3170, 668) #Aceptar
                    time.sleep(1)
                    pyautogui.click(3276, 233) #Continuar
                    time.sleep(1)
                    pyautogui.click(3320, 994) #Click en siguiente
                    pyautogui.click(3131, 497) #Click de la nada
                    pyautogui.scroll(-1000)
                    time.sleep(1)
                    pyautogui.scroll(-1000)
                    time.sleep(1)
                    pyautogui.scroll(-1000)
                    time.sleep(1)
                    pyautogui.scroll(-1000)
                    time.sleep(1)
                    pyautogui.click(2464, 918) #Nada de lo anterior
                    time.sleep(1)
                    pyautogui.click(3107, 404) #Enviar calificacion
                    time.sleep(7)
                    pyautogui.click(3311, 998) #Siguiente
                    time.sleep(1)
                    # COMENTAR ARRIBA SI ME QUITAN LA MONETIZACION
                    pyautogui.click(3262, 520) #Añadir pantalla Final
                    time.sleep(3)
                    pyautogui.click(2546, 382) #Añadir elemento
                    time.sleep(1)
                    pyautogui.click(3285, 226) #Guardar
                    time.sleep(5)
                    pyautogui.click(3296, 1001) #Click en siguiente
                    time.sleep(1)
                    pyautogui.click(3296, 1001) #Click en siguiente
                    #time.sleep(15)
                    pyautogui.click(2941, 733) #Programar
                    time.sleep(1)
                    pyautogui.click(2615, 674) #Fecha
                    pyautogui.hotkey('ctrl', 'a')
                    time.sleep(1)
                    pyautogui.write(publish_date_string)#Ponemos cualquier fecha aleatoria en el intervalo de 30 dias
                    time.sleep(1)
                    pyautogui.press('enter')
                    time.sleep(1)
                    pyautogui.click(2708, 674)
                    pyautogui.hotkey('ctrl', 'a')
                    pyautogui.write(publish_time_string)#Ponemos cualquier hora aleatoria con intervalo de 15 minutos
                    pyautogui.press('enter')
                    pyautogui.click(3294, 1005) #Programar
                    time.sleep(1)
                    pyautogui.click(3041, 660)