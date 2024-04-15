import os
import pyautogui
import time
import random
import webbrowser
import os

# Directorio raíz donde comenzar la búsqueda
root_dir = "D:\\01_edicion_automatizada\\audio_scripts"

url_upload = 'https://www.youtube.com/upload'

#---------------------------------------------------------------------------------------------------------
    # Primera parte: Abrir 7 ventanas en blanco en el navegador
for _ in range(7):
    webbrowser.open_new_tab(url_upload)

#TODO hacer la seleccion de los videos de manera aleatoria y que al momento de seleccionarlo lleve la carpeta a una nueva direccion que es el "D:\01_edicion_automatizada\upload_video"
# Recorrer todas las subcarpetas y archivos en el directorio raíz
for dirpath, dirnames, filenames in os.walk(root_dir):
    # Recorrer todos los archivos en la carpeta actual
    for filename in filenames:
        # Si el archivo es un .mp4, procesarlo
        if filename.endswith('.mp4'):
            print(os.path.join(dirpath, filename))

            # Buscar el archivo .txt en la misma carpeta
            for txt_filename in filenames:
                if txt_filename.endswith('.txt'):
                    # Leer el contenido del archivo .txt
                    with open(os.path.join(dirpath, txt_filename), 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Dividir el contenido en título y descripción
                    paragraphs = content.split('\n\n')  # Supone que los párrafos están separados por dos saltos de línea
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
        