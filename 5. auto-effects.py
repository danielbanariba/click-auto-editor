import os
import shutil
import pyautogui
import time
from PIL import Image
import numpy as np
import random

# Definir el color del espectro de audio
inside_color = '#000000'

# Definir la ruta de la carpeta que contiene los audios
main_dir_path = "E:\\01_edicion_automatizada\\audio_scripts"

# Definir la ruta del directorio de destino
destination_dir_path = "E:\\01_edicion_automatizada\\after_effects_terminado"

# Recorre todos los directorios en la ruta principal
for folder_name in os.listdir(main_dir_path):
    folder_path = os.path.join(main_dir_path, folder_name)

    if os.path.isdir(folder_path):
        new_folder_path = folder_path.replace('–', '-')
        # Mueve el directorio si el nombre ha cambiado
        if folder_path != new_folder_path:
            try:
                shutil.move(folder_path, new_folder_path)
                folder_path = new_folder_path
            except PermissionError:
                print(f"PermissionError: The folder {folder_path} is currently in use by another process.")
                continue

        # Use the folder_path as the ruta in your script
        ruta = folder_path

        # Busca la imagen .jpg en el directorio
        for file_name in os.listdir(ruta):
            if file_name.endswith('.jpg'):
                img_path = os.path.join(ruta, file_name)

                # Abre la imagen
                img = Image.open(img_path)

                # Convierte la imagen a un array de numpy
                data = np.array(img)

                # Verifica el número de dimensiones de la matriz
                if len(data.shape) == 3:
                    # Verifica el número de canales de color
                    if data.shape[2] == 3:
                        # Calcula el promedio de los colores
                        r, g, b = data.mean(axis=(0, 1))
                    elif data.shape[2] == 4:
                        # Calcula el promedio de los colores y el canal alfa
                        r, g, b, a = data.mean(axis=(0, 1))
                    else:
                        raise ValueError(f'Unexpected number of channels: {data.shape[2]}')
                elif len(data.shape) == 2:
                    # Calcula un solo promedio para una imagen en escala de grises
                    r = g = b = data.mean()
                else:
                    raise ValueError(f'Unexpected number of dimensions: {len(data.shape)}')
                
                # Convierte el color promedio a hexadecimal
                inside_color = '#{:02x}{:02x}{:02x}'.format(int(r), int(g), int(b))
    
    #---------------------------------------------------------------------------------------------------------
        # Primera parte: Abrir Adobe After Effects
        pyautogui.press('winleft')  # abre el menú de inicio
        time.sleep(1)
        pyautogui.write('Adobe After Effects')  # escribe el nombre del programa
        time.sleep(1)
        pyautogui.press('enter')  # abre el programa
        time.sleep(20)  # espera a que el programa se abra
    
    # ---------------------------------------------------------------------------------------------------------
        # Segunda parte: Crear un nuevo proyecto
        pyautogui.click(2000, 193) # hace clic en "New Project"
        time.sleep(2)
        pyautogui.click(3020, 397)# Nueva Composicion con Footage
        time.sleep(2)
    
    #---------------------------------------------------------------------------------------------------------
        # Tercera parte: Importar los archivos que se va a editar
        pyautogui.click(2559, 100)
        pyautogui.write(ruta) # Ruta de la carpeta que contiene los archivos a editar
        time.sleep(1)
        pyautogui.press('enter')  # confirma la nueva direccion
        time.sleep(1)
        pyautogui.click(2721, 274)
        pyautogui.hotkey('ctrl', 'a') #seleccionar todos los archivos en la carpeta
        time.sleep(1)
        pyautogui.keyDown('ctrl') # No selecionar el archivo txt
        time.sleep(1)
        pyautogui.click(2262, 218) 
        pyautogui.keyUp('ctrl')
        pyautogui.press('enter')
        time.sleep(2)
        pyautogui.press('enter')
    
    #---------------------------------------------------------------------------------------------------------
        # Cuarta parte: Eliminar los archivos de la composicion
        pyautogui.mouseDown(2133, 903)
        pyautogui.moveTo(2134, 853, duration=1)
        pyautogui.mouseUp()
        time.sleep(1)
        pyautogui.hotkey('ctrl', 'x') # Elimina los archivos de la composicion ya que el suprimir no funciona
        time.sleep(1)
    
    #---------------------------------------------------------------------------------------------------------
        # Quinta parte: Llevar los archivos a la composicion
        pyautogui.mouseDown(1995, 336)
        pyautogui.moveTo(2013, 282, duration=1)
        pyautogui.mouseUp()
        time.sleep(1)
        pyautogui.mouseDown(2013, 282)  
        pyautogui.moveTo(2127, 864, duration=1)  
        pyautogui.mouseUp()  
    
    #---------------------------------------------------------------------------------------------------------
        # Sexta parte: Ejecutar el script de generacion de keyframes por audio
        pyautogui.click(1945, 38)
        pyautogui.click(1984, 541)
        pyautogui.click(2317, 539)
        time.sleep(1)
        pyautogui.write("C:\\Users\\banar\\Desktop\\click-auto-editor\\Affter Effects\\audio_to_keyframes.jsx")
        time.sleep(1)
        pyautogui.press('enter')
        time.sleep(60)
    
    #---------------------------------------------------------------------------------------------------------
        # Septima parte: Darle movimiento a la imagen
        pyautogui.click(1945, 38)
        pyautogui.click(1984, 541)
        pyautogui.click(2317, 539)
        time.sleep(2)
        pyautogui.write("C:\\Users\\banar\\Desktop\\click-auto-editor\\Affter Effects\\imagen_movimiento.jsx")
        time.sleep(1)
        pyautogui.press('enter')
        time.sleep(1)
    
    #---------------------------------------------------------------------------------------------------------
        # Octava parte: Crear el espectro de audio
        pyautogui.click(1945, 38)
        pyautogui.click(1984, 541)
        pyautogui.click(2317, 539)
        time.sleep(1)
        pyautogui.write("C:\\Users\\banar\\Desktop\\click-auto-editor\\Affter Effects\\espectro_de_audio.jsx")
        time.sleep(1)
        pyautogui.press('enter')
        time.sleep(1)
        
    #---------------------------------------------------------------------------------------------------------
        # Novena parte: Cambiar de color en el espectro del audio
        pyautogui.click(2107, 82)
        pyautogui.click(2106, 363) # Abrimos el menu del Inside Color
        time.sleep(1)
        pyautogui.write(inside_color) # Escribimos el color en hexadecimal 
        pyautogui.press('enter')
        
        pyautogui.click(2107, 82)
        pyautogui.click(2112, 382) # Abrimos el menu del Outside Color
        time.sleep(1)
        pyautogui.write(inside_color) # Escribimos el color en hexadecimal 
        pyautogui.press('enter')
        pyautogui.click(2625, 1002)
        
    #---------------------------------------------------------------------------------------------------------    
        # Decima parte: Guardar el proyecto
        pyautogui.hotkey('ctrl', 'shift', 's')
        time.sleep(1)
        random_numbers = str(random.randint(0, 9)) + str(random.randint(0, 9))
        pyautogui.write(folder_name + random_numbers)
        time.sleep(1)
        pyautogui.click(2506, 108)
        pyautogui.write("C:\\Users\\banar\\Desktop\\save_after_effects")
        time.sleep(1)
        pyautogui.press('enter')
        time.sleep(1)
        pyautogui.press('enter')
        time.sleep(1)
        print("El proyecto " + folder_name + " ha sido editado excitosamente")
        pyautogui.click(2722, 547)
        time.sleep(1)

    #---------------------------------------------------------------------------------------------------------
        #Ultima parte: Cerrar Adobe After Effects
        pyautogui.click(3809, 8)
        #pyautogui.hotkey('alt', 'f4')
        time.sleep(8)
        
    #---------------------------------------------------------------------------------------------------------
        # Mueve la carpeta al directorio de destino después de terminar de editar
        try:
            shutil.move(folder_path, os.path.join(destination_dir_path, os.path.basename(folder_path)))
        except PermissionError:
            print(f"PermissionError: No se puede mover la carpeta: {folder_path} Porque un archivo dentro de esta esta siendo utilizado por otro proceso.")
        time.sleep(3)
        #Se repite el proceso para cada carpeta en la ruta principal