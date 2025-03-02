import os
import shutil
import pyautogui
import time
from PIL import Image
import numpy as np
import random

#pyautogui.FAILSAFE = False;

# Definir la ruta de la carpeta que contiene los audios
main_dir_path = "E:\\01_edicion_automatizada\\audio_scripts"

# Recoge todos los directorios en la ruta principal, eliminando duplicados directamente
folders = {folder_name for folder_name in os.listdir(main_dir_path) if os.path.isdir(os.path.join(main_dir_path, folder_name))}

# Convertir el conjunto de nuevo en una lista si es necesario
folders = list(folders)

# Mezcla la lista de carpetas
random.shuffle(folders)

# Limita la lista al n de numeros
folders = folders[:150]

def get_complementary_color(r, g, b):
    # Convertir RGB a HSV
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn
    
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/diff) + 120) % 360
    else:
        h = (60 * ((r-g)/diff) + 240) % 360
    
    s = 0 if mx == 0 else (diff/mx)
    v = mx

    # Calcular el color complementario
    h = (h + 180) % 360
    
    # Convertir HSV de vuelta a RGB
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r = int((r + m) * 255)
    g = int((g + m) * 255)
    b = int((b + m) * 255)

    return r, g, b

def auto_effects():
    for folder_name in folders:
        folder_path = os.path.join(main_dir_path, folder_name)
        
        if os.path.isdir(folder_path):
            new_folder_path = folder_path.replace('–', '-')
            if folder_path != new_folder_path:
                try:
                    shutil.move(folder_path, new_folder_path)
                    folder_path = new_folder_path
                except PermissionError:
                    print(f"PermissionError: The folder {folder_path} is currently in use by another process.")
                    continue
            
            ruta = folder_path
            
            # Busca la primera imagen .jpg en el directorio
            for file_name in os.listdir(ruta):
                if file_name.endswith('.png'):
                    img_path = os.path.join(ruta, file_name)
                    
                    # Procesa la imagen
                    img = Image.open(img_path)
                    data = np.array(img)
                    
                    # Calcula el color promedio
                    if len(data.shape) == 3 and data.shape[2] in [3, 4]:
                        avg_color = data.mean(axis=(0,1))[:3]
                    elif len(data.shape) == 2:
                        avg_color = [data.mean()] * 3
                    else:
                        print(f"Unexpected image format for {file_name}")
                        continue
                    
                    # Calcula el color complementario
                    r, g, b = get_complementary_color(*avg_color)
                    
                    # Convierte el color complementario a hexadecimal
                    inside_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
                    
                    #print(f"Color complementario para {file_name}: {inside_color}")
                    
                    # Usa el color complementario
                    pyautogui.write(inside_color)
                    
                    # Solo procesa la primera imagen
                    break
            else:
                print(f"No se encontraron imágenes .png en {ruta}")
        
        #---------------------------------------------------------------------------------------------------------
            # Primera parte: Abrir Adobe After Effects
            pyautogui.press('winleft')  # abre el menú de inicio
            time.sleep(1)
            pyautogui.write('Adobe After Effects')  # escribe el nombre del programa
            time.sleep(1)
            pyautogui.press('enter')  # abre el programa
            time.sleep(14)  # espera a que el programa se abra
        
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
            time.sleep(1)
        #---------------------------------------------------------------------------------------------------------
            # Sexta parte: Ejecutar el script de generacion de keyframes por audio
            pyautogui.click(1945, 38)
            pyautogui.click(1958, 576)
            pyautogui.click(2339, 572)
            time.sleep(1)
            pyautogui.write("C:\\Users\\banar\\Desktop\\click-auto-editor\\Affter Effects\\audio_to_keyframes.jsx")
            time.sleep(1)
            pyautogui.press('enter')
            time.sleep(35)
        
        #---------------------------------------------------------------------------------------------------------
            # Septima parte: Darle movimiento a la imagen
            pyautogui.click(1945, 38)
            pyautogui.click(1958, 576)
            pyautogui.click(2339, 572)
            time.sleep(2)
            pyautogui.write("C:\\Users\\banar\\Desktop\\click-auto-editor\\Affter Effects\\imagen_movimiento.jsx")
            time.sleep(1)
            pyautogui.press('enter')
            time.sleep(1)
        
        #---------------------------------------------------------------------------------------------------------
            # Octava parte: Crear el espectro de audio
            pyautogui.click(1945, 38)
            pyautogui.click(1958, 576)
            pyautogui.click(2339, 572)
            time.sleep(1)
            pyautogui.write("C:\\Users\\banar\\Desktop\\click-auto-editor\\Affter Effects\\espectro_de_audio.jsx")
            time.sleep(1)
            pyautogui.press('enter')
            time.sleep(1)
            
        #---------------------------------------------------------------------------------------------------------
            # Novena parte: Cambiar de color en el espectro del audio
            pyautogui.click(2107, 82)
            pyautogui.click(2126, 364) # Abrimos el menu del Inside Color
            time.sleep(1)
            pyautogui.write(inside_color) # Escribimos el color en hexadecimal 
            pyautogui.press('enter')
            
            pyautogui.click(2107, 82)
            pyautogui.click(2129, 380) # Abrimos el menu del Outside Color
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
            time.sleep(4)
            
        #---------------------------------------------------------------------------------------------------------
            # Mueve la carpeta al directorio de destino después de terminar de editar
            # try:
            #     shutil.move(folder_path, os.path.join(destination_dir_path, os.path.basename(folder_path)))
            # except PermissionError:
            #     print(f"PermissionError: No se puede mover la carpeta: {folder_path} Porque un archivo dentro de esta esta siendo utilizado por otro proceso.")
            # time.sleep(2)
            #Se repite el proceso para cada carpeta en la ruta principal

if __name__ == "__main__":
    auto_effects()