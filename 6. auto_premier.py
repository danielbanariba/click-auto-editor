import time
import random
import os
import pyautogui
import psutil 
from PIL import ImageGrab 

# pyautogui.FAILSAFE = False

# Definir la ruta de la carpeta que contiene los archivos de After Effects
main_dir_path = "C:\\Users\\banar\\Desktop\\save_after_effects"
ruta_intro = "C:\\Users\\banar\\Documents\\Intro Daniel Banariba"
path_render = "E:\\01_edicion_automatizada\\audio_scripts"
premier_dir = "C:\\Users\\banar\\Desktop\\save_premier_pro"

# Obtener todos los archivos .aep en la ruta principal
aep_files = [file for file in os.listdir(main_dir_path) if file.endswith('.aep')]
processed_files = set()  # Conjunto para almacenar archivos ya procesados

def wait_for_progress_bar(x=3671, y=1070, timeout=5):
    """
    Espera hasta que la barra de progreso azul desaparezca
    x, y: coordenadas donde buscar la barra de progreso
    timeout: tiempo máximo de espera en segundos
    """
    start_time = time.time()
    blue_color = (28, 108, 198)  # Color aproximado de la barra de progreso
    tolerance = 30  # Tolerancia para la detección del color

    def is_blue_present():
        # Captura un pixel en las coordenadas especificadas
        pixel = ImageGrab.grab(bbox=(x, y, x+1, y+1))
        pixel_color = pixel.getpixel((0, 0))
        
        # Compara si el color está dentro del rango de tolerancia
        return all(abs(a - b) <= tolerance for a, b in zip(pixel_color, blue_color))

    # Espera a que aparezca la barra azul primero
    while time.time() - start_time < timeout:
        if is_blue_present():
            print("Barra de progreso detectada, esperando que termine...")
            break
        time.sleep(0.5)

    # Una vez detectada, espera a que desaparezca
    while time.time() - start_time < timeout:
        if not is_blue_present():
            time.sleep(1)  # Espera adicional para asegurarse
            return True
        time.sleep(0.5)

    print("WARNING: Timeout esperando la barra de progreso")
    return False


def is_premier_running():
    """Verifica si Adobe Premier Pro está ejecutándose"""
    for proc in psutil.process_iter(['name']):
        try:
            if 'Adobe Premiere Pro.exe' in proc.info['name']:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

def wait_for_premier_close(timeout=300):  # timeout de 5 minutos por defecto
    """Espera hasta que Premier Pro se cierre o se alcance el timeout"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not is_premier_running():
            # Esperamos 5 segundos adicionales para asegurarnos que todos los procesos relacionados se cierren
            time.sleep(5)
            return True
        time.sleep(2)  # Chequeamos cada 2 segundos
    return False

# Recorre todos los archivos en la ruta principal
def auto_premier():
    while aep_files:
        # Seleccionar un archivo .aep de forma aleatoria
        file_name = random.choice(aep_files)
        
        if file_name not in processed_files:
            processed_files.add(file_name)  # Marcar el archivo como procesado

            # Extraer el nombre del proyecto del archivo
            name_proyect = os.path.splitext(file_name)[0]

            save_premier_pro = premier_dir
        #---------------------------------------------------------------------------------------------------------
            # Primera parte: Abrir Premier Pro
            pyautogui.press('winleft')  # abre el menú de inicio
            time.sleep(1)
            pyautogui.write('Premier')  # escribe el nombre del programa
            time.sleep(1)
            pyautogui.press('enter')  # abre el programa
            time.sleep(17)  # espera a que el programa se abra
        
        #---------------------------------------------------------------------------------------------------------    
            # Segunda parte: Crear un nuevo proyecto
            pyautogui.click(1985, 196) # hace clic en "New Project"
            time.sleep(2)
            random_numbers = str(random.randint(0, 9)) + str(random.randint(0, 9))
            pyautogui.write(name_proyect + random_numbers)
            # pyautogui.click(2500, 108)
            # time.sleep(1)
            # pyautogui.click(2474, 205)
            # time.sleep(1)
            # pyautogui.click(2473, 238)
            # time.sleep(1)
            # pyautogui.click(2385, 272)
            # pyautogui.click(2568, 99)
            # time.sleep(1)
            # pyautogui.write(save_premier_pro)
            pyautogui.press('enter')
            time.sleep(1)
            pyautogui.click(2700, 554)
            time.sleep(1)
            pyautogui.click(3782, 990)
            time.sleep(2)
        
        #---------------------------------------------------------------------------------------------------------
            # Tercera parte: Importar los archivos de after effects
            pyautogui.click(2627, 283)
            time.sleep(1)
            pyautogui.click(1934, 32)
            time.sleep(1)
            pyautogui.click(2040, 531)
            # pyautogui.hotkey('ctrl', 'i') # importarmos los archivos
            time.sleep(1)
            pyautogui.click(2333, 102)
            pyautogui.write(main_dir_path) # ruta de los archivos
            pyautogui.press('enter')
            time.sleep(2)
            pyautogui.click(2175, 543)
            pyautogui.write(name_proyect + ".aep") # nombre del archivo
            time.sleep(2)
            pyautogui.press('enter')
            time.sleep(10)
            # Esta parte se repite dos veces para evitar errores
            pyautogui.click(2730, 419)
            pyautogui.click(2947, 732)
            pyautogui.click(2743, 403)
            pyautogui.click(2947, 732)
            time.sleep(2)
            pyautogui.hotkey('ctrl', 'i') # importarmos el intro del video
            # time.sleep(19)
            # Esperar a que se complete la importación
            if not wait_for_progress_bar():
                print(f"Error: La importación de {name_proyect} puede no haberse completado correctamente")
            pyautogui.click(2333, 102)
            pyautogui.write(ruta_intro) # ruta del intro
            pyautogui.press('enter')
            time.sleep(2)
            pyautogui.click(2175, 543)
            pyautogui.write("0000000000000000.mp4") # Intro del video
            time.sleep(1)
            pyautogui.press('enter')
            time.sleep(1)
            # Esta parte se repite dos veces para evitar errores
            # pyautogui.click(2735, 401)
            # pyautogui.click(2955, 733)
            # time.sleep(2)
            # pyautogui.click(3105, 717)
            # time.sleep(1)
            
            # jalamos los archivos a la linea de tiempo
            pyautogui.mouseDown(2239, 786)
            pyautogui.moveTo(2650, 859, duration=3)
            pyautogui.mouseUp()
            time.sleep(2)
            pyautogui.mouseDown(2035, 789)  
            pyautogui.moveTo(3210, 880, duration=3)  
            pyautogui.mouseUp()
            time.sleep(1)
            
            # ponemos la transicion vhs
            pyautogui.click(2473, 664)
            pyautogui.click(2492, 776)
            pyautogui.click(1980, 691)
            pyautogui.write('vhs')
            # Arreglamos la transicion vhs
            pyautogui.mouseDown(2049, 894)
            pyautogui.moveTo(3199, 871, duration=2)
            pyautogui.mouseUp()
            time.sleep(1)
            pyautogui.click(3019, 574)
            time.sleep(1)
            pyautogui.click(3072, 778)
            
            # Exportarlo a media encoder
            pyautogui.hotkey('ctrl', 'm')
            time.sleep(2)
            pyautogui.click(2424, 203) # Selecionar la ruta de guardado
            time.sleep(2)
            pyautogui.click(2569, 100) # Pone el nombre al video
            pyautogui.write((path_render + "\\" + name_proyect)[:-2])
            pyautogui.press('enter')
            time.sleep(2)
            pyautogui.click(2712, 548)
            time.sleep(2)
            pyautogui.click(2644, 146) # Pone el nombre al video (File Name)
            time.sleep(2)
            pyautogui.write(name_proyect + ".mp4")
            time.sleep(2)
            # pyautogui.click(2496, 240)# preset
            # time.sleep(2)
            # pyautogui.click(2472, 486) # 4k
            # time.sleep(2)
            pyautogui.click(3632, 1030) # Send to media encoder
            time.sleep(2)
            pyautogui.click(3811, 2) # Cerramos premier pro
            time.sleep(2)
            pyautogui.press('enter')
            
            if wait_for_premier_close():
                print("El proyecto " + name_proyect + " se ha exportado correctamente")
            else:
                print("WARNING: Timeout esperando que Premier Pro se cierre para" + {name_proyect})
            
            # El ciclo continuará automáticamente cuando Premier Pro se cierre

if __name__ == "__main__":
    auto_premier()