import time
import random
import os
import pyautogui

#pyautogui.FAILSAFE = False

# Definir la ruta de la carpeta que contiene los archivos de After Effects
main_dir_path = "C:\\Users\\banar\\Desktop\\save_after_effects"
ruta_intro = "C:\\Users\\banar\\Documents\\Intro Daniel Banariba"
path_render = "E:\\01_edicion_automatizada\\audio_scripts"

# Obtener todos los archivos .aep en la ruta principal
aep_files = [file for file in os.listdir(main_dir_path) if file.endswith('.aep')]
processed_files = set()  # Conjunto para almacenar archivos ya procesados

# Recorre todos los archivos en la ruta principal
def auto_premier():
    while aep_files:
        # Seleccionar un archivo .aep de forma aleatoria
        file_name = random.choice(aep_files)
        
        if file_name not in processed_files:
            file_path = os.path.join(main_dir_path, file_name)
            processed_files.add(file_name)  # Marcar el archivo como procesado

            # Extraer el nombre del proyecto del archivo
            name_proyect = os.path.splitext(file_name)[0]

            save_premier_pro = "C:\\Users\\banar\\Desktop\\save_premier_pro"
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
            pyautogui.click(1992, 190) # hace clic en "New Project"
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
            #pyautogui.write(save_premier_pro)
            pyautogui.press('enter')
            time.sleep(1)
            pyautogui.click(2700, 554)
            time.sleep(1)
            pyautogui.click(3782, 990)
            time.sleep(2)
        
        #---------------------------------------------------------------------------------------------------------
            #Tercera parte: Importar los archivos de after effects
            pyautogui.click(2627, 283)
            pyautogui.hotkey('ctrl', 'i') # importarmos los archivos
            time.sleep(2)
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
            time.sleep(19)
            pyautogui.click(2333, 102)
            pyautogui.write(ruta_intro) # ruta del intro
            pyautogui.press('enter')
            time.sleep(2)
            pyautogui.click(2175, 543)
            pyautogui.write("0000000000000000.mp4") # Intro del video
            time.sleep(1)
            pyautogui.press('enter')
            time.sleep(1)
            # # Esta parte se repite dos veces para evitar errores
            # pyautogui.click(2735, 401)
            # pyautogui.click(2955, 733)
            # time.sleep(2)
            # pyautogui.click(3105, 717)
            # time.sleep(1)
            
            #jalamos los archivos a la linea de tiempo
            pyautogui.mouseDown(2239, 786)
            pyautogui.moveTo(2650, 859, duration=1)
            pyautogui.mouseUp()
            time.sleep(2)
            pyautogui.mouseDown(2035, 789)  
            pyautogui.moveTo(3210, 880, duration=1)  
            pyautogui.mouseUp()
            time.sleep(1)
            
            #ponemos la transicion vhs
            pyautogui.click(2473, 664)
            pyautogui.click(2492, 776)
            pyautogui.click(1980, 691)
            pyautogui.write('vhs')
            #Arreglamos la transicion vhs
            pyautogui.mouseDown(2049, 894)
            pyautogui.moveTo(3199, 871, duration=1)
            pyautogui.mouseUp()
            time.sleep(1)
            pyautogui.click(3019, 574)
            time.sleep(1)
            pyautogui.click(3072, 778)
            
            #Exportarlo a media encoder
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
            print("El proyecto " + name_proyect + " se ha exportado correctamente")
            # El ciclo se repite para el siguiente archivo
            time.sleep(90)

if __name__ == "__main__":
    auto_premier()