import pyautogui
import time
import keyboard
import threading
#------------------------------------------
continuar1 = 3287, 882
continuar2 = 3287, 930
licencia = 2463, 564
seleccionar_cancion = 3208, 714
accepto_los_terminos = 2433, 772
impugnar = 2763, 428
informacion_de_tu_licencia = 2640, 467
click_firma = 2506, 818
check1 = 2438, 620
check2 = 2435, 670
check3 = 2430, 728
cerrar = 3115, 685
cerrar2 = 3120, 690

mensaje = "Hola nuevamente! Lastima que rechazaron mi solicitud, la razon por la cual creo que deberian de retirar el reclamo es que afecta directamente mi canal, y el alcance que puede llegar los videos hacia mas gente, y de esa manera poder llegar a mas personas, y que puedan disfrutar de la musica que ustedes crean, espero que puedan reconsiderar su decision, saludos desde Honduras!"

firma = "Daniel Alejandro Barrientos Anariba"
#------------------------------------------
#------------------------------------------
apelar_abajo = 2542, 760
apelar_arriba = 2596, 415
entiendo_los_riesgos = 2444, 845
nombre_y_apellido_coordenadas = 2540, 501
nombre_y_apellido = "Daniel Alejandro Barrientos Anariba"
dirreccion_postal_coordenadas = 3045, 498
dirreccion_postal = "Honduras, Francisco Morazan, Comayaguela, Districto Central, Residencial la Ca#ada, Bloque BH, Casa 6312"
correo_electronico_coordenadas = 2535, 581
correo_electronico = "banaribad@gmail.com"
ciudad_coordenadas = 2966, 582
ciudad = "Tegucigalpa"
pais_coordenadas = 2689, 647
pais = "H"
pais_coordenadas2 = 2504, 408
Departamento_coordenadas = 2983, 659
Departamento = "Francisco Morazan"
codigo_postal_coordenadas = 3206, 658
codigo_postal = "12101"

# num = int(input("Cuantas canciones deseas apelar? "))

#time.sleep(10)

while True:
    time.sleep(8)
    if keyboard.is_pressed('p'):
        break
    pyautogui.click(seleccionar_cancion)  
    time.sleep(1)
    pyautogui.click(apelar_arriba)
    pyautogui.click(apelar_abajo)
    pyautogui.click(continuar1)
    pyautogui.click(3184, 708)
    time.sleep(1)
    pyautogui.scroll(-1000)
    time.sleep(1)
    pyautogui.click(entiendo_los_riesgos)
    pyautogui.click(continuar2)
    time.sleep(1)
    pyautogui.click(nombre_y_apellido_coordenadas)
    pyautogui.write(nombre_y_apellido)
    pyautogui.click(dirreccion_postal_coordenadas)
    pyautogui.write(dirreccion_postal)
    pyautogui.click(correo_electronico_coordenadas)
    pyautogui.write(correo_electronico)
    pyautogui.click(ciudad_coordenadas)
    pyautogui.write(ciudad)
    time.sleep(1)
    pyautogui.click(pais_coordenadas)
    time.sleep(1)
    pyautogui.write(pais)
    pyautogui.click(pais_coordenadas2)
    time.sleep(1)
    pyautogui.click(Departamento_coordenadas)
    pyautogui.write(Departamento)
    pyautogui.click(codigo_postal_coordenadas)
    pyautogui.write(codigo_postal)
    pyautogui.click(continuar2)
    time.sleep(1)
    pyautogui.write(mensaje)
    time.sleep(1)
    pyautogui.click(check1)
    pyautogui.click(check2)
    pyautogui.click(check3)
    pyautogui.click(click_firma)
    pyautogui.write(firma)
    time.sleep(1)
    pyautogui.click(continuar2)
    time.sleep(4)
    pyautogui.click(cerrar)
    # time.sleep(3)