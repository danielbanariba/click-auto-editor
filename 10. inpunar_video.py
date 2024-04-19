import pyautogui
import time
import keyboard
import threading

continuar1 = 3287, 882
continuar2 = 3287, 930
licencia = 2463, 564
seleccionar_cancion = 3221, 694
accepto_los_terminos = 2433, 772
impugnar = 2763, 428
informacion_de_tu_licencia = 2640, 467
click_firma = 2506, 818
check1 = 2437, 653
check2 = 2438, 685
check3 = 2434, 725
cerrar = 3115, 685
cerrar2 = 3120, 690

mensaje = "Hola! Lastimosamente no pude conseguir los permisos correspondientes del álbum debido a que no encontré correo de algún representante, integrante de la banda o de la banda misma para poder comunicarme con ustedes y hacer la solicitud, por eso mando este mensaje, para pedirles permiso de poder subir su musica a mi canal de YouTube, estaré complacido si se comunican conmigo y podamos llegar a un acuerdo! saludos desde Honduras!"

firma = "Daniel Alejandro Barrientos Anariba"


# Resto de tu código
for i in range(10):
    pyautogui.click(seleccionar_cancion)  
    time.sleep(1)
    pyautogui.click(impugnar)
    pyautogui.click(continuar1)
    pyautogui.click(continuar2)
    pyautogui.click(licencia)
    pyautogui.click(continuar2)
    pyautogui.click(accepto_los_terminos)
    pyautogui.click(continuar2)
    pyautogui.click(informacion_de_tu_licencia)
    pyautogui.write(mensaje)
    time.sleep(3)
    pyautogui.click(check1)
    pyautogui.click(check2)
    pyautogui.click(check3)
    pyautogui.click(click_firma)
    pyautogui.write(firma)
    time.sleep(1)
    pyautogui.click(continuar2)
    time.sleep(5)
    pyautogui.click(cerrar)
    time.sleep(5)