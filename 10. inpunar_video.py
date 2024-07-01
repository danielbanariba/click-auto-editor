import pyautogui
import time
import keyboard

continuar1 = 3287, 882
continuar2 = 3287, 930
licencia = 2463, 564
seleccionar_cancion = 3221, 694
accepto_los_terminos = 2433, 772
impugnar = 2763, 428
impugnar2 = 2571, 762
informacion_de_tu_licencia = 2640, 467
click_firma = 2506, 818
check1 = 2437, 653
check2 = 2438, 685
check3 = 2434, 725
cerrar = 3115, 685
cerrar2 = 3120, 690

mensaje = "Hola! Lastimosamente no pude conseguir los permisos correspondientes del álbum debido a que no encontré correo de algún representante, integrante de la banda o de la banda misma para poder comunicarme con ustedes y hacer la solicitud, por eso mando este mensaje, para pedirles permiso de poder subir su musica a mi canal de YouTube, estaré complacido si se comunican conmigo y podamos llegar a un acuerdo! saludos desde Honduras!"
#mensaje = "Hola, creo y reafirmo firmemente que se trata de un error, ya que solo es un ritmo normal de bateria, no es una pista de audio protegida por derechos de autor, por favor, revisenlo y eliminen la reclamación de derechos de autor. Gracias."
#mensaje = "Hola miembros de arsenal! solo queria pedirles permiso para publicar su album, ya me conocen, soy Daniel Banariba :) el que anda con ustedes a todos lados y que les anda grabando los conciertos."

#Mensaje si se trata de un error evidente de la reclamacion
#mensaje = "Hola, creo y reafirmo firmemente que se trata de un error, ya que no se trata de ninguna cancion, y si se escuchan bien todo es parte de la cancion y solo capta esa parte que no tiene nada que ver con la cancion con la que se esta reclamando, no es una pista de audio protegida por derechos de autor, por favor, revisenlo y eliminen la reclamación de derechos de autor. Gracias."

firma = "Daniel Alejandro Barrientos Anariba"


#num = int(input("Cuantas canciones deseas inpugnar? "))

for i in range(1):
    pyautogui.click(seleccionar_cancion)  
    time.sleep(1)
    pyautogui.click(continuar1) #Comentar cuando cambie el inicio
    pyautogui.click(impugnar)
    pyautogui.click(impugnar2)
    pyautogui.click(continuar1)
    pyautogui.click(continuar2)
    time.sleep(1)
    pyautogui.click(continuar2)
    pyautogui.click(licencia)
    time.sleep(1)
    pyautogui.click(continuar2)
    pyautogui.click(continuar2)
    pyautogui.click(accepto_los_terminos)
    pyautogui.click(continuar2)
    pyautogui.click(informacion_de_tu_licencia)
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
    # time.sleep(4)
