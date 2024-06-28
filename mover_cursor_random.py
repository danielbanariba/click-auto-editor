# Este programa lo hago con el fin de hacer que la computadora no baje su rendimiento por inactividad

import random
import pyautogui
import time
import keyboard

# Obtener las dimensiones de la pantalla
ancho_pantalla, altura_pantalla = pyautogui.size()

# Bucle infinito para mover el cursor aleatoriamente
while True:
    # Comprobar si la tecla 'q' ha sido presionada para salir
    if keyboard.is_pressed('q'):  # Si presionas 'q', se rompe el bucle
        print("Deteniendo el script.")
        break

    # Generar coordenadas aleatorias dentro de los límites de la pantalla
    x = random.randint(0, ancho_pantalla)
    y = random.randint(0, altura_pantalla)
    
    # Mover el cursor a las coordenadas generadas
    pyautogui.moveTo(x, y, duration=1)
    
    # Esperar un tiempo aleatorio entre movimientos para simular actividad humana
    time.sleep(random.randint(1, 10))