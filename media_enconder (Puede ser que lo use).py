import os
import pyautogui
from PIL import Image

# Define la ruta donde se encuentran tus archivos de Premier Pro
premier_pro_files_path = "C:\\Users\\banar\\Desktop\\premier_pro_files"

# Obtiene una lista de todos los archivos en esa ruta
files = os.listdir(premier_pro_files_path)

# Recorre cada archivo en la lista
for file in files:
    # Abre cada archivo con Premier Pro (esto puede requerir automatización del teclado y el ratón con pyautogui)
    # Aquí necesitarías el código para abrir el archivo con Premier Pro

    # Mueve el cursor a la ubicación del botón rojo
    pyautogui.moveTo(x, y)  # Reemplaza x, y con las coordenadas del botón rojo

    # Toma una captura de pantalla
    screenshot = pyautogui.screenshot()

    # Convierte la captura de pantalla a un objeto de imagen PIL
    img = Image.open(screenshot)

    # Obtiene el color del píxel en la posición del cursor
    color = img.getpixel((x, y))

    # Si el color del píxel es rojo (lo que indica que el botón rojo está presente), realiza la operación para agregar audio
    if color == (255, 0, 0):  # Reemplaza (255, 0, 0) con el color exacto del botón rojo
        # Aquí necesitarías el código para agregar audio al archivo
        pass