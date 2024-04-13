import pyautogui
import time
from pynput import mouse

class ClickListener(mouse.Listener):
    def __init__(self):
        super(ClickListener, self).__init__(on_click=self.on_click)
        self.clicked = False

    def on_click(self, x, y, button, pressed):
        if pressed:
            self.clicked = True

listener = ClickListener()
listener.start()

try:
    while True:
        if listener.clicked:
            break
        # Obtiene y muestra las coordenadas actuales del cursor
        x, y = pyautogui.position()
        print(f"Las coordenadas actuales del cursor son ({x}, {y})")
        time.sleep(0.1)  # Espera un poco antes de la próxima iteración para no sobrecargar la CPU
except KeyboardInterrupt:
    print("\nScript terminado.")
finally:
    listener.stop()