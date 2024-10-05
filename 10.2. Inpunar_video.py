import pyautogui
import time
import random
import logging
from PIL import Image

# Configuración de logging
logging.basicConfig(filename='impugnaciones.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# Rutas de las imágenes para reconocimiento
SELECCIONAR_ACCION = 'seleccionar_accion.png'
IMPUGNACION_EN_PROCESO = 'impugnacion_en_proceso.png'

# Coordenadas y mensajes (ajusta según tus necesidades)
COORDENADAS = {
    'continuar1': (3287, 882),
    'continuar2': (3287, 930),
    'licencia': (2463, 564),
    'impugnar': (2763, 428),
    'impugnar2': (2571, 762),
    'info_licencia': (2640, 467),
    'firma': (2506, 818),
    'check1': (2437, 653),
    'check2': (2438, 685),
    'check3': (2434, 725),
    'cerrar': (3115, 685),
}

MENSAJE = """Hola! Lastimosamente no pude conseguir los permisos correspondientes del álbum debido a que no encontré correo de algún representante, integrante de la banda o de la banda misma para poder comunicarme con ustedes y hacer la solicitud, por eso mando este mensaje, para pedirles permiso de poder subir su música a mi canal de YouTube, estaré complacido si se comunican conmigo y podamos llegar a un acuerdo! saludos desde Honduras!"""

FIRMA = "Daniel Alejandro Barrientos Anariba"

def localizar_imagen(imagen):
    """Localiza una imagen en la pantalla sin usar OpenCV."""
    try:
        posicion = pyautogui.locateOnScreen(imagen)
        if posicion:
            return pyautogui.center(posicion)
        return None
    except pyautogui.ImageNotFoundException:
        return None

def hacer_clic(coordenadas):
    """Realiza un clic en las coordenadas especificadas."""
    pyautogui.click(coordenadas)
    time.sleep(random.uniform(0.5, 1.5))

def disputar_video():
    """Proceso de disputa de un video."""
    logging.info("Iniciando proceso de disputa para un video")
    
    # Secuencia de clics para el proceso de disputa
    for accion in ['continuar1', 'impugnar', 'impugnar2', 'continuar1', 'continuar2', 
                   'continuar2', 'licencia', 'continuar2', 'info_licencia']:
        hacer_clic(COORDENADAS[accion])
    
    # Escribir el mensaje
    pyautogui.write(MENSAJE)
    time.sleep(1)
    
    # Marcar las casillas y firmar
    for check in ['check1', 'check2', 'check3', 'firma']:
        hacer_clic(COORDENADAS[check])
    
    pyautogui.write(FIRMA)
    time.sleep(1)
    
    # Finalizar y cerrar
    hacer_clic(COORDENADAS['continuar2'])
    time.sleep(4)
    hacer_clic(COORDENADAS['cerrar'])
    
    logging.info("Proceso de disputa completado para un video")

def main():
    disputas_realizadas = 0
    max_disputas = 30  # Ajusta según sea necesario
    
    while disputas_realizadas < max_disputas:
        if localizar_imagen(SELECCIONAR_ACCION):
            logging.info(f"Iniciando disputa {disputas_realizadas + 1}")
            disputar_video()
            disputas_realizadas += 1
        elif localizar_imagen(IMPUGNACION_EN_PROCESO):
            logging.info("Video ya en proceso de impugnación, saltando...")
            pyautogui.scroll(-300)  # Ajusta el valor según sea necesario
        else:
            logging.info("No se encontraron más videos para disputar")
            break
        
        # Espera aleatoria entre acciones
        time.sleep(random.uniform(2, 5))
    
    logging.info(f"Proceso completado. Total de disputas realizadas: {disputas_realizadas}")

if __name__ == "__main__":
    main()