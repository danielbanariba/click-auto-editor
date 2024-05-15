from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import os
import time
import requests
from PIL import Image
from io import BytesIO

directorio = "D:\\01_edicion_automatizada\\no_tienen_portada"

def extraer_primer_parrafo(archivo_txt):
    with open(archivo_txt, 'r', encoding='utf-8') as archivo:
        primer_parrafo = archivo.readline().strip()
    return primer_parrafo

def descargar_imagen(url, ruta_destino):
    respuesta = requests.get(url)
    imagen = Image.open(BytesIO(respuesta.content))
    imagen.save(ruta_destino)

def descargar_primera_imagen(directorio):
    for carpeta, _, archivos in os.walk(directorio):
        for archivo in archivos:
            if archivo.endswith('.txt'):
                ruta_completa = os.path.join(carpeta, archivo)
                texto_busqueda = extraer_primer_parrafo(ruta_completa)
                
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
                driver.get("https://images.google.com/")
                
                caja_busqueda = driver.find_element(By.NAME, "q")
                caja_busqueda.send_keys(texto_busqueda + Keys.RETURN)
                
                time.sleep(5)  # Ajustar según la velocidad de conexión
                imagenes = driver.find_elements(By.CSS_SELECTOR, "img.rg_i.Q4LuWd")
                resoluciones = []
                urls = []

                for imagen in imagenes[:4]:  # Limitar a las primeras 4 imágenes
                    imagen.click()
                    time.sleep(2)  # Esperar a que la imagen grande se cargue
                    imagen_grande = driver.find_element(By.CSS_SELECTOR, "img.n3VNCb")  # Selector para la imagen grande
                    urls.append(imagen_grande.get_attribute('src'))
                    ancho = imagen_grande.get_attribute('naturalWidth')
                    alto = imagen_grande.get_attribute('naturalHeight')
                    resoluciones.append(int(ancho) * int(alto))

                if resoluciones:
                    # Seleccionar la URL de la imagen con mayor resolución
                    url_max_res = urls[resoluciones.index(max(resoluciones))]
                    # Descargar la imagen seleccionada
                    ruta_destino = f"{directorio}\\imagen_descargada.jpg"  # Ajusta esta ruta
                    descargar_imagen(url_max_res, ruta_destino)

                driver.quit()

descargar_primera_imagen(directorio)