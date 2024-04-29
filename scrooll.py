from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

# Inicializa el driver del navegador (en este caso, Chrome)
driver = webdriver.Chrome()

# Abre la página web
driver.get("https://deathgrind.club/posts/genres/19")

# Espera un poco para que la página cargue
time.sleep(2)

# Este bucle se ejecutará mientras haya más contenido para cargar
while True:
    # Guarda la posición del scroll antes de moverse
    last_height = driver.execute_script("return document.body.scrollHeight")

    # Ejecuta un scroll gradual hasta el final de la página
    for i in range(10):
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight/10);")
        time.sleep(0.5)

    # Espera a que se cargue el nuevo contenido
    time.sleep(200)

    # Comprueba si la posición del scroll ha cambiado
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        # Si la posición no ha cambiado, significa que no hay más contenido para cargar
        break