import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import PASS as PASS
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = Options()
options.profile = r'C:\Users\banar\AppData\Roaming\Mozilla\Firefox\Profiles\joztjb27.default'
driver = webdriver.Firefox(options=options)

def upload_video(filename, title, description):
    driver.get('https://accounts.google.com/signin')

    # Iniciar sesión en tu cuenta de Google
    driver.find_element(By.ID, 'identifierId').send_keys(PASS.CORREO)
    driver.find_element(By.ID, 'identifierNext').click()
    
    # Esperar hasta que el elemento esté presente
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, 'password')))

    driver.find_element(By.NAME, 'password').send_keys(PASS.PASSWORD)
    driver.find_element(By.ID, 'passwordNext').click()
    time.sleep(2) # Esperar a que se inicie la sesión

    # Navegar a la página de subida de YouTube
    driver.get('https://www.youtube.com/upload')

    # Subir el video
    driver.find_element(By.NAME, 'file').send_keys(filename)

    # Rellenar los detalles del video
    driver.find_element(By.NAME, 'title').send_keys(title)
    driver.find_element(By.NAME, 'description').send_keys(description)

    # Publicar el video
    driver.find_element(By.NAME, 'private').click()

    # Esperar a que se complete la subida
    while driver.find_element(By.NAME, 'upload_status').text != 'Done':
        time.sleep(10)

    driver.quit()

def upload_all_videos_in_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.mp4'):
                filename = os.path.join(root, file)
                txt_file = filename.replace('.mp4', '.txt')
                if os.path.exists(txt_file):
                    with open(txt_file, 'r') as f:
                        lines = f.readlines()
                        title = lines[0].strip() # El título es la primera línea
                        description = ''.join(lines[1:]).strip() # La descripción son todas las demás líneas
                        upload_video(filename, title, description)

upload_all_videos_in_folder('D:\\01_edicion_automatizada\\audio_scripts')