import os
import shutil

def contar_archivos_en_subdirectorios(directorio, directorio_destino):
    for ruta, dirs, archivos in os.walk(directorio):
        contador = len(archivos)
        if contador == 3:
            print(f'{contador} {ruta}')
            shutil.move(ruta, os.path.join(directorio_destino, os.path.basename(ruta)))

contar_archivos_en_subdirectorios('D:\\01_edicion_automatizada\\audio_scripts', 'D:\\01_edicion_automatizada\\Ya')