import os

def contar_archivos_en_subdirectorios(directorio):
    for ruta, dirs, archivos in os.walk(directorio):
        contador = len(archivos)
        if contador != 3:
            print(f'{contador} {ruta}')

contar_archivos_en_subdirectorios('D:\\01_limpieza_de_impurezas')