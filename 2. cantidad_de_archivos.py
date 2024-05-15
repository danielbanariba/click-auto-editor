import os
import shutil

ruta_principal = 'D:\\01_edicion_automatizada\\audio_scripts'
ruta_temporal = 'D:\\01_edicion_automatizada\\Ya'
ruta_sin_portada = 'D:\\01_edicion_automatizada\\no_tienen_portada'
ruta_raro = 'D:\\01_edicion_automatizada\\raro'
ruta_volver_a_buscar = 'D:\\01_edicion_automatizada\\02_volver_a_buscar'

def contar_archivos_en_subdirectorios(directorio, directorio_destino, directorio_sin_portada, directorio_raro, directorio_volver_a_buscar):
    for ruta, dirs, archivos in os.walk(directorio):
        # Verifica si 'audio_script' está en alguno de los nombres de archivo
        contiene_audio_script = any('audio_script' in archivo for archivo in archivos)
        # Continúa solo si la carpeta contiene un archivo 'audio_script'
        if contiene_audio_script:
            contador = len(archivos)
            # Cuando la carpeta tiene 3 archivos es porque esta completo
            if contador == 3:
                print(f'{contador} archivos en {ruta}')
                shutil.move(ruta, os.path.join(directorio_destino, os.path.basename(ruta)))
            # Cuando la carpeta tiene 2 archivos es porque no tiene portada y hay que buscarle una portada
            elif contador == 2:
                print(f'{contador} archivos (sin portada) en {ruta}')
                shutil.move(ruta, os.path.join(directorio_sin_portada, os.path.basename(ruta)))
            # Cuando la carpeta tiene 1 archivo o mas de 4 archivos es porque es esta rara la vaina
            elif contador == 1 or contador >= 4:
                print(f'{contador} archivos (raro) en {ruta}')
                shutil.move(ruta, os.path.join(directorio_raro, os.path.basename(ruta)))
            # Cuando la carpeta tiene 0 archivos, tengo que volver a descargar los archivos
            elif contador == 0:
                print(f'{contador} archivos (vacío) en {ruta}')
                shutil.move(ruta, os.path.join(directorio_volver_a_buscar, os.path.basename(ruta)))

contar_archivos_en_subdirectorios(ruta_principal, ruta_temporal, ruta_sin_portada, ruta_raro, ruta_volver_a_buscar)