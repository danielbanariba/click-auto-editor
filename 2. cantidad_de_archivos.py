import os
import shutil

ruta_principal = 'E:\\01_edicion_automatizada\\02_juntar_audios'
ruta_temporal = 'E:\\01_edicion_automatizada\\Ya'
ruta_sin_portada = 'E:\\01_edicion_automatizada\\no_tienen_portada'
ruta_raro = 'E:\\01_edicion_automatizada\\raro'
ruta_volver_a_buscar = 'E:\\01_edicion_automatizada\\03_volver_a_buscar'

def contar_archivos_en_subdirectorios(directorio, directorio_destino, directorio_sin_portada, directorio_raro, directorio_volver_a_buscar):
    rutas_destino = []
    rutas_sin_portada = []
    rutas_raro = []
    rutas_volver_a_buscar = []

    print(f"Ruta principal: {directorio}")

    for ruta, dirs, archivos in os.walk(directorio, topdown=False):
        print(f"Revisando {ruta}...")
        print(f"Subdirectorios: {dirs}")
        contador = len(archivos)
        print(f"Encontrados {contador} archivos en {ruta}.")
        if contador == 3:
            print(f'Moviendo {ruta} a {directorio_destino}')
            rutas_destino.append(ruta)
        elif contador == 2:
            print(f"Encontrados {contador} archivos en {ruta}. Moviendo a {directorio_sin_portada}")
            rutas_sin_portada.append(ruta)
        elif contador == 1 or contador == 4:
            print(f'{contador} archivos (raro) en {ruta}')
            rutas_raro.append(ruta)
        elif contador == 0:
            print(f'{contador} archivos (vacío) en {ruta}')
            rutas_volver_a_buscar.append(ruta)

    print("Rutas destino:", rutas_destino)
    print("Rutas sin portada:", rutas_sin_portada)
    print("Rutas raro:", rutas_raro)
    print("Rutas volver a buscar:", rutas_volver_a_buscar)

    print("Moviendo archivos...")
    for ruta in rutas_destino:
        print(f"Moviendo {ruta} a {directorio_destino}")
        shutil.move(ruta, os.path.join(directorio_destino, os.path.basename(ruta)))
    for ruta in rutas_sin_portada:
        print(f"Moviendo {ruta} a {directorio_sin_portada}")
        shutil.move(ruta, os.path.join(directorio_sin_portada, os.path.basename(ruta)))

if __name__ == "__main__":
    contar_archivos_en_subdirectorios(ruta_principal, ruta_temporal, ruta_sin_portada, ruta_raro, ruta_volver_a_buscar)