from mutagen.easyid3 import EasyID3
from mutagen.flac import FLAC
from mutagen import File
import os
import shutil

# Directorios de origen y destino
dir_origen = "E:\\01_edicion_automatizada\\no_tienen_carpetas"
dir_destino = "E:\\01_edicion_automatizada\\01_limpieza_de_impurezas"

def limpiar_nombre(nombre):
    caracteres_invalidos = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for c in caracteres_invalidos:
        nombre = nombre.replace(c, '-')
    return nombre

# Modificar la función procesar_archivos_audio para usar limpiar_nombre
def procesar_archivos_audio(directorio, dir_destino):
    for archivo in os.listdir(directorio):
        ruta_completa = os.path.join(directorio, archivo)
        if os.path.isfile(ruta_completa):
            try:
                if archivo.lower().endswith('.mp3'):
                    audiofile = EasyID3(ruta_completa)
                elif archivo.lower().endswith('.flac'):
                    audiofile = FLAC(ruta_completa)
                elif archivo.lower().endswith('.wav'):
                    audiofile = File(ruta_completa)  # Usar File para intentar leer metadatos de archivos .wav
                    # Nota: El manejo de metadatos en archivos .wav puede ser limitado
                else:
                    continue  # Saltar archivos que no son .mp3, .flac, o .wav

                artista = audiofile.get('artist', ['Desconocido'])[0]
                album = audiofile.get('album', ['Desconocido'])[0]
                artista = limpiar_nombre(artista)
                album = limpiar_nombre(album)
                dir_artista_album = os.path.join(dir_destino, f"{artista} - {album}")
                if not os.path.exists(dir_artista_album):
                    os.makedirs(dir_artista_album)
                shutil.move(ruta_completa, dir_artista_album)
            except Exception as e:
                print(f"Error al procesar {archivo}: {e}")

# Procesar los archivos de audio
procesar_archivos_audio(dir_origen, dir_destino)