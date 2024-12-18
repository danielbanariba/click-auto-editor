from limpieza.eliminar_archivos_rar import eliminar_archivos_rar
from limpieza.eliminar_archivos_txt import eliminar_archivos_txt
from limpieza.eliminar_image import delete_images
from limpieza.eliminar_subcarpetas import delete_subfolders
from limpieza.mover_archivos import mover_archivos
from limpieza.eliminar_image import delete_images
from limpieza.eliminar_url import delete_specific_files
from limpieza.normalizar_folder import normalize_folder_names
from limpieza.crear_nuevas_carpetas import procesar_archivos_audio
from limpieza.mover_carpetas import mover_carpetas
import time

# Directorio raíz desde donde comenzar la búsqueda
root_dir = "E:\\01_edicion_automatizada\\01_limpieza_de_impurezas"
root_destino = "E:\\01_edicion_automatizada\\02_juntar_audios"
no_tienen_carpetas = "E:\\01_edicion_automatizada\\no_tienen_carpetas"

# Paso 1 - Eliminar archivos RAR
print("************************************************************************************************************************************************")
print("----------------------------------------------------ELIMINAR ARCHIVOS RAR---------------------------------------------------------------------")
print("************************************************************************************************************************************************")
eliminar_archivos_rar(root_dir)

# Paso 2 - Mover archivos
time.sleep(5)
print("************************************************************************************************************************************************")
print("----------------------------------------------------MOVER ARCHIVOS---------------------------------------------------------------------")
print("************************************************************************************************************************************************")
mover_archivos(root_dir)

# Paso 3 - Crear carpetas
time.sleep(5)
print("************************************************************************************************************************************************")
print("----------------------------------------------------CREAR CARPETAS---------------------------------------------------------------------")
print("************************************************************************************************************************************************")
procesar_archivos_audio(no_tienen_carpetas, root_dir)

# Paso 4 - Eliminar archivos TXT
time.sleep(5)
print("************************************************************************************************************************************************")
print("----------------------------------------------------ELIMINAR ARCHIVOS TXT---------------------------------------------------------------------")
print("************************************************************************************************************************************************")
eliminar_archivos_txt(root_dir)

# Paso 5 - Eliminar imágenes
time.sleep(5)
print("************************************************************************************************************************************************")
print("----------------------------------------------------ELIMINAR IMAGENES---------------------------------------------------------------------")
print("************************************************************************************************************************************************")
delete_images(root_dir)

# Paso 6 - Eliminar archivos específicos
time.sleep(5)
print("************************************************************************************************************************************************")
print("----------------------------------------------------ELIMINAR ARCHIVOS ESPECIFICOS------------------------------------------------------------")
print("************************************************************************************************************************************************")
delete_specific_files(root_dir)

# Paso 7 - Eliminar subcarpetas
time.sleep(5)
print("************************************************************************************************************************************************")
print("----------------------------------------------------ELIMINAR SUBCARPETAS---------------------------------------------------------------------")
print("************************************************************************************************************************************************")
delete_subfolders(root_dir)

# Paso 8 - Normalizar nombres de carpetas
time.sleep(5)
print("************************************************************************************************************************************************")
print("----------------------------------------------------NORMALIZAR NOMBRES DE CARPETAS------------------------------------------------------------")
print("************************************************************************************************************************************************")
normalize_folder_names(root_dir)

# Paso 9 - Mover todos los archivos a la carpeta 02_juntar_audios
time.sleep(5)
print("************************************************************************************************************************************************")
print("----------------------------------------------------MOVER ARCHIVOS A 02_JUNTAR_AUDIOS-----------------------------------------------------")
print("************************************************************************************************************************************************")
mover_carpetas(root_dir, root_destino)
print("************************************************************************************************************************************************")
print("----------------------------------------------------FIN DE LA LIMPIEZA DE IMPUREZAS--------------------------------------------------------")
print("************************************************************************************************************************************************")