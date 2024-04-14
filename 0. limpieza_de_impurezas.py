from limpieza.eliminar_archivos_rar import eliminar_archivos_rar
from limpieza.eliminar_archivos_txt import eliminar_archivos_txt
from limpieza.eliminar_image import delete_images
from limpieza.eliminar_subcarpetas import delete_subfolders
from limpieza.mover_archivos import mover_archivos
from limpieza.eliminar_image import delete_images
from limpieza.eliminar_url import delete_specific_files
from limpieza.normalizar_folder import normalize_folder_names
import time

# Directorio raíz desde donde comenzar la búsqueda
root_dir = "D:\\01_edicion_automatizada\\01_limpieza_de_impurezas"

print("************************************************************************************************************************************************")
print("----------------------------------------------------ELIMINAR ARCHIVOS RAR---------------------------------------------------------------------")
#eliminar_archivos_rar(root_dir)
# time.sleep(5)
print("************************************************************************************************************************************************")
print("----------------------------------------------------MOVER ARCHIVOS---------------------------------------------------------------------")
# #mover_archivos(root_dir)
# time.sleep(5)
print("************************************************************************************************************************************************")
print("----------------------------------------------------ELIMINAR ARCHIVOS TXT---------------------------------------------------------------------")
#eliminar_archivos_txt(root_dir)
# time.sleep(5)
print("************************************************************************************************************************************************")
print("----------------------------------------------------ELIMINAR IMAGENES---------------------------------------------------------------------")
# delete_images(root_dir)
# time.sleep(5)
print("************************************************************************************************************************************************")
print("----------------------------------------------------ELIMINAR ARCHIVOS ESPECIFICOS------------------------------------------------------------")
# delete_specific_files(root_dir)
# time.sleep(5)
print("************************************************************************************************************************************************")
print("----------------------------------------------------ELIMINAR SUBCARPETAS---------------------------------------------------------------------")
#delete_subfolders(root_dir)
# time.sleep(5)
print("************************************************************************************************************************************************")
print("----------------------------------------------------NORMALIZAR NOMBRES DE CARPETAS------------------------------------------------------------")
#normalize_folder_names(root_dir)
print("************************************************************************************************************************************************")
