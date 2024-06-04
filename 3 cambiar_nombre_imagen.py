# Recorrer todos los directorios en la ruta principal
# for folder_name in os.listdir(main_dir_path):
#     folder_path = os.path.join(main_dir_path, folder_name)

#     if os.path.isdir(folder_path):
#         new_folder_path = folder_path.replace('–', '-')

#         # Mover el directorio si el nombre ha cambiado
#         if folder_path != new_folder_path:
#             shutil.move(folder_path, new_folder_path)
#             folder_path = new_folder_path

#         # Buscar todos los archivos de imagen en el directorio
#         archivos_imagen = glob.glob(os.path.join(folder_path, '*.[jpg][png][jpeg][tiff][raw]*'))

#         # Para cada archivo de imagen, cambiar su nombre al nombre del directorio
#         for archivo in archivos_imagen:
#             extension = archivo.split('.')[-1]
#             nuevo_nombre = os.path.join(folder_path, "Z"+f"{folder_name}.{extension}")
#             os.rename(archivo, nuevo_nombre)

#             # Open the image file
#             with Image.open(nuevo_nombre) as img:
#                 # Resize the image and save it
#                 img_resized = img.resize((2050, 2050))
#                 img_resized.save(f"{nuevo_nombre}_2050x2050.{extension}")
            
#                 # Add shadow to the 2050x2050 image
#                 add_shadow(f"{nuevo_nombre}_2050x2050.{extension}", f"{nuevo_nombre}_2050x2050_shadow.png", 255, 135, 53, 37)
            
#                 img_resized = img.resize((4050, 4050))
#                 img_resized.save(f"{nuevo_nombre}_4050x4050.{extension}")
            
#             # Remove the original resized file
#             os.remove(nuevo_nombre)
            
#             # Remove the first created image (2050x2050)
#             os.remove(f"{nuevo_nombre}_2050x2050.{extension}")

#! IMPORTANTE!         
# #Esto es bien raro, a mi siempre me ha funcionado el de arriba pero ahora ya no quiero funcionar la maldita esta, asi que si falla, descomentar el de abaajo
import os
import shutil
from PIL import Image
from effects.sombra import add_shadow
import stat

# Definir la ruta de la carpeta principal
main_dir_path = "E:\\01_edicion_automatizada\\YA"
# Definir la ruta del directorio destino
destination_dir_path = "E:\\01_edicion_automatizada\\verificacion"

# Recorrer todos los directorios en la ruta principal
for folder_name in os.listdir(main_dir_path):
    folder_path = os.path.join(main_dir_path, folder_name)

    if os.path.isdir(folder_path):
        new_folder_path = folder_path.replace('–', '-')

        # Mover el directorio si el nombre ha cambiado
        if folder_path != new_folder_path:
            shutil.move(folder_path, new_folder_path)
            folder_path = new_folder_path

        # Buscar todos los archivos de imagen en el directorio
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.raw')):
                archivo = os.path.join(folder_path, filename)
                
                # Verificar si el archivo existe antes de intentar renombrarlo
                if os.path.exists(archivo):
                    # Renombrar el archivo de imagen
                    nuevo_nombre = os.path.join(folder_path, "Z" + f"{folder_name}.png")
                    os.rename(archivo, nuevo_nombre)
                    
                    # Abrir el archivo de imagen
                    with Image.open(nuevo_nombre) as img:
                        # Cambiar el tamaño de la imagen a 2050x2050 y guardar
                        img_resized = img.resize((2050, 2050))
                        if img_resized.mode in ['RGBA', 'CMYK']:
                            img_resized = img_resized.convert('RGB')
                        img_resized.save(f"{nuevo_nombre}_2050x2050.png")
                        
                        # Intentar añadir una sombra a la imagen
                        try:
                            add_shadow(f"{nuevo_nombre}_2050x2050.png", f"{nuevo_nombre}_2050x2050_shadow.png", 255, 135, 53, 37)
                        except Exception as e:
                            print(f"Error adding shadow to {nuevo_nombre}_2050x2050.png: {e}")
                        
                        # Cambiar el tamaño de la imagen a 4050x4050 y guardar
                        img_resized = img.resize((4050, 4050))
                        if img_resized.mode in ['RGBA', 'CMYK']:
                            img_resized = img_resized.convert('RGB')
                        img_resized.save(f"{nuevo_nombre}_4050x4050.png")

                    # Obtener los permisos actuales del archivo
                    file_stat = os.stat(nuevo_nombre)

                    # Agregar permiso de escritura
                    os.chmod(nuevo_nombre, file_stat.st_mode | stat.S_IWRITE)

                    # Ahora se puede eliminar el archivo original
                    os.remove(nuevo_nombre)
                    
                    # Eliminar la imagen redimensionada creada primero (2050x2050)
                    os.remove(f"{nuevo_nombre}_2050x2050.png")
                else:
                    print(f"File not found: {archivo}")

        # Mover la carpeta procesada al directorio destino
        shutil.move(folder_path, os.path.join(destination_dir_path, folder_name))