import os
import shutil

regresar_save_after_effects = "C:\\Users\\banar\\Desktop\\regresar_save_after_effects"
audio_scripts = "D:\\01_edicion_automatizada\\audio_scripts"
save_after_effects = "C:\\Users\\banar\\Desktop\\save_after_effects"

# Obtener los nombres de las carpetas en el directorio audio_scripts
audio_scripts_folders = [name for name in os.listdir(audio_scripts) if os.path.isdir(os.path.join(audio_scripts, name))]
print(f'Carpetas en audio_scripts: {audio_scripts_folders}')

# Obtener los nombres de los archivos en el directorio regresar_save_after_effects
regresar_files = os.listdir(regresar_save_after_effects)
print(f'Archivos en regresar_save_after_effects: {regresar_files}')

for file in regresar_files:
    # Eliminar los dos últimos caracteres del nombre del archivo
    file_name = file[:-6]
    print(f'Nombre de archivo después de eliminar los dos últimos caracteres: {file_name}')
    
    # Si el nombre del archivo está en el directorio audio_scripts
    if file_name in audio_scripts_folders:
        # Mover el archivo al directorio save_after_effects
        print(f'Moviendo el archivo {file} a {save_after_effects}')
        shutil.move(os.path.join(regresar_save_after_effects, file), save_after_effects)