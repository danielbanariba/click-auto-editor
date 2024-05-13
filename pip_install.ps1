python -m venv /path/to/new/virtual/environment

# Crear un entorno virtual
python -m venv env

# Activar el entorno virtual
Set-Location env/Scripts
.\activate.ps1

Set-Location ..
Set-Location ..

# Ahora instalar los paquetes
pip install image
pip install pyautogui
pip install numpy
pip install unidecode
pip install eyed3
pip install pydub