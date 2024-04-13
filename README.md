<div align="center">
  <h1 align="center">Algoritmos de edicion automatica</a></h1>
</div>

<!-- Installation -->
### :gear: Instalacion

En un solo comando: 
```sh
pip install image pyautogui numpy unidecode eyed3 pydub
``` 
  
Oh uno por uno (Si es que tuvo problemas en instalarlo de un solo):
```sh
pip install image
```
```sh
pip install pyautogui
```
```sh
pip install numpy
```
```sh
pip install unidecode
```
```sh
pip install eyed3
```
```sh
pip install pydub
```

<!-- TENGO QUE DEJAR PASO A PASO QUE TIENE QUE HACER! -->
## Explicacion de que hace cada script: 

#### 0. limipieza_impuerzas.py:
Va a recorrer todas las carpetas que contenga el directorio en que esta y va a eliminar todo lo que no sea o una imagen y un sonidom, si existen mas de 2 imagenes le va a dar mas prioridad el que contega la palabra reservada 'cover.'[formato de la imagen].

#### 1. juntar-audios.py:
Agarra todos los audios que contiene la carpeta y los va aunir en un solo audio, pero antes de hacer eso va hacer dos verificiaciones previas, verifi/ca que existe una imagen, y si no existe va a extraer la imagen que contengan en los metadados del audio, va a generar un archivo .txt que lo que va hacer es sumar la duracion de las canciones para de esa manera tener una predicion de la proxima cancion que sigue, y por ultimo va a extraer el anio de la cancion y el genero y si no tiene va a poner por defecto "Unknown".

#### 2. cambiar_nombre_imagen:
Cambia el nombre de la imagen para que esa manera ponerlo en la ultima posicion del resultado, tambien va aumentar el tamanio de la imagen o mejor dicho va a rescalar la imagen, y la portada principal osea la de 2500x2500 va a llamar una funcion que lo que va hacer es aplicar un efecto sombra y lo va a guardar en formato png.

#### 03. auto-effects:
Abre Adobe After Effects y de manera automatica va a empezar a editar, puntos importantes es que va alicar dos scripts en formato .jsx que son el movimiento de la imagen, que tenga rotacion de manera aleatoria y la imagen va a reaccionar al ritmo de la musica, y va a ejecutar otro script que le va a permitir generar un efecto llamado 'espectrum audio', tambien va agarrar la imagen, va a examinar la imagen y obtendra el promedio en hexadecimal del los colores para aplicar dicho valor en el espectrum audio.
>
> [!WARNING] 
> Si algo se mueve ya minusculo pixel que sea, hay que generar ese cambio porque sino todo se va al carajo
>

#### 04. render:
Abra premier pro, arrastra el trabajo que ya hizo previamente el algoritmo anterior y lo junta con el intro que ya tengo previamente guardado, le aplica una transicion sencilla y lo manda a al programa Media Conder para su proxima renderizacion.