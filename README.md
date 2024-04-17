<div align="center">
  <h1 align="center">Algoritmos de edición automática</a></h1>
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
Va a recorrer todas las carpetas que contenga el directorio en que esta y va a eliminar todo lo que no sea o una imagen y un sonido, si existen mas de 2 imagenes le va a dar mas prioridad el que contenga la palabra reservada 'cover.'[formato de la imagen].

#### 1. juntar-audios:
Agarra todos los audios que contiene la carpeta y los va a unir en un solo audio, pero antes de hacer eso va hacer dos verificiaciones previas, verifica que existe una imagen, y si no existe va a extraer la imagen que contengan en los metadados del audio, va a generar un archivo .txt que lo que va hacer la descripcion del video y es la suma de la duracion de las canciones para de esa manera tener una predicion de la proxima cancion que sigue, y por ultimo va a extraer el anio de la cancion y el genero y si no tiene va a poner por defecto "Unknown".

#### 2. cambiar_nombre_imagen:
Cambia el nombre de la imagen para que esa manera ponerlo en la ultima posicion del resultado de archivos, tambien va a rescalar la imagen, y la portada principal osea la de 2500x2500 va a llamar una funcion le aplica el efecto sombra y lo va a guardar en formato .png.

#### 3. cantidad_de_archivos.py:
Va a recorre todas las carpetas y me a imprimir por pantalla la cantidad que contiene cada carpeta, eso se hace con el fin de poder siempre tener solo 3 archivos, osea 1 audio y 2 imagenes.

#### 4. auto-effects:
Abre Adobe After Effects y de manera automatica va a empezar a editar, puntos importantes es que va aplicar dos scripts en formato .jsx que son el movimiento de la imagen, que tenga rotacion de manera aleatoria y la imagen va a reaccionar al ritmo de la musica, y va a ejecutar otro script que le va a permitir generar un efecto llamado 'espectrum audio', el agoritmo va agarrar la imagen, va a examinar la imagen y obtendra el promedio en hexadecimal de los colores para aplicar dicho valor en el espectrum audio.
>
> [!CAUTION] 
> Si algo se mueve ya minusculo pixel que sea, hay que generar ese cambio porque sino todo se va al carajo
>

#### 5. verificacion_humana:
Esto es experimental, pero antes de renderizar y perder tiempo en el procesamiento que este conlleva, vamos a primero verificar en youtube solo a ver si lo que tenemos no esta ya publicado en Youtube, y le vamos a poner un "YES" y al momento de poner este comando se eliminara todo los archivos y va abrir otra pestaña y asi hasta terminar con todas las carpetas.

#### 6. auto-premier:
Abra premier pro, arrastra el trabajo que ya hizo previamente el algoritmo anterior (4. auto-effects.py) y lo junta con el intro que tengo guardado, le aplica una transicion sencilla y lo manda a exportar a Adobe Media Encoder y el ciclo se repite.
>
> [!IMPORTANT]
> Se tiene que abrir Adobe Media Encoder antes de ejecutar el script.
>

#### 7. mover_videos_terminados:
Una vez terminado de renderizas los videos, tenemos que mover los archivos con extension .prproj y .aep a la carpeta donde contiene el video, esto se hace con el unico fin de poder liberar almacenamiento

#### 8. subir_video_coordenadas:
Sube los videos que se tiene almacenados al canal de youtube, va a poner una fecha y una hora random siempre respectando 30 dias desde hoy y la hora de 24 horas con intervalo de 15 minutos.