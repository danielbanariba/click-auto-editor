<div align="center">
  <h1 align="center">Algoritmos de edición automática</a></h1>
</div>

### :gear: Instalacion

```sh
.\pip_install.ps1
```

## Explicacion de que hace cada script

#### 0. limipieza_impuerzas.py

Va a recorrer todas las carpetas que contenga el directorio en que esta y va a empezar a eliminar todo lo que no sea una imagen y un sonido, si existen mas de 2 imagenes le va a dar mas prioridad el que contenga la palabra reservada 'cover.'[formato de la imagen].

#### 1. juntar_audios

Agarra todos los audios que contiene la carpeta y los va a unir en un solo audio, pero antes de hacer eso va hacer dos verificiaciones previas, verifica que existe una imagen, y si no existe va a extraer la imagen que contengan en los metadados del audio, va a generar un archivo **.txt** (que es la informacion de la descripcion del video) y la duracion de cada cancion los va a sumar para obtener la duracion de las canciones para de esa manera tener una predicion de la proxima cancion que sigue, y por ultimo va a extraer el anio de la cancion y el genero y si no tiene va a poner por defecto **"Unknown"**.

#### 2. cambiar_nombre_imagen

Cambia el nombre de la imagen para ponerlo en la ultima posicion del resultado de archivos, tambien va a rescalar la imagen, y la portada principal osea la de **2500x2500** va a llamar una funcion le aplica el efecto sombra y lo va a guardar en formato **.png**.

#### 3. cantidad_de_archivos.py

Va a recorre todas las carpetas y me a imprimir por pantalla la cantidad que contiene cada carpeta, eso se hace para que siempre tenga 3 archivos (1 audio y 2 imagenes)

#### 4. verificacion_humana.py

Comprobamos que los albumes que tengamos no esten subidos ya a Youtube, de forma manual, por si el primer algoritmo del otro proyecto no fue tan preciso.

#### 5. auto_effects.py

Abre Adobe After Effects y de manera automatica va a empezar a editar, va a ejecutar varios scripts en formato .jsx que son:

 1) **imagen_movimiento.jsx** = le da el movimiento a la imagen de fondo, que tenga rotacion de manera aleatoria y la imagen va a reaccionar al ritmo de la musica,
 2) **espectro_de_audio.jsx** = Agarra la imagen, va a examinar la imagen y obtendra el promedio en hexadecimal de los colores para aplicar dicho valor en el espectrum audio.
>
> [!CAUTION]
> Si algo se mueve ya minusculo pixel que sea, hay que generar ese cambio porque sino todo se va al carajo
>

#### 6. auto_premier.py

El programa va abrir premier pro, arrastra el trabajo que ya se hizo con el algoritmo anterior **(4. auto_effects.py)** lo junta con el intro que tengo guardado, le aplica una transicion sencilla y lo manda a renderizar al programa Adobe Media Encoder y el ciclo se repite.
>
> [!IMPORTANT]
> Se tiene que abrir Adobe Media Encoder antes de ejecutar el script.
>

#### 7. mover_videos_terminados

Una vez terminado de renderizas los videos, este algoritmo va a mover los archivos con extension .prproj y .aep a la carpeta donde contiene el video, esto se hace con el unico fin de poder liberar almacenamiento y tener un orden.

#### 8. subir_video_coordenadas

Todos los videos que se renderizaron y se movieron correctamente se va a subir al canal de youtube, va a poner una fecha y una hora random (24 horas con intervalo de 15 minutos).
