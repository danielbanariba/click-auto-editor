"""La portada de un álbum: la que el servicio muestra en Telegram y la que usa el render."""

import re
from pathlib import Path

EXTENSIONES_IMAGEN = (".png", ".jpg", ".jpeg")

# Palabras del nombre que dicen que la imagen es la portada: "Front.jpg", "Album - Cover.jpg".
PALABRAS_DE_PORTADA = {"front", "cover", "portada", "caratula", "carátula"}

# Palabras del nombre de las otras imágenes que vienen con el disco: la contraportada, el
# librillo, el logo, las miniaturas de Windows Media (AlbumArtSmall.jpg), los
# espectrogramas... Ganan sobre las de portada: "Back Cover.jpg" es la contraportada.
PALABRAS_DE_OTRA_IMAGEN = {
    "back", "contraportada", "albumartsmall", "small", "thumb", "thumbnail", "inlay", "inside",
    "interior", "booklet", "tray", "logo", "band", "scan", "scans", "spec", "spectral",
    "spectrogram", "flyer", "poster",
}

# El escaneo del disco ("cd.jpg"). Solo cuenta sin palabra de portada: los nombres de las
# ediciones en FLAC llevan "cd" ("...-cd-flac-2016-front.jpg").
PALABRAS_DEL_DISCO = {"cd", "disc"}


def es_propaganda(imagen):
    """Las imágenes de DeathGrind Club que vienen dentro de los archivos: DGC.jpg, DGC - PURE HATRED.jpg..."""
    return Path(imagen).stem.upper().startswith("DGC")


def _prioridad(imagen):
    if imagen.name.lower().startswith("cover."):
        return 0
    palabras = set(re.findall(r"[^\W\d_]+", imagen.stem.lower()))
    if palabras & PALABRAS_DE_OTRA_IMAGEN:
        return 3
    if palabras & PALABRAS_DE_PORTADA:
        return 1
    if palabras & PALABRAS_DEL_DISCO:
        return 3
    return 2


def elegir_portada(carpeta):
    """cover.* primero; si no, la que se llama front o cover; si no, la primera por nombre.

    Las imágenes que por su nombre son otra cosa (contraportada, librillo, miniatura...) van
    al final, y las propagandas de DeathGrind Club no cuentan. Con solo sombras, la primera
    sombra. Sin ninguna imagen devuelve None: el render saca entonces la portada del audio.
    """
    imagenes = sorted(
        (
            p
            for p in Path(carpeta).iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONES_IMAGEN and not es_propaganda(p)
        ),
        key=lambda p: (_prioridad(p), p.stem.lower(), p.name),
    )
    sin_sombra = [i for i in imagenes if not i.name.endswith("_shadow.png")]
    return next(iter(sin_sombra or imagenes), None)
