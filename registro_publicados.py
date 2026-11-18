"""Registro de álbumes ya publicados, para no volver a renderizarlos ni descargarlos.

Un archivo de texto con una línea por álbum y cinco columnas separadas por tabuladores:
clave, nombre, origen (`canal`, `eliminar` o `subir`), video_id y fecha. Lo escriben
`pipeline_agente.py` (al eliminar, al subir y al sembrar con los títulos del canal) y lo
consulta `renderizar` antes de elegir un álbum. Los scrapers comparan con la misma clave.

La clave es "banda|álbum" sin mayúsculas, acentos, signos, año, etiqueta de tipo ni el
sufijo `__N`, así una carpeta ("Banda - Álbum (2020) [EP]") y un título del canal
("🇺🇸 BANDA - ÁLBUM (2020) • [Death Metal] ⟨FULL EP⟩") dan lo mismo. Las letras no latinas
se conservan. La censura del canal deja la primera letra y asteriscos del mismo largo
("C**"): esas claves coinciden con cualquier letra en cada asterisco.

Es una comparación de nombres, no una prueba: una reedición con el mismo nombre coincide.
Por eso quien la usa aparta el álbum en lugar de borrarlo.

La lista negra de bandas (`lista_negra_bandas.txt`) usa la misma normalización, solo con la
parte de banda: `clave_banda`.
"""

import re
import unicodedata
from collections import defaultdict
from datetime import datetime

SEPARADORES = (" - ", " – ", " — ", " ― ", " － ", " | ")
# Año o tipo entre paréntesis, corchetes o ⟨⟩: "(2020)", "[EP]", "(Full Album)", "⟨FULL EP⟩".
ETIQUETA = re.compile(
    r"[\[(⟨]\s*(?:(?:19|20)\d{2}|(?:full\s+)?(?:album|ep|e\.p\.|lp|demo|single|split|compilation"
    r"|live|boxset|mini[- ]?album))\s*[\])⟩]",
    re.IGNORECASE,
)
COLUMNAS = ("clave", "nombre", "origen", "video_id", "fecha")
ENCABEZADO = "# " + "\t".join(COLUMNAS)


def _compacto(texto):
    texto = unicodedata.normalize("NFKD", texto.casefold())
    return "".join(c for c in texto if c.isalnum() or c == "*")


def _limpiar(nombre):
    """El texto de una carpeta o de un título sin marcas invisibles, género, sufijo `__N` ni etiquetas."""
    texto = unicodedata.normalize("NFKC", str(nombre or ""))
    # Marcas invisibles como U+200E (visto en un título: "Escuela Grind ‎- ...").
    texto = "".join(c for c in texto if unicodedata.category(c) != "Cf")
    texto = texto.split(" • ")[0]  # el género del título
    texto = re.sub(r"__\d+$", "", texto.strip())  # el sufijo del render cuando el nombre se repite
    return ETIQUETA.sub(" ", texto)


def _partes(nombre):
    """(banda, álbum) compactos; sin separador, el nombre entero y None."""
    texto = _limpiar(nombre)
    for separador in SEPARADORES:
        if separador in texto:
            banda, album = (_compacto(parte) for parte in texto.split(separador, 1))
            return banda, album
    return _compacto(texto), None


def clave(nombre):
    """La clave "banda|álbum" de una carpeta o de un título del canal; None si falta una de las dos."""
    banda, album = _partes(nombre)
    return f"{banda}|{album}" if banda and album else None


def clave_banda(nombre):
    """La clave de la banda: la parte de banda de `clave`, y para un nombre sin separador, él entero.

    Así la línea "Prophecy" de la lista negra coincide con "PROPHECY - Devastated by Mankind
    (2026) [EP]". None si el nombre no tiene letras ni números.
    """
    return _partes(nombre)[0] or None


def _coincide(censurada, otra):
    return len(censurada) == len(otra) and all(a == b or a == "*" for a, b in zip(censurada, otra))


class Registro:
    def __init__(self, filas):
        self.filas = filas
        self._exactas = {}
        self._censuradas = defaultdict(list)
        for fila in filas:
            if "*" in fila["clave"]:
                self._censuradas[len(fila["clave"])].append(fila)
            else:
                self._exactas.setdefault(fila["clave"], fila)

    def buscar(self, nombre):
        """La fila con la que coincide `nombre`, o None."""
        buscada = clave(nombre)
        if buscada is None:
            return None
        if buscada in self._exactas:
            return self._exactas[buscada]
        return next((f for f in self._censuradas[len(buscada)] if _coincide(f["clave"], buscada)), None)

    def video_ids(self):
        return {f["video_id"] for f in self.filas if f["video_id"]}


def leer(ruta):
    try:
        # Una escritura cortada a mitad de una letra no puede frenar los renders: se reemplaza.
        texto = ruta.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        texto = ""
    # split("\n") y no splitlines(), que también corta en U+2028 y otros que un título puede traer.
    lineas = [linea.rstrip("\r") for linea in texto.split("\n")]
    filas = [linea.split("\t") for linea in lineas if linea and not linea.startswith("#")]
    return Registro([dict(zip(COLUMNAS, f)) for f in filas if len(f) == len(COLUMNAS)])


def leer_bandas(ruta):
    """Las claves de banda del archivo de la lista negra: una banda por línea, `#` comenta.

    Un archivo que no existe es una lista vacía. Se lee entero en cada llamada: agregar una banda
    no necesita reiniciar el servicio.
    """
    try:
        texto = ruta.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return set()
    lineas = (linea.strip() for linea in texto.split("\n"))
    claves = (clave_banda(linea) for linea in lineas if linea and not linea.startswith("#"))
    return {k for k in claves if k}


def anotar(ruta, filas):
    """Agrega al final las filas (nombre, origen, video_id); saltea las que no dan clave.

    Devuelve cuántas agregó. Quien llama serializa las escrituras.
    """
    fecha = datetime.now().astimezone().isoformat(timespec="seconds")
    lineas = []
    for nombre, origen, video_id in filas:
        nombre = re.sub(r"[\t\r\n]+", " ", nombre)
        k = clave(nombre)
        if k:
            lineas.append("\t".join((k, nombre, origen, video_id, fecha)))
    if lineas:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        nuevo = not ruta.exists()
        with open(ruta, "a", encoding="utf-8") as archivo:
            if nuevo:
                archivo.write(ENCABEZADO + "\n")
            archivo.write("\n".join(lineas) + "\n")
    return len(lineas)


def quitar(ruta, nombre, origen):
    """Saca las filas de ese nombre exacto y ese origen; las demás quedan como estaban.

    Devuelve cuántas sacó. Quien llama serializa las escrituras.
    """
    nombre = re.sub(r"[\t\r\n]+", " ", nombre)
    try:
        # newline="": las líneas quedan como están, sin cambiar un "\r\n" por "\n".
        with open(ruta, encoding="utf-8", newline="") as archivo:
            texto = archivo.read()
    except FileNotFoundError:
        return 0
    # split("\n") como `leer`: splitlines() también corta en U+2028 dentro de un título.
    lineas = texto.split("\n")
    quedan = [
        linea for linea in lineas
        if linea.startswith("#") or linea.rstrip("\r").split("\t")[1:3] != [nombre, origen]
    ]
    if len(quedan) == len(lineas):
        return 0
    temporal = ruta.with_name(f"{ruta.name}.tmp")
    with open(temporal, "w", encoding="utf-8", newline="") as archivo:
        archivo.write("\n".join(quedan))
    temporal.replace(ruta)
    return len(lineas) - len(quedan)
