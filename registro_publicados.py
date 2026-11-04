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


def clave(nombre):
    """La clave "banda|álbum" de una carpeta o de un título del canal; None si falta una de las dos."""
    texto = unicodedata.normalize("NFKC", str(nombre or ""))
    # Marcas invisibles como U+200E (visto en un título: "Escuela Grind ‎- ...").
    texto = "".join(c for c in texto if unicodedata.category(c) != "Cf")
    texto = texto.split(" • ")[0]  # el género del título
    texto = re.sub(r"__\d+$", "", texto.strip())  # el sufijo del render cuando el nombre se repite
    texto = ETIQUETA.sub(" ", texto)
    for separador in SEPARADORES:
        if separador in texto:
            banda, album = (_compacto(parte) for parte in texto.split(separador, 1))
            return f"{banda}|{album}" if banda and album else None
    return None


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
        lineas = ruta.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lineas = []
    filas = [linea.split("\t") for linea in lineas if linea and not linea.startswith("#")]
    return Registro([dict(zip(COLUMNAS, f)) for f in filas if len(f) == len(COLUMNAS)])


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
