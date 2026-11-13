"""Censura de portadas explícitas con un modelo de visión local (Ollama).

El modelo solo dice dónde están las zonas; el relleno lo pone este módulo: negro y opaco, nunca
desenfoque ni pixelado, porque un clasificador puede reconocer el cuerpo a través de ellos.

    buscar_cajas   pregunta al modelo por la imagen entera o por pedazos, según el intento
    ampliar_cajas  agranda cada caja, la recorta al borde y une las que se tocan
    pintar_cajas   las rellena de negro

Cada intento mira con más cuidado que el anterior: el primero la imagen entera, el segundo cuatro
pedazos, del tercero en adelante nueve pedazos y la entera.
"""
import base64
import io
import json
import math
import os
import re
import shutil
import subprocess
import time

import requests
from PIL import Image, ImageDraw, ImageOps

from config import MARGEN_CENSURA, MODELO_CENSURA

LADO_AL_MODELO = 1024
LADO_DE_LA_VISTA = 1280
ESPERA_LLAMADA_S = 300
# Tope de un análisis completo: del tercer intento en adelante son diez llamadas.
PRESUPUESTO_ANALISIS_S = 25 * 60
# Entre una llamada y otra el modelo sigue cargado; si el proceso muere sin descargarlo, Ollama
# lo suelta solo a los pocos segundos.
KEEP_ALIVE_ENTRE_LLAMADAS = "60s"
ESPERA_DESCARGA_S = 30
HOST_POR_DEFECTO = "http://localhost:11434"

# Los textos van en inglés: es lo que mejor entiende el modelo.
PROMPT_IMAGEN_ENTERA = (
    "This is album cover art. Find every exposed penis, testicles, vulva, anus and female "
    "nipple in the image, painted or drawn included. Answer ONLY with JSON: "
    '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}] with coordinates normalized 0-1000. '
    "Answer [] if there is none."
)
PROMPT_PEDAZO = (
    "Content moderation for a YouTube upload. In this crop of a painted album cover, locate "
    "any visible penis (even partly hidden or stylized), testicles, vulva or anus. Answer ONLY "
    'with JSON [{"label": "...", "bbox_2d": [x1, y1, x2, y2]}], coordinates normalized 0-1000; '
    "[] if none."
)

# Lo que el modelo llama a cada zona, dicho para el usuario. La primera clave que aparezca gana.
ETIQUETAS = (
    ("testic", "testículos"),
    ("penis", "pene"),
    ("vulva", "vulva"),
    ("anus", "ano"),
    ("nipple", "pezón"),
)


class ErrorDeCensura(Exception):
    """El análisis no se pudo hacer; `motivo` es lo que el servicio le dice al usuario."""

    def __init__(self, motivo, detalle=""):
        super().__init__(detalle or motivo)
        self.motivo = motivo
        self.detalle = detalle


def etiqueta_legible(etiqueta):
    bajo = str(etiqueta).lower()
    for clave, texto in ETIQUETAS:
        if clave in bajo:
            return texto
    return str(etiqueta).strip() or "zona"


# ---------------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------------


def se_tocan(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def unir_cajas(cajas):
    """Funde en una las cajas que se tocan o se superponen, hasta que ya no quede ningún par."""
    pendientes = [tuple(caja) for caja in cajas]
    hubo_union = True
    while hubo_union:
        hubo_union = False
        unidas = []
        for caja in pendientes:
            for posicion, otra in enumerate(unidas):
                if se_tocan(otra, caja):
                    unidas[posicion] = (
                        min(otra[0], caja[0]),
                        min(otra[1], caja[1]),
                        max(otra[2], caja[2]),
                        max(otra[3], caja[3]),
                    )
                    hubo_union = True
                    break
            else:
                unidas.append(caja)
        pendientes = unidas
    return pendientes


def ampliar_cajas(cajas, ancho, alto, margen=MARGEN_CENSURA):
    """Cada caja crece `margen` de su ancho a cada lado y `margen` de su alto arriba y abajo.

    Con margen 0.5 mide el doble en cada dimensión. Después se recorta al borde de la imagen y
    las que se tocan se unen en una.
    """
    grandes = []
    for x1, y1, x2, y2 in cajas:
        crece_x, crece_y = (x2 - x1) * margen, (y2 - y1) * margen
        grandes.append((
            max(0, x1 - crece_x),
            max(0, y1 - crece_y),
            min(ancho, x2 + crece_x),
            min(alto, y2 + crece_y),
        ))
    return unir_cajas(grandes)


def pedazos(ancho, alto, intento):
    """Qué mira el modelo en cada intento: [((x1, y1, x2, y2) en píxeles, es_la_imagen_entera)].

    1: la imagen entera. 2: 2x2 pedazos del 60 % del lado. 3 en adelante: 3x3 pedazos de cerca
    del 45 % y, además, la imagen entera. Los pedazos se superponen: nada queda justo en un corte.
    """
    entera = ((0, 0, ancho, alto), True)
    if intento <= 1:
        return [entera]
    cantidad, lado = (2, 0.6) if intento == 2 else (3, 0.45)
    salidas = [i * (1 - lado) / (cantidad - 1) for i in range(cantidad)]
    trozos = [
        ((round(fx * ancho), round(fy * alto), round((fx + lado) * ancho), round((fy + lado) * alto)), False)
        for fy in salidas
        for fx in salidas
    ]
    return trozos if intento == 2 else [*trozos, entera]


def pintar_cajas(imagen, cajas):
    """Una copia de la imagen con cada caja rellena de negro opaco."""
    salida = imagen.copy()
    dibujo = ImageDraw.Draw(salida)
    for x1, y1, x2, y2 in cajas:
        dibujo.rectangle([math.floor(x1), math.floor(y1), math.ceil(x2) - 1, math.ceil(y2) - 1], fill=(0, 0, 0))
    return salida


# ---------------------------------------------------------------------------
# La imagen
# ---------------------------------------------------------------------------


def abrir_portada(ruta):
    """La portada como RGB, derecha según su EXIF: las cajas se calculan sobre lo que se ve."""
    try:
        with Image.open(ruta) as archivo:
            return ImageOps.exif_transpose(archivo).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ErrorDeCensura("imagen_ilegible", f"{ruta}: {exc}") from exc


def a_jpeg(imagen, lado=LADO_AL_MODELO):
    """La imagen achicada para caber en un cuadrado de `lado`, como JPEG."""
    chica = imagen.copy()
    chica.thumbnail((lado, lado))
    memoria = io.BytesIO()
    chica.save(memoria, "JPEG", quality=90)
    return memoria.getvalue()


# ---------------------------------------------------------------------------
# El modelo
# ---------------------------------------------------------------------------


def url_de_ollama(host=None):
    host = (host or "").strip() or HOST_POR_DEFECTO
    if "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


class ClienteOllama:
    """Habla con el Ollama local. Todo fallo de red o de HTTP sale como ErrorDeCensura."""

    def __init__(self, host=None, modelo=MODELO_CENSURA, pausa=time.sleep, espera_descarga_s=ESPERA_DESCARGA_S):
        self.url = url_de_ollama(host if host is not None else os.environ.get("OLLAMA_HOST"))
        self.modelo = modelo
        self.pausa = pausa
        self.espera_descarga_s = espera_descarga_s

    def _pedir(self, metodo, ruta, timeout_s, **datos):
        try:
            respuesta = getattr(requests, metodo)(f"{self.url}{ruta}", timeout=timeout_s, **datos)
        except requests.ConnectionError as exc:
            raise ErrorDeCensura("ollama_caido", f"Ollama no responde en {self.url}") from exc
        except requests.Timeout as exc:
            raise ErrorDeCensura("tiempo_agotado", f"Ollama no contestó en {timeout_s:.0f} s") from exc
        except requests.RequestException as exc:
            raise ErrorDeCensura("ollama_error", str(exc)) from exc
        if respuesta.status_code == 404:
            raise ErrorDeCensura("falta_el_modelo", f"Ollama no tiene {self.modelo}: ollama pull {self.modelo}")
        if respuesta.status_code >= 400:
            raise ErrorDeCensura("ollama_error", f"HTTP {respuesta.status_code}: {respuesta.text[:200]}")
        return respuesta

    def chat(self, prompt, jpeg, timeout_s=ESPERA_LLAMADA_S):
        """Lo que contesta el modelo a `prompt` mirando la imagen. Sin razonar y con temperatura 0."""
        cuerpo = {
            "model": self.modelo,
            "stream": False,
            "think": False,
            "keep_alive": KEEP_ALIVE_ENTRE_LLAMADAS,
            "options": {"temperature": 0},
            "messages": [
                {"role": "user", "content": prompt, "images": [base64.b64encode(jpeg).decode("ascii")]}
            ],
        }
        respuesta = self._pedir("post", "/api/chat", timeout_s, json=cuerpo)
        try:
            return respuesta.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ErrorDeCensura("respuesta_invalida", "Ollama contestó algo que no es un mensaje") from exc

    def esta_cargado(self):
        """True si /api/ps todavía lista el modelo. Sin Ollama no hay nada cargado."""
        try:
            modelos = self._pedir("get", "/api/ps", 10).json().get("models") or []
        except ErrorDeCensura as exc:
            if exc.motivo == "ollama_caido":
                return False
            return True
        except (ValueError, AttributeError):
            return True
        return any(self.modelo in (m.get("model"), m.get("name")) for m in modelos)

    def _esperar_descarga(self):
        limite = self.espera_descarga_s
        while True:
            if not self.esta_cargado():
                return True
            if limite <= 0:
                return False
            self.pausa(1)
            limite -= 1

    def descargar(self):
        """Saca el modelo de la GPU y confirma que /api/ps ya no lo lista.

        Pide la descarga con `keep_alive: 0`; si el modelo sigue cargado, `ollama stop`. Devuelve
        False solo si después de las dos sigue ahí: el próximo render puede quedarse sin memoria.
        """
        try:
            self._pedir("post", "/api/generate", 60, json={"model": self.modelo, "keep_alive": 0})
        except ErrorDeCensura as exc:
            if exc.motivo == "ollama_caido":
                return True
        if self._esperar_descarga():
            return True
        ollama = shutil.which("ollama")
        if ollama:
            try:
                subprocess.run([ollama, "stop", self.modelo], timeout=60, check=False, capture_output=True)
            except (OSError, subprocess.SubprocessError):
                pass
        return self._esperar_descarga()


def leer_detecciones(texto):
    """Las cajas del JSON que contestó el modelo: [(etiqueta, (x1, y1, x2, y2))] de 0 a 1000.

    None si no hay un JSON de lista (el modelo no entendió); [] si contestó que no hay nada. Una
    caja mal formada se salta sin tirar las demás.
    """
    coincidencia = re.search(r"\[.*\]", texto or "", re.S)
    if not coincidencia:
        return None
    try:
        datos = json.loads(coincidencia.group(0))
    except ValueError:
        return None
    if not isinstance(datos, list):
        return None
    detecciones = []
    for item in datos:
        caja = item.get("bbox_2d") if isinstance(item, dict) else None
        numeros = (
            isinstance(caja, (list, tuple))
            and len(caja) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in caja)
        )
        if not numeros:
            continue
        a, b, c, d = (min(max(float(v), 0.0), 1000.0) for v in caja)
        x1, x2, y1, y2 = min(a, c), max(a, c), min(b, d), max(b, d)
        if x2 <= x1 or y2 <= y1:
            continue
        detecciones.append((str(item.get("label") or "zona"), (x1, y1, x2, y2)))
    return detecciones


def buscar_cajas(imagen, intento, cliente, reloj=time.monotonic, presupuesto_s=PRESUPUESTO_ANALISIS_S):
    """Dónde hay zonas que tapar: [(etiqueta, x1, y1, x2, y2)] en píxeles de `imagen`.

    Un pedazo cuya respuesta no es un JSON válido no tira las cajas de los demás; si fallan todos,
    el intento falla. Entre pedazo y pedazo mira el reloj: pasado el presupuesto, corta.
    """
    ancho, alto = imagen.size
    limite = reloj() + presupuesto_s
    encontradas, entendidas = [], 0
    for (x1, y1, x2, y2), entera in pedazos(ancho, alto, intento):
        restante = limite - reloj()
        if restante <= 0:
            raise ErrorDeCensura("tiempo_agotado", f"el análisis pasó los {presupuesto_s / 60:.0f} min")
        texto = cliente.chat(
            PROMPT_IMAGEN_ENTERA if entera else PROMPT_PEDAZO,
            a_jpeg(imagen.crop((x1, y1, x2, y2))),
            min(ESPERA_LLAMADA_S, restante),
        )
        detecciones = leer_detecciones(texto)
        if detecciones is None:
            continue
        entendidas += 1
        for etiqueta, (a, b, c, d) in detecciones:
            ancho_p, alto_p = x2 - x1, y2 - y1
            encontradas.append((
                etiqueta,
                x1 + a * ancho_p / 1000,
                y1 + b * alto_p / 1000,
                x1 + c * ancho_p / 1000,
                y1 + d * alto_p / 1000,
            ))
    if not entendidas:
        raise ErrorDeCensura("respuesta_invalida", "el modelo no contestó un JSON que pueda leer")
    return encontradas
