#!/usr/bin/env python3
"""Pasos del pipeline que ejecuta servicio_publicar.py, de a un álbum.

Cada subcomando termina imprimiendo UNA línea JSON para que el servicio
decida el paso siguiente; el resto de la salida va a data/agente/logs/.

    renderizar   renderiza el próximo álbum y devuelve su carpeta
    verificar    corre los tests del video y genera la captura
    subir        sube el video como privado y borra la carpeta del álbum
    eliminar     borra la carpeta del álbum sin subirlo (ya estaba publicado)
    a_censura    aparta el álbum en Censura/ para censurar su portada (botón «🔞 Censurar»)
    censurar     busca con un modelo local qué tapar de la portada y deja una candidata censurada
    aceptar_censura  pone la portada censurada en el álbum y lo devuelve a su carpeta de origen
    programar    espera el procesado, revisa reclamos, activa la monetización y programa
    eliminar_video  borra un video de YouTube (no se puede deshacer; lo pide un botón)
    impugnar        impugna en Studio los reclamos de un video (declaración legal; lo pide un botón)
    estado       videos sin programar, álbumes sin aprobar y render en curso
"""
import argparse
import contextlib
import errno
import fcntl
import functools
import importlib.util
import json
import os
import random
import re
import shutil
import ssl
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import httplib2
from google.auth.exceptions import RefreshError, TransportError
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from config import (
    AUDIO_FORMATS,
    CARPETA_RESCATE_RENDER,
    DIR_CENSURA,
    DIR_ERROR404,
    DIR_RENDERIZANDO,
    DIR_UPLOAD,
    DIR_YA_PUBLICADOS,
    DIRS_ORIGEN_RENDER,
    INTRO_DURATION,
    MIN_FREE_SPACE_GB,
    PROJECT_ROOT,
    RENDER_SSD_DIR,
    SSD_TEMP_DIR,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
import censura_portadas
import registro_publicados
from portadas import elegir_portada, reemplazar_portada

DIR_AGENTE = PROJECT_ROOT / "data" / "agente"
PENDIENTES_JSON = DIR_AGENTE / "pendientes.json"
# Los álbumes ya publicados (eliminados, subidos o en el canal): no se vuelven a renderizar.
REGISTRO_PUBLICADOS = DIR_AGENTE / "ya_publicados.tsv"
PROGRAMADOS_JSONL = DIR_AGENTE / "programados.jsonl"
DIR_CAPTURAS = DIR_AGENTE / "capturas"
LISTA_NEGRA_RECLAMANTES = PROJECT_ROOT / "lista_negra_reclamantes.txt"
# Los videos que ya se apelaron (o que no se pueden apelar): a ninguno se le manda nada dos veces.
APELACIONES_JSON = DIR_AGENTE / "apelaciones.json"
DIR_LOGS = DIR_AGENTE / "logs"
RAICES_BORRABLES = (DIR_RENDERIZANDO, DIR_UPLOAD)
# Lo que `a_censura` y `censurar` anotan de cada álbum en Censura/ (de dónde vino, las cajas de cada
# intento, la candidata y sus vistas) y, en `originales/`, las portadas que Aceptar reemplazó.
# Va fuera de la carpeta del álbum: hasta Aceptar, nada cambia dentro de ella.
DIR_CENSURA_DATOS = DIR_AGENTE / "censura"
CARPETA_ORIGINALES = "originales"
# Lo que aparta `eliminar` se puede devolver con `restaurar` hasta que `vaciar_papelera` lo borra.
PLAZO_PAPELERA_S = 24 * 3600

TOLERANCIA_DURACION_S = 3.0
VOLUMEN_MEDIO_MINIMO_DB = -45.0
SILENCIO_LARGO_S = 20
# Mínimo de anticipación al programar: YouTube rechaza un publishAt que ya pasó.
ANTICIPACION_MIN = 2
# Entre un video publicado o programado y otro tiene que haber al menos esto.
SEPARACION_ENTRE_VIDEOS = timedelta(hours=1)
MARGEN_RECLAMO_MIN = 60
ESPERA_ENTRE_CONSULTAS_S = 300
REINTENTOS_CONEXION = 3
ESPERA_CONEXION_S = 2
# Los mismos fallos que googleapiclient considera transitorios, más el envoltorio
# de google-auth para cuando falla el refresco del token.
ERRORES_DE_CONEXION = (
    ConnectionError,
    TimeoutError,
    ssl.SSLError,
    httplib2.ServerNotFoundError,
    TransportError,
)
# «Red inalcanzable» llega de httplib2 como un OSError simple, no como un ConnectionError, y
# solo se distingue de un archivo ausente o un permiso denegado por su errno.
ERRNOS_DE_RED = frozenset({errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ENETDOWN})
# Lo que hay que atrapar para después decidir con `es_error_de_conexion`: un `except` solo
# entiende de clases, y el errno no es una.
POSIBLES_ERRORES_DE_CONEXION = (*ERRORES_DE_CONEXION, OSError)
PROCESO_RENDER = "ffmpeg_render.py"
# Un análisis de portada es este script con el subcomando `censurar`: ocupa ~17 GB de VRAM, y el
# render y el análisis no pueden correr a la vez.
SCRIPT_PIPELINE = "pipeline_agente.py"
SUBCOMANDO_CENSURA = "censurar"
ESPERA_RENDER_MAX_S = 6 * 3600
ESCANEO_PROGRAMADOS = int(os.environ.get("AGENTE_ESCANEO_PROGRAMADOS", "500"))
CAMPOS_STATUS_CONSERVADOS = ("embeddable", "license", "publicStatsViewable")
# Playwright agrega estos dos flags a Chromium en Linux: con ellos Brave no usa el
# llavero del sistema y no puede descifrar las cookies del perfil copiado, así que
# Studio pide iniciar sesión.
ARGS_QUE_ROMPEN_EL_LLAVERO = ("--password-store=basic", "--use-mock-keychain")
VENTANA_STUDIO = {"width": 1400, "height": 1000}
# En headless el navegador se anuncia como HeadlessChrome y Studio responde que no es
# compatible: se anuncia como el Chrome normal de la misma versión.
USER_AGENT_STUDIO = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/{version}.0.0.0 Safari/537.36"
)
VERSION_STUDIO_POR_DEFECTO = 141
# Con ellas Brave elige el llavero del escritorio (KWallet) para descifrar las cookies.
VARIABLES_DE_ESCRITORIO = (
    "XDG_CURRENT_DESKTOP",
    "DESKTOP_SESSION",
    "KDE_FULL_SESSION",
    "KDE_SESSION_VERSION",
    "DBUS_SESSION_BUS_ADDRESS",
)

P_SIN_PROBLEMAS = re.compile(
    r"(no se encontraron problemas|no se encontr[oó] ning[uú]n problema|"
    r"no se han encontrado reclamaciones|"
    r"no hay problemas de derechos de autor|no copyright issues|no issues found|"
    r"no claims found)",
    re.IGNORECASE,
)
# Texto de un reclamo encontrado. La página sin reclamos dice "No se ha encontrado contenido
# protegido por derechos de autor en tu vídeo": por eso el "no" delante lo descarta.
P_CON_RECLAMO = re.compile(
    r"(?<!no )se ha encontrado contenido protegido por derechos de autor", re.IGNORECASE
)
ESPERA_RESULTADO_RECLAMOS_S = 20


class SinCredenciales(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def cargar_script(nombre_archivo, alias):
    spec = importlib.util.spec_from_file_location(alias, PROJECT_ROOT / nombre_archivo)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


@functools.lru_cache(maxsize=None)
def uploader_real():
    return cargar_script("3. subir_video_API.py", "subir_video_api")


@contextlib.contextmanager
def bloqueo(nombre):
    """Serializa entre procesos: varios `programar` pueden correr a la vez."""
    PENDIENTES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PENDIENTES_JSON.parent / f"{nombre}.lock", "w", encoding="utf-8") as archivo:
        fcntl.flock(archivo, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(archivo, fcntl.LOCK_UN)


def responder(datos):
    print(json.dumps(datos, ensure_ascii=False))
    return 0 if datos.get("ok") else 1


def ruta_log(prefijo):
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    return DIR_LOGS / f"{prefijo}_{datetime.now():%Y%m%d-%H%M%S}.log"


def listar_carpetas(directorio):
    if not directorio.is_dir():
        return []
    # os.scandir no hace un stat por carpeta: en el disco NTFS, iterdir() con
    # is_dir() tardaba minutos con los miles de álbumes de Album/ y EP/.
    with os.scandir(directorio) as entradas:
        return sorted(entrada.name for entrada in entradas if entrada.is_dir())


def ordenar_por_anio(nombres):
    """Los más nuevos primero, según el año entre paréntesis; sin año, al final."""

    def clave(nombre):
        anio = re.search(r"\(((?:19|20)\d{2})\)", nombre)
        return (0, -int(anio.group(1)), nombre.lower()) if anio else (1, 0, nombre.lower())

    return sorted(nombres, key=clave)


def siguiente_album():
    """El próximo álbum para renderizar, recorriendo DIRS_ORIGEN_RENDER en orden."""
    for origen in DIRS_ORIGEN_RENDER:
        nombres = listar_carpetas(origen)
        if nombres:
            return origen / ordenar_por_anio(nombres)[0]
    return None


def anotar_publicados(filas):
    """Agrega al registro las filas (nombre, origen, video_id)."""
    with bloqueo("publicados"):
        return registro_publicados.anotar(REGISTRO_PUBLICADOS, filas)


def destino_libre(directorio, nombre):
    """`directorio/nombre`, o con sufijo `__N` si ya existe: rename pisaría una carpeta vacía."""
    destino, n = directorio / nombre, 2
    while destino.exists():
        destino, n = directorio / f"{nombre}__{n}", n + 1
    return destino


def siguiente_sin_publicar(aleatorio=False, excluir=()):
    """Como siguiente_album, pero aparta en DIR_YA_PUBLICADOS los que están en el registro.

    Devuelve el álbum (o None) y los apartados; uno que no se pudo mover lleva `error` y
    queda en su carpeta sin renderizar. Con `aleatorio`, el orden es al azar entre todas las
    carpetas de origen. Los nombres de `excluir` (la lista del /crear_lista) se saltan.
    """
    # `restaurar` devuelve álbumes a los orígenes con este candado: sin él, un álbum devuelto
    # mientras se recorre podría verse con el registro leído antes y apartarse.
    with bloqueo("origenes"):
        return _siguiente_sin_publicar(aleatorio, excluir)


def _siguiente_sin_publicar(aleatorio, excluir):
    registro = registro_publicados.leer(REGISTRO_PUBLICADOS)
    apartados = []
    # En orden se listan de a una: un origen se recorre solo si los anteriores no tienen nada.
    candidatos = (
        (origen, nombre)
        for origen in DIRS_ORIGEN_RENDER
        for nombre in ordenar_por_anio(listar_carpetas(origen))
    )
    if aleatorio:
        candidatos = list(candidatos)
        random.shuffle(candidatos)
    for origen, nombre in candidatos:
        if nombre in excluir:
            continue
        fila = registro.buscar(nombre)
        if fila is None:
            return origen / nombre, apartados
        apartado = {"album": nombre, "origen": fila["origen"], "como": fila["nombre"]}
        try:
            DIR_YA_PUBLICADOS.mkdir(parents=True, exist_ok=True)
            destino = destino_libre(DIR_YA_PUBLICADOS, nombre)
            # rename y no shutil.move: los orígenes y Ya publicados/ están en BASE_DIR, y si
            # rename falla, shutil.move copia el álbum y, sin poder borrar el original, deja
            # una copia entera en cada vuelta.
            (origen / nombre).rename(destino)
            apartado["destino"] = str(destino)
        except OSError as exc:
            apartado["error"] = str(exc)
        apartados.append(apartado)
    return None, apartados


def slug(texto):
    return re.sub(r"[^\w.-]+", "_", texto).strip("_") or "video"


def formato_duracion(segundos):
    total = int(round(segundos))
    horas, resto = divmod(total, 3600)
    minutos, segs = divmod(resto, 60)
    if horas:
        return f"{horas}:{minutos:02d}:{segs:02d}"
    return f"{minutos}:{segs:02d}"


def raiz_borrable(carpeta, raices=RAICES_BORRABLES):
    carpeta = carpeta.resolve()
    for raiz in raices:
        raiz = Path(raiz).resolve()
        if raiz in carpeta.parents:
            return raiz
    return None


# ---------------------------------------------------------------------------
# Estado local (videos subidos que faltan programar)
# ---------------------------------------------------------------------------


def leer_pendientes():
    if not PENDIENTES_JSON.exists():
        return []
    with open(PENDIENTES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def albumes_por_aprobar():
    """Álbumes ya renderizados que todavía no se subieron (esperan el OK del usuario)."""
    return [
        str(DIR_RENDERIZANDO / nombre)
        for nombre in listar_carpetas(DIR_RENDERIZANDO)
        if any(p.suffix.lower() == ".mp4" for p in (DIR_RENDERIZANDO / nombre).iterdir())
    ]


def estado(esperar_render=False, intervalo=30, maximo_s=ESPERA_RENDER_MAX_S):
    if esperar_render:
        limite = time.monotonic() + maximo_s
        while render_en_curso() and time.monotonic() < limite:
            time.sleep(intervalo)
    return {
        "ok": True,
        "pendientes": leer_pendientes(),
        "por_aprobar": albumes_por_aprobar(),
        "render_en_curso": render_en_curso(),
    }


def guardar_pendientes(pendientes):
    PENDIENTES_JSON.parent.mkdir(parents=True, exist_ok=True)
    temporal = PENDIENTES_JSON.with_name(f"{PENDIENTES_JSON.name}.{os.getpid()}.tmp")
    with open(temporal, "w", encoding="utf-8") as f:
        json.dump(pendientes, f, ensure_ascii=False, indent=2)
    temporal.replace(PENDIENTES_JSON)


def registrar_pendiente(video_id, titulo, album, subido_en):
    with bloqueo("estado"):
        pendientes = [p for p in leer_pendientes() if p["video_id"] != video_id]
        pendientes.append(
            {
                "video_id": video_id,
                "titulo": titulo,
                "album": album,
                "subido_en": subido_en.isoformat(),
            }
        )
        guardar_pendientes(pendientes)


def momento_procesado(video_id, ahora):
    """Primera vez que se vio el video procesado: el margen corre desde ahí."""
    with bloqueo("estado"):
        pendientes = leer_pendientes()
        pendiente = next((p for p in pendientes if p["video_id"] == video_id), None)
        if pendiente is None:
            pendiente = {"video_id": video_id, "titulo": video_id, "album": None, "subido_en": None}
            pendientes.append(pendiente)
        if not pendiente.get("procesado_en"):
            pendiente["procesado_en"] = ahora.isoformat()
            guardar_pendientes(pendientes)
        return datetime.fromisoformat(pendiente["procesado_en"])


def buscar_pendiente(video_id):
    for pendiente in leer_pendientes():
        if pendiente["video_id"] == video_id:
            return pendiente
    return None


def registrar_programado(video_id, titulo, publica_en, con_reclamo):
    PROGRAMADOS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRAMADOS_JSONL, "a", encoding="utf-8") as f:
        registro = {
            "video_id": video_id,
            "titulo": titulo,
            "publica_en": publica_en.isoformat(),
            "con_reclamo": con_reclamo,
            "programado_en": datetime.now().astimezone().isoformat(),
        }
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    with bloqueo("estado"):
        guardar_pendientes([p for p in leer_pendientes() if p["video_id"] != video_id])


def fechas_programadas_localmente():
    if not PROGRAMADOS_JSONL.exists():
        return []
    fechas = []
    with open(PROGRAMADOS_JSONL, "r", encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                fechas.append(datetime.fromisoformat(json.loads(linea)["publica_en"]))
    return fechas


# ---------------------------------------------------------------------------
# Horarios: 24 por día, uno por hora
# ---------------------------------------------------------------------------


def horas_ocupadas(fechas):
    return {(fecha.date(), fecha.hour) for fecha in fechas}


def siguiente_hora_libre(desde, ocupadas):
    hora = desde.replace(minute=0, second=0, microsecond=0)
    if hora < desde:
        hora += timedelta(hours=1)
    while (hora.date(), hora.hour) in ocupadas:
        hora += timedelta(hours=1)
    return hora


def sumar_dos_anios(momento):
    try:
        return momento.replace(year=momento.year + 2)
    except ValueError:  # 29 de febrero
        return momento.replace(year=momento.year + 2, day=28)


def siguiente_momento_libre(ahora, tomados, separacion=SEPARACION_ENTRE_VIDEOS):
    """El primer momento desde `ahora` a `separacion` o más de todos los `tomados`."""
    momento = ahora
    for tomado in sorted(tomados):
        # Ordenados: si el siguiente queda cerca, se espera a que cumpla su hora y se sigue.
        if abs(momento - tomado) < separacion:
            momento = tomado + separacion
    return momento


def elegir_fecha(ahora, con_reclamo, tomados):
    """Cuándo sale el video. Sin reclamo puede devolver `ahora`: se publica en el acto."""
    if con_reclamo:
        return siguiente_hora_libre(sumar_dos_anios(ahora), horas_ocupadas(tomados))
    momento = siguiente_momento_libre(ahora, tomados)
    if momento <= ahora:
        return ahora
    return max(momento, ahora + timedelta(minutes=ANTICIPACION_MIN))


def armar_status_programado(status_actual, publica_en):
    # videos.update borra los campos de status que no se mandan.
    status = {
        campo: status_actual[campo]
        for campo in CAMPOS_STATUS_CONSERVADOS
        if campo in status_actual
    }
    status["privacyStatus"] = "private"
    status["publishAt"] = publica_en.isoformat()
    return status


def armar_status_publicado(status_actual):
    status = armar_status_programado(status_actual, datetime.now().astimezone())
    status["privacyStatus"] = "public"
    del status["publishAt"]
    return status


def clasificar_reclamo(botones_detalle, texto_pagina, en_pagina_copyright):
    if not en_pagina_copyright:
        return "desconocido"
    if botones_detalle > 0:
        return "con_reclamo"
    # Sin reclamo solo si no hay nada que lo contradiga: publicar un video con reclamo
    # pierde ingresos y no se deshace.
    texto = texto_pagina or ""
    if P_SIN_PROBLEMAS.search(texto) and not P_CON_RECLAMO.search(texto):
        return "sin_reclamo"
    return "desconocido"


# ---------------------------------------------------------------------------
# renderizar
# ---------------------------------------------------------------------------


def cola_de_log(log, lineas=20):
    with open(log, "r", encoding="utf-8", errors="replace") as f:
        texto = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", f.read())
    # Las barras de progreso reescriben la línea con \r: queda la última versión.
    limpias = [linea.split("\r")[-1].rstrip() for linea in texto.splitlines()]
    return "\n".join([linea for linea in limpias if linea][-lineas:])


def _python_corre(coincide):
    """True si algún proceso de Python tiene argumentos que cumplen `coincide(argumentos)`."""
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argumentos = cmdline.read_bytes().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        # Solo Python cuenta: un editor con el script abierto también lo tiene en sus argumentos.
        if Path(argumentos[0]).name.startswith("python") and coincide(argumentos[1:]):
            return True
    return False


def render_en_curso():
    """True si corre un render, lo haya lanzado esta sesión, otra o el usuario a mano."""
    return _python_corre(lambda argumentos: any(a.endswith(PROCESO_RENDER) for a in argumentos))


def analisis_en_curso():
    """True si corre un `censurar` (el modelo ocupa la GPU), lo haya lanzado el servicio u otra sesión."""
    return _python_corre(
        lambda argumentos: any(
            a.endswith(SCRIPT_PIPELINE) and argumentos[i + 1 : i + 2] == [SUBCOMANDO_CENSURA]
            for i, a in enumerate(argumentos)
        )
    )


def limpiar_temporales_render():
    """Borra lo que los renders dejaron en su carpeta temporal del SSD, menos los videos rescatados.

    Al salir, el render vacía solo el ramdisk, y uno cortado no limpia nada: cada álbum fallido
    dejaba ahí su copia y su video a medias. Son copias (el álbum sigue en su carpeta de origen),
    pero borrarlas con un render corriendo lo rompería: solo se llama cuando no hay ninguno.
    """
    if not SSD_TEMP_DIR.is_dir():
        return
    for entrada in SSD_TEMP_DIR.iterdir():
        if entrada.name == CARPETA_RESCATE_RENDER:
            continue
        # Lo que no se puede borrar ahora se intenta de nuevo en el próximo render.
        try:
            if entrada.is_dir() and not entrada.is_symlink():
                shutil.rmtree(entrada)
            else:
                entrada.unlink()
        except OSError:
            continue


def renderizar(carpeta=None):
    # Dos renders 4K a la vez llenan el disco y la GPU.
    if render_en_curso():
        return {"ok": False, "motivo": "render_en_curso"}
    # El modelo que analiza las portadas ocupa la GPU: el render espera a que termine.
    if analisis_en_curso():
        return {"ok": False, "motivo": "analisis_en_curso"}
    # El álbum renderizado queda en el SSD: sin él montado no se renderiza.
    if not os.path.ismount(RENDER_SSD_DIR):
        return {
            "ok": False,
            "motivo": "disco_no_disponible",
            "detalle": f"el SSD {RENDER_SSD_DIR} no está montado",
        }
    # Antes de medir el espacio: lo que dejó un render cortado no cuenta como ocupado.
    limpiar_temporales_render()
    try:
        libre_gb = shutil.disk_usage(RENDER_SSD_DIR).free / 1024**3
    except OSError as exc:
        return {"ok": False, "motivo": "disco_no_disponible", "detalle": str(exc)}
    if libre_gb < MIN_FREE_SPACE_GB:
        return {"ok": False, "motivo": "sin_espacio", "libre_gb": round(libre_gb, 1)}

    apartados = []
    if carpeta:
        # Solo un álbum listado: "..", por ejemplo, haría mover la carpeta madre entera.
        candidatas = [o / carpeta for o in DIRS_ORIGEN_RENDER if carpeta in listar_carpetas(o)]
        if not candidatas:
            return {"ok": False, "motivo": "album_no_encontrado", "album": carpeta}
        origen = candidatas[0]
    else:
        origen, apartados = siguiente_sin_publicar()
        if origen is None:
            return {"ok": False, "motivo": "sin_albumes", "ya_publicados": apartados}

    renderizando_antes = set(listar_carpetas(DIR_RENDERIZANDO))
    error_antes = set(listar_carpetas(DIR_ERROR404))
    # Siempre con la ruta: sin ella, el render solo mira 01_limpieza_de_impurezas.
    comando = [sys.executable, "2. ffmpeg_render.py", "--test", str(origen)]

    log = ruta_log("render")
    with open(log, "w", encoding="utf-8") as salida:
        # El render sale con 0 aunque falle: el resultado se lee de las carpetas.
        subprocess.run(
            comando, cwd=PROJECT_ROOT, stdout=salida, stderr=subprocess.STDOUT, check=False
        )
    # Un render que falla deja su temporal, con el video a medias (decenas de GB). Si mientras
    # tanto alguien lanzó otro a mano, ese temporal es suyo: se limpia en el próximo render.
    if not render_en_curso():
        limpiar_temporales_render()

    nuevas = [c for c in listar_carpetas(DIR_RENDERIZANDO) if c not in renderizando_antes]
    restantes = sum(len(listar_carpetas(o)) for o in DIRS_ORIGEN_RENDER)
    for nombre in nuevas:
        destino = DIR_RENDERIZANDO / nombre
        video = uploader_real().encontrar_video(destino)
        if video:
            return {
                "ok": True,
                "album": nombre,
                "carpeta": str(destino),
                "video": str(video),
                "restantes": restantes,
                "ya_publicados": apartados,
                "log": str(log),
            }

    album = origen.name
    con_error = [c for c in listar_carpetas(DIR_ERROR404) if c not in error_antes]
    return {
        "ok": False,
        "motivo": "render_fallido",
        "album": album,
        "movido_a_error404": bool(con_error),
        "restantes": restantes,
        "ya_publicados": apartados,
        "log": str(log),
        "log_final": cola_de_log(log),
    }


# ---------------------------------------------------------------------------
# verificar
# ---------------------------------------------------------------------------


def sondear(ruta):
    resultado = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", str(ruta),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if resultado.returncode != 0:
        return None
    return json.loads(resultado.stdout)


def medir_audio(ruta):
    resultado = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(ruta), "-vn",
            "-af", f"silencedetect=n=-50dB:d={SILENCIO_LARGO_S},volumedetect",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    texto = resultado.stderr
    medio = re.search(r"mean_volume:\s*(-?[\d.]+|-inf) dB", texto)
    silencios = [
        (float(inicio), float(duracion))
        for inicio, duracion in zip(
            re.findall(r"silence_start:\s*(-?[\d.]+)", texto),
            re.findall(r"silence_duration:\s*([\d.]+)", texto),
        )
    ]
    return {
        "volumen_medio_db": float(medio.group(1)) if medio else None,
        "silencios": silencios,
    }


def duracion_esperada(carpeta, intro=INTRO_DURATION):
    extensiones = {ext.lower() for ext in AUDIO_FORMATS}
    pistas = [p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in extensiones]
    if not pistas:
        return None
    total = 0.0
    for pista in pistas:
        sondeo = sondear(pista)
        if not sondeo:
            return None
        total += float(sondeo.get("format", {}).get("duration", 0))
    return intro + total


def evaluar_video(sondeo, audio, esperada, ancho, alto):
    def check(nombre, ok, detalle, nivel="error"):
        return {"nombre": nombre, "ok": ok, "nivel": nivel, "detalle": detalle}

    if not sondeo:
        return [check("abre", False, "el archivo no abre")]
    streams = sondeo.get("streams", [])
    duracion = float(sondeo.get("format", {}).get("duration", 0) or 0)
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    pista_audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    checks = [check("abre", duracion > 0, f"dura {formato_duracion(duracion)}")]
    if video:
        tamanio = f"{video.get('width')}x{video.get('height')}"
        checks.append(
            check("resolucion", (video.get("width"), video.get("height")) == (ancho, alto), tamanio)
        )
    else:
        checks.append(check("resolucion", False, "no tiene imagen"))

    if not pista_audio:
        checks.append(check("tiene_audio", False, "no tiene pista de audio"))
        return checks
    checks.append(check("tiene_audio", True, "tiene pista de audio"))

    medio = audio.get("volumen_medio_db")
    sonando = medio is not None and medio > VOLUMEN_MEDIO_MINIMO_DB
    detalle = "sin medición" if medio is None else f"volumen medio {medio:.1f} dB"
    checks.append(check("no_silencio", sonando, detalle))

    if esperada is None:
        checks.append(check("duracion", False, "no se pudo medir el audio del álbum", "aviso"))
    else:
        diferencia = abs(duracion - esperada)
        checks.append(
            check(
                "duracion",
                diferencia <= TOLERANCIA_DURACION_S,
                f"{formato_duracion(duracion)}, esperado {formato_duracion(esperada)}",
            )
        )

    for inicio, largo in audio.get("silencios", [])[:5]:
        checks.append(
            check(
                "silencio_largo",
                False,
                f"{int(largo)} s de silencio en {formato_duracion(inicio)}",
                "aviso",
            )
        )
    return checks


ETIQUETAS = {
    "abre": "el archivo abre",
    "resolucion": "resolución",
    "tiene_audio": "tiene sonido",
    "no_silencio": "no está en silencio",
    "duracion": "duración",
    "silencio_largo": "silencio largo",
}


def armar_resumen(album, duracion, checks):
    lineas = [f"{album} · {formato_duracion(duracion)}"]
    for c in checks:
        marca = "✓" if c["ok"] else ("⚠️" if c["nivel"] == "aviso" else "❌")
        lineas.append(f"{marca} {ETIQUETAS.get(c['nombre'], c['nombre'])}: {c['detalle']}")
    return "\n".join(lineas)


def generar_captura(video, duracion, destino):
    instante = duracion / 2
    destino.parent.mkdir(parents=True, exist_ok=True)
    resultado = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{instante:.2f}",
            "-i", str(video), "-frames:v", "1", "-vf", "scale=1280:-2", "-q:v", "3",
            str(destino),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return destino if resultado.returncode == 0 and destino.exists() else None


# Alto de la ventana de la búsqueda: entran unos ocho resultados. Telegram
# acepta fotos de hasta 10000 px sumando ancho y alto.
VENTANA_BUSQUEDA = {"width": 1280, "height": 2560}
ESPERA_MINIATURAS_MS = 15000
MINIATURAS_CARGADAS_JS = """() => {
    const visibles = [...document.images].filter((img) => {
        const r = img.getBoundingClientRect();
        return r.width >= 20 && r.bottom > 0 && r.top < window.innerHeight;
    });
    return visibles.length > 0 && visibles.every((img) => img.complete && img.naturalWidth > 0);
}"""


def esperar_miniaturas(pagina, timeout_ms=ESPERA_MINIATURAS_MS):
    """Espera a que carguen las imágenes visibles. Devuelve False si se acaba el tiempo.

    YouTube muestra los resultados y recién después pone la URL de cada
    miniatura: capturar antes deja los recuadros en blanco.
    """
    from playwright.sync_api import Error as PlaywrightError

    try:
        pagina.wait_for_function(MINIATURAS_CARGADAS_JS, timeout=timeout_ms)
    except PlaywrightError:
        return False
    return True


def capturar_busqueda(carpeta, uploader=None):
    """Captura la búsqueda pública del álbum en YouTube, para ver si ya está publicado.

    Usa el mismo título que la verificación manual de `3. subir_video_API.py`.
    Navegador propio y sin perfil: no choca con la revisión de Studio.
    """
    carpeta = Path(carpeta)
    uploader = uploader or uploader_real()
    titulo = uploader.build_preview_title(carpeta, uploader.cargar_repertorio())
    destino = DIR_CAPTURAS / f"busqueda_{slug(carpeta.name)}.png"
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        print(f"Sin Playwright, no hay captura de la búsqueda: {exc}")
        return None, titulo
    try:
        with sync_playwright() as playwright:
            navegador = playwright.chromium.launch(headless=True)
            try:
                pagina = navegador.new_page(viewport=VENTANA_BUSQUEDA, locale="es-419")
                pagina.goto(
                    f"https://www.youtube.com/results?search_query={quote(titulo)}",
                    wait_until="domcontentloaded",
                    timeout=60000,
                )
                try:
                    pagina.wait_for_selector("ytd-video-renderer", timeout=20000)
                except PlaywrightError:
                    pass  # Sin resultados también sirve: se captura lo que haya.
                # Si alguna miniatura no carga, se captura igual.
                esperar_miniaturas(pagina)
                pagina.screenshot(path=str(destino))
            finally:
                navegador.close()
    except PlaywrightError as exc:
        print(f"No se pudo capturar la búsqueda en YouTube: {exc}")
        return None, titulo
    return destino, titulo


def verificar(carpeta, ancho=VIDEO_WIDTH, alto=VIDEO_HEIGHT, intro=INTRO_DURATION):
    carpeta = Path(carpeta)
    # El mismo archivo que después elige `subir`.
    video = uploader_real().encontrar_video(carpeta) if carpeta.is_dir() else None
    if not video:
        return {"ok": False, "motivo": "sin_video", "carpeta": str(carpeta)}
    sondeo = sondear(video)
    audio = medir_audio(video) if sondeo else {}
    checks = evaluar_video(sondeo, audio, duracion_esperada(carpeta, intro), ancho, alto)
    duracion = float((sondeo or {}).get("format", {}).get("duration", 0) or 0)
    captura = generar_captura(video, duracion, DIR_CAPTURAS / f"{slug(carpeta.name)}.jpg") if sondeo else None
    errores = [c for c in checks if not c["ok"] and c["nivel"] == "error"]
    captura_busqueda, busqueda = capturar_busqueda(carpeta)
    return {
        "ok": not errores,
        "album": carpeta.name,
        "video": str(video),
        "captura": str(captura) if captura else None,
        "captura_busqueda": str(captura_busqueda) if captura_busqueda else None,
        "busqueda": busqueda,
        "checks": checks,
        "resumen": armar_resumen(carpeta.name, duracion, checks),
    }


# ---------------------------------------------------------------------------
# subir
# ---------------------------------------------------------------------------


def iniciar_deathgrind(uploader, generos):
    try:
        sesion = uploader.crear_sesion_autenticada()
    except OSError as exc:
        print(f"Error al iniciar sesion DeathGrind: {exc}")
        return None
    if sesion is not None and not uploader.deathgrind_smoke_test(sesion, generos):
        uploader.DEATHGRIND_DISABLED = True
        print("DeathGrind desactivado para esta ejecucion por fallas en la prueba.")
    return sesion


def subir_con_rotacion(uploader, youtube, video, metadata):
    while True:
        try:
            return uploader.subir_video(
                youtube,
                video,
                metadata["title"],
                metadata["description"],
                privacy="private",
                publish_at=None,
                label=metadata.get("title"),
            )
        except uploader.QuotaExceededError as exc:
            print(f"Cuota agotada: {exc}. Rotando credencial de upload...")
        except RefreshError as exc:
            if not uploader.is_invalid_grant_error(exc):
                raise
            print(f"Token de upload revocado/expirado: {exc}. Rotando...")
        youtube = uploader.authenticate_next(prefix="upload")
        if youtube is None:
            raise SinCredenciales("No hay mas credenciales de upload utilizables.")


def borrar_album(uploader, carpeta, raiz):
    """Borra la carpeta del álbum (audio, portada y video) y sus capturas."""
    borrada = uploader.safe_delete_folder(carpeta, raiz)
    borrar_capturas(carpeta.name)
    return bool(borrada)


def borrar_capturas(album):
    for captura in (
        DIR_CAPTURAS / f"{slug(album)}.jpg",
        DIR_CAPTURAS / f"busqueda_{slug(album)}.png",
    ):
        captura.unlink(missing_ok=True)


def poner_miniatura(uploader, youtube, video_id, video, captura):
    """Pone de miniatura el fotograma del medio del video; sin esto YouTube elige uno del intro.

    Usa la captura de `verificar`; si no está (una subida a mano), la saca. Un fallo no deshace
    la subida: devuelve (False, motivo) y la miniatura se pone a mano en Studio.
    """
    try:
        if not captura.exists():
            sondeo = sondear(video)
            duracion = float((sondeo or {}).get("format", {}).get("duration", 0) or 0)
            if not duracion or not generar_captura(video, duracion, captura):
                return False, "no se pudo sacar el fotograma del video"
        ejecutar_api(
            uploader,
            {"cliente": youtube},
            lambda c: c.thumbnails().set(
                videoId=video_id, media_body=MediaFileUpload(str(captura), mimetype="image/jpeg")
            ).execute(),
        )
    except (HttpError, RefreshError, SinCredenciales, OSError, *ERRORES_DE_CONEXION) as exc:
        print(f"No se pudo poner la miniatura de {video_id}: {exc}")
        return False, str(exc)
    return True, None


def buscar(uploader=None, aleatorio=False, excluir=()):
    """Elige el próximo álbum sin renderizarlo y captura su búsqueda en YouTube.

    El servicio pregunta con esa captura y la portada antes de renderizar: un álbum que ya está
    publicado se borra sin gastar el render. Los que están en el registro se apartan como en
    `renderizar`. Sin portada en la carpeta (puede venir dentro del audio), va solo la búsqueda.
    Con `aleatorio` (el /aleatorio del servicio), elige al azar entre las carpetas de origen.
    Salta los de `excluir`: los que ya están en la lista del /crear_lista.
    """
    origen, apartados = siguiente_sin_publicar(aleatorio, excluir)
    if origen is None:
        return {"ok": False, "motivo": "sin_albumes", "ya_publicados": apartados}
    captura, busqueda = capturar_busqueda(origen, uploader)
    portada = elegir_portada(origen)
    return {
        "ok": True,
        "album": origen.name,
        "carpeta": str(origen),
        "portada": str(portada) if portada else None,
        "captura_busqueda": str(captura) if captura else None,
        "busqueda": busqueda,
        "ya_publicados": apartados,
    }


# ---------------------------------------------------------------------------
# Censura de portadas
# ---------------------------------------------------------------------------


def datos_de_censura(album):
    return DIR_CENSURA_DATOS / slug(album)


def escribir_json(ruta, datos):
    ruta.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")


def origen_de_censura(album):
    """La carpeta donde estaba el álbum antes de mandarlo a Censura/; None si no quedó anotada."""
    try:
        origen = json.loads((datos_de_censura(album) / "origen.json").read_text(encoding="utf-8"))
        return Path(origen["carpeta"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def limpiar_datos_de_censura(album):
    """Borra lo anotado del análisis de un álbum; las portadas originales guardadas se quedan."""
    datos = datos_de_censura(album)
    if not datos.is_dir():
        return
    for entrada in datos.iterdir():
        if entrada.is_file():
            entrada.unlink(missing_ok=True)
    try:
        datos.rmdir()
    except OSError:
        pass  # todavía tiene `originales/`


def cajas_de_censura(album, hasta):
    """Las cajas ya encontradas en los intentos 1..`hasta`: [(etiqueta, x1, y1, x2, y2)]."""
    cajas = []
    for intento in range(1, hasta + 1):
        try:
            leidas = json.loads((datos_de_censura(album) / f"cajas_{intento}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cajas += [tuple(caja) for caja in leidas]
    return cajas


def a_censura(carpeta):
    """Aparta el álbum de su carpeta de origen en Censura/ y anota de dónde salió.

    Es lo primero que pasa al tocar «🔞 Censurar»: el próximo álbum se pregunta sin esperar al
    análisis. Un Censurar nuevo empieza de cero (las cajas de uno anterior no cuentan), pero
    nunca borra las portadas originales que Aceptar guardó antes. Repetirlo con el álbum ya
    apartado (el servicio se reinició a mitad) dice que está.
    """
    carpeta = Path(carpeta)
    album = carpeta.name
    destino = Path(DIR_CENSURA) / album
    if not carpeta.is_dir():
        if destino.is_dir() and origen_de_censura(album) is not None:
            return {"ok": True, "album": album, "carpeta": str(destino), "reanudado": True}
        return {"ok": False, "motivo": "carpeta_no_existe", "carpeta": str(carpeta)}
    raiz = raiz_borrable(carpeta, DIRS_ORIGEN_RENDER)
    # Solo un álbum entero de una carpeta de origen, como en `eliminar`.
    if raiz is None or carpeta.resolve().parent != raiz:
        return {"ok": False, "motivo": "carpeta_fuera_de_raiz", "carpeta": str(carpeta)}
    if destino.exists():
        return {"ok": False, "motivo": "ya_existe", "carpeta": str(destino)}
    carpeta = carpeta.resolve()
    with bloqueo("origenes"):
        try:
            destino.parent.mkdir(parents=True, exist_ok=True)
            limpiar_datos_de_censura(album)
            datos_de_censura(album).mkdir(parents=True, exist_ok=True)
            escribir_json(datos_de_censura(album) / "origen.json", {"carpeta": str(carpeta)})
            # rename y no shutil.move: si falla, shutil.move copiaría el álbum entero.
            carpeta.rename(destino)
        except OSError as exc:
            return {"ok": False, "motivo": "no_se_pudo_mover", "album": album, "detalle": str(exc)}
    return {"ok": True, "album": album, "carpeta": str(destino), "origen": str(carpeta.parent)}


def carpeta_de_censura(carpeta):
    """`carpeta` como Path si es un álbum de Censura/; si no, el resultado de error que corresponde."""
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        return None, {"ok": False, "motivo": "carpeta_no_existe", "carpeta": str(carpeta)}
    if carpeta.resolve().parent != Path(DIR_CENSURA).resolve():
        return None, {"ok": False, "motivo": "carpeta_fuera_de_censura", "carpeta": str(carpeta)}
    return carpeta, None


def vista_para_el_celular(imagen, destino):
    """La imagen achicada para juzgarla en el celular (Telegram igual la achica a 1280 px)."""
    destino.write_bytes(censura_portadas.a_jpeg(imagen, censura_portadas.LADO_DE_LA_VISTA))
    return destino


def censurar(carpeta, intento, cliente=None):
    """Busca con el modelo local qué tapar de la portada y deja la candidata censurada.

    Intento 1 mira la imagen entera, el 2 cuatro pedazos y del 3 en adelante nueve más la entera.
    Suma las cajas de los intentos anteriores: la censura solo crece. Solo lee la portada y escribe
    en `DIR_CENSURA_DATOS`: nada cambia dentro de la carpeta del álbum hasta `aceptar_censura`, y
    repetir un intento (el servicio se reinició a mitad) reemplaza sus cajas. Nunca corre con un
    render en marcha, y al terminar descarga el modelo de la GPU, haya salido bien o mal.
    """
    carpeta, error = carpeta_de_censura(carpeta)
    if error:
        return error
    album = carpeta.name
    if render_en_curso():
        return {"ok": False, "motivo": "render_en_curso"}
    portada = elegir_portada(carpeta)
    if portada is None:
        return {"ok": False, "motivo": "sin_portada", "album": album, "intento": intento}
    try:
        imagen = censura_portadas.abrir_portada(portada)
    except censura_portadas.ErrorDeCensura as exc:
        return {"ok": False, "motivo": exc.motivo, "detalle": exc.detalle, "album": album, "intento": intento}
    cliente = cliente or censura_portadas.ClienteOllama()
    fallo = None
    try:
        busqueda = censura_portadas.buscar_cajas(imagen, intento, cliente)
    except censura_portadas.ErrorDeCensura as exc:
        fallo = exc
    finally:
        descargado = cliente.descargar()
    if fallo is not None:
        return {
            "ok": False, "motivo": fallo.motivo, "detalle": fallo.detalle, "album": album,
            "intento": intento, "descargado": descargado,
        }
    nuevas = busqueda.cajas

    datos = datos_de_censura(album)
    datos.mkdir(parents=True, exist_ok=True)
    anteriores = cajas_de_censura(album, intento - 1)
    zonas = censura_portadas.ampliar_cajas(
        [caja[1:] for caja in [*anteriores, *nuevas]], *imagen.size
    )
    candidata = datos / "candidata.png"
    if zonas:
        censura_portadas.pintar_cajas(imagen, zonas).save(candidata)
    else:
        candidata.unlink(missing_ok=True)
    original = vista_para_el_celular(imagen, datos / "vista_original.jpg")
    censurada = None
    if zonas:
        censurada = vista_para_el_celular(censura_portadas.pintar_cajas(imagen, zonas), datos / "vista_censurada.jpg")
    # Lo último: con las cajas anotadas, el intento cuenta como hecho.
    escribir_json(datos / f"cajas_{intento}.json", [list(caja) for caja in nuevas])
    etiquetas = []
    for caja in nuevas:
        legible = censura_portadas.etiqueta_legible(caja[0])
        if legible not in etiquetas:
            etiquetas.append(legible)
    return {
        "ok": True,
        "album": album,
        "intento": intento,
        "encontradas": etiquetas,
        "nuevas": len(nuevas),
        "zonas": len(zonas),
        "pedazos": busqueda.pedazos,
        "sin_leer": busqueda.sin_leer,
        "original": str(original),
        "censurada": str(censurada) if censurada else None,
        "descargado": descargado,
    }


def aceptar_censura(carpeta):
    """Pone la portada censurada en el álbum y lo devuelve a la carpeta donde estaba.

    La censurada pasa a ser la que elige `elegir_portada` (así la usan el render, el fondo y la
    miniatura); la original se guarda en `originales/` y no se borra. Sin nada que tapar, el álbum
    vuelve con su portada tal cual. Si en el origen ya hay otra carpeta con ese nombre, no cambia
    nada. Repetirlo con el álbum ya devuelto (el servicio se reinició a mitad) dice que está.
    """
    carpeta = Path(carpeta)
    album = carpeta.name
    datos = datos_de_censura(album)
    marca = datos / "aplicada.json"
    if not carpeta.is_dir():
        try:
            devuelto = Path(json.loads(marca.read_text(encoding="utf-8"))["destino"])
        except (OSError, ValueError, KeyError, TypeError):
            return {"ok": False, "motivo": "carpeta_no_existe", "carpeta": str(carpeta)}
        if devuelto.is_dir():
            return {"ok": True, "album": album, "carpeta": str(devuelto), "reanudado": True}
        return {"ok": False, "motivo": "carpeta_no_existe", "carpeta": str(carpeta)}
    carpeta, error = carpeta_de_censura(carpeta)
    if error:
        return error
    analizado = any(datos.glob("cajas_*.json"))
    candidata = datos / "candidata.png"
    if not analizado:
        return {"ok": False, "motivo": "sin_analisis", "album": album}
    destino = origen_de_censura(album) or Path(DIRS_ORIGEN_RENDER[0]) / album
    if destino.exists():
        return {"ok": False, "motivo": "ya_existe", "album": album, "carpeta": str(destino)}
    originales = []
    try:
        if candidata.is_file() and not marca.exists():
            originales = reemplazar_portada(carpeta, candidata, datos / CARPETA_ORIGINALES)
            escribir_json(marca, {"destino": str(destino), "originales": [str(o) for o in originales]})
        with bloqueo("origenes"):
            carpeta.rename(destino)
    except OSError as exc:
        return {"ok": False, "motivo": "no_se_pudo_aplicar", "album": album, "detalle": str(exc)}
    escribir_json(marca, {"destino": str(destino), "originales": [str(o) for o in originales]})
    for archivo in (candidata, datos / "vista_original.jpg", datos / "vista_censurada.jpg"):
        archivo.unlink(missing_ok=True)
    return {"ok": True, "album": album, "carpeta": str(destino), "originales": [str(o) for o in originales]}


def papelera_de(raiz):
    """La papelera del disco de `raiz`, al lado de ella: mover ahí es un rename y no una copia."""
    return Path(raiz).resolve().parent / "Papelera"


def raices_de_eliminar():
    return (*RAICES_BORRABLES, *DIRS_ORIGEN_RENDER, DIR_CENSURA)


def eliminar(carpeta, raices=None):
    """Aparta en la papelera un álbum que ya estaba publicado, sin subirlo.

    El servicio pregunta antes de renderizar, así que el álbum puede estar todavía en su carpeta
    de origen; uno renderizado antes de ese cambio está en Renderizando/. Cada álbum va a una
    entrada propia de la papelera de su disco, con `origen.json` (de dónde salió y cuándo):
    `restaurar` lo devuelve y `vaciar_papelera` lo borra pasado PLAZO_PAPELERA_S.

    Uno que está en Censura/ vuelve, al deshacer, a la carpeta donde estaba antes de Censurar (no
    a Censura/, donde nada lo vuelve a preguntar), y sus datos de censura se borran; las portadas
    originales guardadas se quedan.
    """
    carpeta = Path(carpeta)
    if raices is None:
        raices = raices_de_eliminar()
    if not carpeta.is_dir():
        return {"ok": False, "motivo": "carpeta_no_existe", "carpeta": str(carpeta)}
    raiz = raiz_borrable(carpeta, raices)
    # Solo un álbum entero: en su carpeta de origen es la única copia, y una subcarpeta mal
    # pasada apartaría parte de él.
    if raiz is None or carpeta.resolve().parent != raiz:
        return {"ok": False, "motivo": "carpeta_fuera_de_raiz", "carpeta": str(carpeta)}
    # Antes de apartarlo: si el proceso muere a mitad, el álbum ya quedó anotado.
    anotar_publicados([(carpeta.name, "eliminar", "")])
    carpeta = carpeta.resolve()
    desde_censura = raiz == Path(DIR_CENSURA).resolve()
    entrada = papelera_de(raiz) / f"{datetime.now():%Y%m%d-%H%M%S}-{os.urandom(3).hex()}"
    try:
        # Sin exist_ok: una entrada con el mismo nombre es de otro álbum y no se toca.
        entrada.mkdir(parents=True)
    except OSError as exc:
        return {"ok": False, "motivo": "no_se_pudo_borrar", "album": carpeta.name, "detalle": str(exc)}
    try:
        volver_a = carpeta
        if desde_censura:
            volver_a = origen_de_censura(carpeta.name) or Path(DIRS_ORIGEN_RENDER[0]) / carpeta.name
        origen = {"carpeta": str(volver_a), "album": carpeta.name, "eliminado": time.time()}
        (entrada / "origen.json").write_text(json.dumps(origen, ensure_ascii=False), encoding="utf-8")
        # rename y no shutil.move: si falla, shutil.move copiaría el álbum entero.
        carpeta.rename(entrada / carpeta.name)
    except OSError as exc:
        shutil.rmtree(entrada, ignore_errors=True)
        return {"ok": False, "motivo": "no_se_pudo_borrar", "album": carpeta.name, "detalle": str(exc)}
    borrar_capturas(carpeta.name)
    if desde_censura:
        limpiar_datos_de_censura(carpeta.name)
    return {"ok": True, "album": carpeta.name, "papelera": str(entrada)}


def restaurar(entrada, raices=None):
    """Devuelve a su carpeta un álbum que `eliminar` apartó y saca del registro su fila.

    Solo la fila que puso `eliminar`: una del canal o de una subida se queda. El registro va
    primero y con el candado de los orígenes, así `buscar` nunca ve el álbum de vuelta con la
    fila vieja. Nunca pisa una carpeta que ya está en el lugar del álbum.
    """
    entrada = Path(entrada)
    if raices is None:
        raices = raices_de_eliminar()
    raices = {Path(raiz).resolve() for raiz in raices}
    if entrada.resolve().parent not in {papelera_de(raiz) for raiz in raices}:
        return {"ok": False, "motivo": "fuera_de_papelera", "papelera": str(entrada)}
    with bloqueo("papelera"), bloqueo("origenes"):
        try:
            destino = Path(json.loads((entrada / "origen.json").read_text(encoding="utf-8"))["carpeta"])
        except (OSError, ValueError, KeyError, TypeError):
            return {"ok": False, "motivo": "ya_no_esta", "papelera": str(entrada)}
        album = destino.name
        apartado = entrada / album
        if not apartado.is_dir():
            return {"ok": False, "motivo": "ya_no_esta", "album": album, "papelera": str(entrada)}
        if destino.parent.resolve() not in raices:
            return {"ok": False, "motivo": "carpeta_fuera_de_raiz", "album": album, "carpeta": str(destino)}
        if destino.exists():
            return {"ok": False, "motivo": "ya_existe", "album": album, "carpeta": str(destino)}
        with bloqueo("publicados"):
            registro_publicados.quitar(REGISTRO_PUBLICADOS, album, "eliminar")
        try:
            apartado.rename(destino)
        except OSError as exc:
            return {"ok": False, "motivo": "no_se_pudo_mover", "album": album, "detalle": str(exc)}
        try:
            (entrada / "origen.json").unlink()
            entrada.rmdir()
        except OSError:
            pass  # el álbum ya volvió; lo que quede de la entrada lo borra vaciar_papelera
    return {"ok": True, "album": album, "carpeta": str(destino)}


def eliminado_en(entrada):
    """Cuándo se apartó la entrada; sin `origen.json` legible, la hora de la carpeta."""
    try:
        return float(json.loads((entrada / "origen.json").read_text(encoding="utf-8"))["eliminado"])
    except (OSError, ValueError, KeyError, TypeError):
        return entrada.stat().st_mtime


def vaciar_papelera(raices=None):
    """Borra de cada papelera las entradas apartadas hace más de PLAZO_PAPELERA_S."""
    if raices is None:
        raices = raices_de_eliminar()
    vaciadas, fallos = [], []
    for papelera in sorted({papelera_de(raiz) for raiz in raices}):
        for nombre in listar_carpetas(papelera):
            entrada = papelera / nombre
            # De a una entrada: `restaurar` no espera a que se vacíe toda la papelera.
            with bloqueo("papelera"):
                try:
                    if time.time() - eliminado_en(entrada) < PLAZO_PAPELERA_S:
                        continue
                    shutil.rmtree(entrada)
                except FileNotFoundError:
                    continue  # la devolvió `restaurar` mientras tanto
                except OSError as exc:
                    fallos.append({"papelera": str(entrada), "error": str(exc)})
                    continue
            vaciadas.append(str(entrada))
    return {"ok": not fallos, "vaciadas": vaciadas, "fallos": fallos}


def subir(carpeta, uploader=None, sin_deathgrind=False, raices=RAICES_BORRABLES):
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        return {"ok": False, "motivo": "carpeta_no_existe", "carpeta": str(carpeta)}
    raiz = raiz_borrable(carpeta, raices)
    if raiz is None:
        return {"ok": False, "motivo": "carpeta_fuera_de_raiz", "carpeta": str(carpeta)}

    uploader = uploader or uploader_real()
    log = ruta_log("subir")
    with open(log, "w", encoding="utf-8") as salida, contextlib.redirect_stdout(salida):
        uploader.cargar_env()
        youtube, *_ = uploader.probar_credenciales_disponibles(prefix="upload")
        if not youtube:
            return {"ok": False, "motivo": "sin_credenciales", "log": str(log)}
        video = uploader.encontrar_video(carpeta)
        if not video:
            return {"ok": False, "motivo": "sin_video", "carpeta": str(carpeta)}
        generos, generos_lookup = uploader.cargar_generos()
        repertorio = uploader.cargar_repertorio()
        sesion = None if sin_deathgrind else iniciar_deathgrind(uploader, generos)
        metadata = uploader.procesar_carpeta(
            carpeta, sesion, generos, generos_lookup, repertorio, False
        )
        if not metadata:
            return {"ok": False, "motivo": "sin_metadata", "log": str(log)}
        try:
            respuesta = subir_con_rotacion(uploader, youtube, video, metadata)
        except uploader.UploadLimitExceededError:
            return {"ok": False, "motivo": "limite_canal", "log": str(log)}
        except SinCredenciales:
            return {"ok": False, "motivo": "sin_credenciales", "log": str(log)}
        except HttpError as exc:
            print(f"Error YouTube API: {exc}")
            return {"ok": False, "motivo": "error_api", "detalle": str(exc), "log": str(log)}

    video_id = (respuesta or {}).get("id")
    if not video_id:
        return {"ok": False, "motivo": "sin_id", "log": str(log)}
    registrar_pendiente(
        video_id, metadata["title"], carpeta.name, datetime.now().astimezone()
    )
    anotar_publicados([(carpeta.name, "subir", video_id)])
    with open(log, "a", encoding="utf-8") as salida, contextlib.redirect_stdout(salida):
        # Antes de borrar el álbum, que también borra la captura.
        miniatura, error_miniatura = poner_miniatura(
            uploader, youtube, video_id, video, DIR_CAPTURAS / f"{slug(carpeta.name)}.jpg"
        )
        borrada = borrar_album(uploader, carpeta, raiz)
    return {
        "ok": True,
        "video_id": video_id,
        "titulo": metadata["title"],
        "carpeta_borrada": borrada,
        "miniatura": miniatura,
        "miniatura_error": error_miniatura,
        "studio": f"https://studio.youtube.com/video/{video_id}/edit",
        "log": str(log),
    }


# ---------------------------------------------------------------------------
# programar
# ---------------------------------------------------------------------------


def es_error_de_conexion(exc):
    """Si `exc` es un fallo de la red y no de la máquina: los ERRORES_DE_CONEXION más los
    OSError de red inalcanzable (ERRNOS_DE_RED). Los demás OSError (archivos, permisos) no.

    Es la única definición: `ejecutar_api` y cada paso que convierte el fallo en `sin_conexion`
    atrapan POSIBLES_ERRORES_DE_CONEXION y vuelven a lanzar lo que no pase por aquí.
    """
    if isinstance(exc, ERRORES_DE_CONEXION):
        return True
    return isinstance(exc, OSError) and exc.errno in ERRNOS_DE_RED


def ejecutar_api(uploader, yt, llamada):
    fallos_de_conexion = 0
    while True:
        try:
            return llamada(yt["cliente"])
        except HttpError as exc:
            if not uploader.is_quota_error(exc):
                raise
            print(f"Cuota agotada: {exc}. Rotando credencial de upload...")
        except RefreshError as exc:
            if not uploader.is_invalid_grant_error(exc):
                raise
            print(f"Token revocado/expirado: {exc}. Rotando...")
        except POSIBLES_ERRORES_DE_CONEXION as exc:
            if not es_error_de_conexion(exc):
                raise
            # Tras una espera larga Google ya cerró la conexión keep-alive. httplib2
            # re-lanza EPIPE/ECONNRESET sin cerrar ese socket, así que se descartan
            # las conexiones para que el reintento abra una nueva.
            fallos_de_conexion += 1
            if fallos_de_conexion > REINTENTOS_CONEXION:
                raise
            espera = ESPERA_CONEXION_S * 2 ** (fallos_de_conexion - 1)
            print(
                f"Conexion fallida ({type(exc).__name__}: {exc}). "
                f"Reintento {fallos_de_conexion}/{REINTENTOS_CONEXION} en {espera}s..."
            )
            yt["cliente"].close()
            time.sleep(espera)
            continue
        yt["cliente"] = uploader.authenticate_next(prefix="upload")
        if yt["cliente"] is None:
            raise SinCredenciales("No hay mas credenciales de upload utilizables.")


def consultar_video(uploader, yt, video_id):
    respuesta = ejecutar_api(
        uploader,
        yt,
        lambda c: c.videos().list(part="status,processingDetails", id=video_id).execute(),
    )
    items = respuesta.get("items", [])
    if not items:
        return "no_existe", {}
    status = items[0].get("status", {})
    procesado = items[0].get("processingDetails", {}).get("processingStatus")
    subida = status.get("uploadStatus")
    if subida in ("failed", "rejected", "deleted") or procesado in ("failed", "terminated"):
        return "fallo", status
    if subida == "processed" or procesado == "succeeded":
        return "listo", status
    return "procesando", status


class NavegadorNoDisponible(RuntimeError):
    """No hay navegador con la sesión de Studio; `motivo` va a la respuesta."""

    def __init__(self, motivo):
        super().__init__(motivo)
        self.motivo = motivo


def version_del_navegador(ejecutable):
    if not ejecutable:  # Playwright usa su propio Chromium: no hay ejecutable que consultar
        return VERSION_STUDIO_POR_DEFECTO
    try:
        salida = subprocess.run(
            [ejecutable, "--version"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return VERSION_STUDIO_POR_DEFECTO
    coincidencia = re.search(r"(\d+)\.\d+\.\d+\.\d+", salida)
    return int(coincidencia.group(1)) if coincidencia else VERSION_STUDIO_POR_DEFECTO


def entorno_del_navegador():
    """Entorno del navegador: si el servicio arrancó antes que el escritorio (linger al
    prender el equipo) le faltan las variables del escritorio; se toman del gestor de
    usuario de systemd, que Plasma completa al iniciar sesión."""
    entorno = dict(os.environ)
    if entorno.get("XDG_CURRENT_DESKTOP"):
        return entorno
    try:
        salida = subprocess.run(
            ["systemctl", "--user", "show-environment"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return entorno
    for linea in salida.splitlines():
        nombre, _, valor = linea.partition("=")
        if nombre in VARIABLES_DE_ESCRITORIO and not entorno.get(nombre):
            entorno[nombre] = valor
    return entorno


def abrir_contexto_studio(playwright, user_data_dir, headless, ejecutable, perfil):
    """Lanza el navegador con el perfil copiado y la sesión de Google que trae."""
    kwargs = dict(
        user_data_dir=str(user_data_dir),
        headless=headless,
        args=[
            "--disable-blink-features=AutomationControlled",
            f"--profile-directory={perfil}",
        ],
        ignore_default_args=list(ARGS_QUE_ROMPEN_EL_LLAVERO),
        viewport=dict(VENTANA_STUDIO),
        user_agent=USER_AGENT_STUDIO.format(version=version_del_navegador(ejecutable)),
        env=entorno_del_navegador(),
    )
    if ejecutable:
        kwargs["executable_path"] = ejecutable
    return playwright.chromium.launch_persistent_context(**kwargs)


@functools.lru_cache(maxsize=None)
def apelacion_agente():
    apelacion = cargar_script("11. apelacion.py", "apelacion")
    # Copia propia del perfil para no pisar la de 11. apelacion.py si corre a la vez.
    apelacion.DEFAULT_RUNTIME_PROFILE_DIR = (
        PROJECT_ROOT / ".runtime_browser_profiles" / "agente"
    )
    return apelacion


@functools.lru_cache(maxsize=None)
def preparar_perfil_studio():
    """Copia el perfil del navegador con sesión (una vez por proceso).

    Devuelve (carpeta del perfil copiado, ejecutable, nombre del perfil).
    """
    apelacion = apelacion_agente()
    args = SimpleNamespace(
        user_data_dir=None,
        usar_perfil_chrome=False,
        profile=None,
        executable_path=None,
        channel=None,
    )
    try:
        user_data_dir, _, fuente = apelacion.resolve_user_data_dir(args)
    except OSError as exc:
        print(f"No se pudo preparar el perfil del navegador: {exc}")
        raise NavegadorNoDisponible("perfil_de_navegador") from exc
    if user_data_dir is None:
        raise NavegadorNoDisponible("sin_perfil_de_navegador")
    _, ejecutable = apelacion.resolve_launch_config(args, fuente)
    return user_data_dir, ejecutable, apelacion.profile_name_from_args(args)


def esta_en_login_de_google(pagina):
    return "accounts.google.com" in pagina.url


def es_pagina_de_reclamos(url, video_id):
    # Studio redirige /copyright a /claims.
    return any(f"/video/{video_id}/{ruta}" in url for ruta in ("copyright", "claims"))


def esperar_resultado_de_reclamos(pagina, ver_detalles, timeout_s=ESPERA_RESULTADO_RECLAMOS_S):
    """La lista de reclamos llega después del resumen: espera botones o el texto de 'sin reclamos'."""
    limite = time.monotonic() + timeout_s
    while True:
        texto = pagina.evaluate("() => (document.body && document.body.innerText) || ''")
        botones = pagina.get_by_role("button", name=ver_detalles).count()
        if botones > 0 or P_SIN_PROBLEMAS.search(texto):
            return botones, texto
        if time.monotonic() >= limite:
            return botones, texto
        time.sleep(1)


URL_STUDIO = "https://studio.youtube.com/"
# La cuenta tiene varios canales y a veces YouTube abre "Selecciona un canal"
# (www.youtube.com/signin_prompt) en vez de Studio: los pasos entran con este.
CANAL_DE_STUDIO = "Daniel Banariba"
ESPERA_CANAL_MS = 30_000
# La pregunta del canal puede llegar con una redirección del navegador, después de cargar.
ESPERA_PREGUNTA_DE_CANAL_MS = 10_000


def es_pregunta_de_canal(url):
    return "youtube.com/signin_prompt" in url


def entrar_a_studio(pagina):
    """Abre Studio y, si YouTube pide elegir canal, elige CANAL_DE_STUDIO.

    Lanza NavegadorNoDisponible: "canal_no_disponible" si el canal no está en la lista o
    elegirlo no lleva a Studio; "error_navegador" si Studio no carga.
    """
    from playwright.sync_api import Error as PlaywrightError

    try:
        pagina.goto(URL_STUDIO, wait_until="domcontentloaded", timeout=ESPERA_NAVEGACION_MS)
        try:
            pagina.wait_for_load_state("networkidle", timeout=ESPERA_PREGUNTA_DE_CANAL_MS)
        except PlaywrightError:
            pass
        if not es_pregunta_de_canal(pagina.url):
            return
        print(f"YouTube pidió elegir canal: elijo «{CANAL_DE_STUDIO}».")
        # Texto exacto: "Daniel Banariba Drummer" es otro canal de la misma cuenta.
        canal = pagina.get_by_text(CANAL_DE_STUDIO, exact=True).filter(visible=True).first
        canal.click(timeout=ESPERA_CANAL_MS)
        pagina.wait_for_url(lambda url: url.startswith(URL_STUDIO), timeout=ESPERA_CANAL_MS)
    except PlaywrightError as exc:
        motivo = "canal_no_disponible" if es_pregunta_de_canal(pagina.url) else "error_navegador"
        print(f"No se pudo entrar a Studio ({motivo}): {exc}")
        raise NavegadorNoDisponible(motivo) from exc


@contextlib.contextmanager
def pagina_de_studio(visible=False):
    """Entrega una página del navegador con la sesión de Studio y lo cierra al salir.

    Con `visible` abre una ventana en el escritorio, para que el usuario pase ahí lo que el
    servicio no puede ("Verifica que eres tú").

    Lanza NavegadorNoDisponible (con el motivo) si no hay perfil, no abre el navegador o
    no puede entrar a Studio con el canal. Un error al cerrar no se propaga: el trabajo ya está hecho y descartaría su resultado.
    """
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    def cerrar(que, cerrar_algo):
        try:
            cerrar_algo()
        except PlaywrightError as exc:
            print(f"No se pudo cerrar {que}: {exc}")

    user_data_dir, ejecutable, perfil = preparar_perfil_studio()
    headless = not visible and os.environ.get("AGENTE_STUDIO_HEADLESS", "1") != "0"
    driver = contextlib.ExitStack()
    try:
        playwright = driver.enter_context(sync_playwright())
    except (PlaywrightError, OSError) as exc:
        print(f"No arrancó el driver de Playwright: {exc}")
        raise NavegadorNoDisponible("error_navegador") from exc
    try:
        try:
            contexto = abrir_contexto_studio(playwright, user_data_dir, headless, ejecutable, perfil)
        except PlaywrightError as exc:
            print(f"No se pudo abrir el navegador: {exc}")
            raise NavegadorNoDisponible("error_navegador") from exc
        try:
            pagina = contexto.pages[0] if contexto.pages else contexto.new_page()
            entrar_a_studio(pagina)
            yield pagina
        finally:
            cerrar("el navegador", contexto.close)
    finally:
        cerrar("el driver de Playwright", driver.close)


def revisar_reclamo_en_studio(video_id):
    from playwright.sync_api import Error as PlaywrightError

    apelacion = apelacion_agente()
    captura = DIR_CAPTURAS / f"studio_{video_id}.png"
    captura.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pagina_de_studio() as pagina:
            try:
                reclamantes = apelacion.extract_claimants_from_video(pagina, video_id)
                if esta_en_login_de_google(pagina):
                    pagina.screenshot(path=str(captura))
                    return {
                        "estado": "desconocido",
                        "motivo": "sin_sesion_en_studio",
                        "captura_studio": str(captura),
                    }
                en_pagina_de_reclamos = es_pagina_de_reclamos(pagina.url, video_id)
                botones, texto = esperar_resultado_de_reclamos(pagina, apelacion.P_VER_DETALLES_BTN)
                pagina.screenshot(path=str(captura))
            except PlaywrightError as exc:
                print(f"Error revisando Studio: {exc}")
                return {"estado": "desconocido", "motivo": "error_navegador"}
    except NavegadorNoDisponible as exc:
        return {"estado": "desconocido", "motivo": exc.motivo}

    return {
        "estado": clasificar_reclamo(botones, texto, en_pagina_de_reclamos),
        "reclamantes": reclamantes,
        "captura_studio": str(captura),
    }


# Un video que YouTube retiró por ser copia de otro ya retirado por derechos de autor (visto en
# vivo el 2026-11-04): la API no lo lista ni a su dueño, y en /video/<id>/edit el aviso
# "Retirado" abre un globo ("...es un duplicado de otro que se ha retirado...") cuyo botón
# "Revisar problemas" abre el diálogo con quién pidió la retirada.
P_COPIA_RETIRADA = re.compile(
    r"duplicado de otro que se ha retirado|copia de otro v[ií]deo que se retir", re.IGNORECASE
)
P_DATOS_DE_LA_RETIRADA = re.compile(
    r"(Contenido cuyos derechos de autor se infringen|Solicitud recibida|De):\s*([^\n]+)"
)
ESPERA_AVISO_RETIRADO_MS = 30_000
ESPERA_DIALOGO_RETIRADO_MS = 20_000


def datos_de_la_retirada(texto):
    """Las líneas "Contenido...", "Solicitud recibida" y "De" del diálogo, una vez cada una."""
    encontrado = P_COPIA_RETIRADA.search(texto or "")
    if not encontrado:
        return []
    datos = {}
    for etiqueta, valor in P_DATOS_DE_LA_RETIRADA.findall(texto[encontrado.start():]):
        datos.setdefault(etiqueta, f"{etiqueta}: {valor.strip()}")
    return list(datos.values())


def leer_retiro_en_studio(video_id, abrir_studio=None):
    """Por qué YouTube retiró el video, según Studio.

    Solo lee: pulsa el aviso "Retirado" y "Revisar problemas", que abren información, y nada
    dentro del diálogo. Devuelve `estado` "copia_retirada" (con `detalle`, las líneas de quién
    pidió la retirada, y `captura_studio`, None si la captura falló) solo cuando Studio dice que
    es una copia; cualquier otra cosa, o una página que no se pudo leer, es "otro".
    """
    from playwright.sync_api import Error as PlaywrightError

    captura = DIR_CAPTURAS / f"retiro_{video_id}.png"
    captura_tomada = None
    try:
        with (abrir_studio or pagina_de_studio)() as pagina:
            try:
                pagina.goto(f"{URL_STUDIO}video/{video_id}/edit", wait_until="domcontentloaded")
                aviso = pagina.get_by_text("Retirado", exact=True).first
                try:
                    aviso.wait_for(state="visible", timeout=ESPERA_AVISO_RETIRADO_MS)
                except PlaywrightError:
                    return {"estado": "otro", "motivo": "sin_aviso_de_retirado"}
                aviso.click()
                globo = pagina.locator("ytcp-video-notices-tooltip-body:visible").first
                globo.wait_for(timeout=ESPERA_AVISO_RETIRADO_MS)
                if not P_COPIA_RETIRADA.search(globo.inner_text()):
                    return {"estado": "otro", "motivo": "otro_retiro"}
                detalle = []
                try:
                    globo.get_by_role("button", name="Revisar problemas").first.click(timeout=10_000)
                    pagina.get_by_text("Solicitud recibida").first.wait_for(
                        state="visible", timeout=ESPERA_DIALOGO_RETIRADO_MS
                    )
                    detalle = datos_de_la_retirada(pagina.locator("body").inner_text())
                except PlaywrightError as exc:
                    # El globo ya dice que es una copia: sin el diálogo solo falta quién la pidió.
                    print(f"No se pudo leer el diálogo del retiro: {exc}")
                try:
                    captura.parent.mkdir(parents=True, exist_ok=True)
                    pagina.screenshot(path=str(captura))
                    captura_tomada = str(captura)
                except (PlaywrightError, OSError) as exc:
                    # La copia ya está confirmada: sin foto sigue siendo una copia.
                    print(f"No se pudo tomar la captura del retiro: {exc}")
            except PlaywrightError as exc:
                print(f"Error leyendo el retiro en Studio: {exc}")
                return {"estado": "otro", "motivo": "error_navegador"}
    except NavegadorNoDisponible as exc:
        return {"estado": "otro", "motivo": exc.motivo}
    return {"estado": "copia_retirada", "detalle": detalle, "captura_studio": captura_tomada}


def apartar_copia_retirada(video_id, titulo, retiro, log):
    """Un video que nunca se va a poder publicar: sale de pendientes y su álbum queda registrado."""
    album = (buscar_pendiente(video_id) or {}).get("album")
    if album:
        with bloqueo("publicados"):
            if registro_publicados.leer(REGISTRO_PUBLICADOS).buscar(album) is None:
                registro_publicados.anotar(REGISTRO_PUBLICADOS, [(album, "subir", video_id)])
    with bloqueo("estado"):
        guardar_pendientes([p for p in leer_pendientes() if p["video_id"] != video_id])
    return {
        "ok": False,
        "motivo": "copia_retirada",
        "video_id": video_id,
        "titulo": titulo,
        "detalle": retiro.get("detalle", []),
        "captura_studio": retiro.get("captura_studio"),
        # El aviso solo puede decir que el álbum está en el registro si hay álbum (sin entrada en
        # pendientes no se anota nada); True también cuando ya estaba anotado.
        "album_registrado": bool(album),
        "log": str(log),
    }


def activar_monetizacion_en_studio(video_id):
    """Activa la monetización del video; siempre devuelve un diccionario, nunca lanza."""
    import studio_monetizacion

    captura = DIR_CAPTURAS / f"monetizacion_{video_id}.png"
    captura.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pagina_de_studio() as pagina:
            return studio_monetizacion.activar_monetizacion(pagina, video_id, captura)
    except NavegadorNoDisponible as exc:
        return {"estado": "fallo", "motivo": exc.motivo}


def errores_del_paso_de_studio():
    """Errores que el paso de Studio puede lanzar sin que sea un fallo de programación."""
    errores = [NavegadorNoDisponible, ImportError, OSError, subprocess.SubprocessError]
    try:
        from playwright.sync_api import Error as PlaywrightError
    except ImportError:
        return tuple(errores)
    return tuple(errores + [PlaywrightError])


def activar_sin_frenar(activar, video_id):
    """La monetización nunca frena la publicación: un error del paso es un fallo más."""
    try:
        return activar(video_id)
    except errores_del_paso_de_studio() as exc:
        detalle = f"{type(exc).__name__}: {exc}"[:200]
        print(f"Falló el paso de monetización: {detalle}")
        motivo = exc.motivo if isinstance(exc, NavegadorNoDisponible) else "error_navegador"
        return {"estado": "fallo", "motivo": motivo, "detalle": detalle}


def publicados_recientes(uploader, cliente, desde):
    """Cuándo se publicaron los videos públicos del canal desde `desde`.

    Solo mira la primera página de la lista de subidas: va de más nuevo a más viejo
    según el momento de publicación (o de subida, si sigue privado), así que lo
    publicado en la última hora está arriba.
    """
    lista = uploader.get_uploads_playlist_id(cliente)
    if not lista:
        return []
    items = (
        cliente.playlistItems().list(part="snippet", playlistId=lista, maxResults=50).execute()
    ).get("items", [])
    ids = [
        item["snippet"]["resourceId"]["videoId"]
        for item in items
        if item.get("snippet", {}).get("resourceId", {}).get("videoId")
    ]
    if not ids:
        return []
    videos = cliente.videos().list(part="snippet,status", id=",".join(ids)).execute()
    publicados = []
    for video in videos.get("items", []):
        if video.get("status", {}).get("privacyStatus") != "public":
            continue
        # En videos públicos publishedAt es el momento de publicación (publishAt ya no viene).
        momento = datetime.fromisoformat(video["snippet"]["publishedAt"]).astimezone()
        if momento >= desde:
            publicados.append(momento)
    return publicados


def momentos_tomados(uploader, yt, ahora, con_reclamo):
    """Momentos de otros videos: programados (API y registro local) y, sin reclamo, los recién publicados."""
    programados, _ = ejecutar_api(
        uploader,
        yt,
        lambda c: uploader.fetch_scheduled_publish_times(c, max_items=ESCANEO_PROGRAMADOS),
    )
    tomados = [*programados, *fechas_programadas_localmente()]
    if not con_reclamo:
        tomados += ejecutar_api(
            uploader,
            yt,
            lambda c: publicados_recientes(uploader, c, ahora - SEPARACION_ENTRE_VIDEOS),
        )
    return tomados


def cargar_lista_negra():
    """Las entradas de la lista negra en minúscula; [] si el archivo falta o no se puede leer."""
    return apelacion_agente().parse_filtros_reclamante_archivo(LISTA_NEGRA_RECLAMANTES)


def reclamantes_en_lista_negra(reclamantes, lista_negra):
    """Los reclamantes cuyo texto contiene alguna entrada de la lista, sin distinguir mayúsculas.

    Studio los muestra como «Believe Music, UMG» o «UMG | en nombre de X»: por eso es
    «contiene» y no igualdad. Las entradas ya vienen en minúscula y sin vacías.
    """
    return [
        reclamante
        for reclamante in reclamantes
        if any(entrada in reclamante.lower() for entrada in lista_negra)
    ]


def armar_resumen_programado(
    titulo, publica_en, con_reclamo, reclamantes, monetizacion=None, publicado_ahora=False
):
    fecha = publica_en.strftime("%Y-%m-%d %H:%M")
    if con_reclamo:
        quienes = f" ({', '.join(reclamantes)})" if reclamantes else ""
        return f"⚠️ {titulo}\nTiene reclamo{quienes}: programado para {fecha}, hasta que se resuelva el reclamo."
    cuando = "publicado ahora" if publicado_ahora else f"programado para {fecha}"
    marca = "🎉✅" if publicado_ahora else "✓"
    texto = f"{marca} {titulo}\nSin reclamos: {cuando}."
    if (monetizacion or {}).get("estado") in ("activada", "ya_activa"):
        texto += "\nMonetización activada."
    return texto


def programar(
    video_id,
    esperar=False,
    reclamo=None,
    revisar=revisar_reclamo_en_studio,
    activar=activar_monetizacion_en_studio,
    uploader=None,
    leer_retiro=leer_retiro_en_studio,
):
    uploader = uploader or uploader_real()
    # Un video que se vuelve a programar (su reclamo se liberó) ya no está en pendientes.
    titulo = (buscar_pendiente(video_id) or {}).get("titulo") or titulo_conocido(video_id)
    log = ruta_log("programar")
    with open(log, "w", encoding="utf-8") as salida, contextlib.redirect_stdout(salida):
        uploader.cargar_env()
        youtube, *_ = uploader.probar_credenciales_disponibles(prefix="upload")
        if not youtube:
            return {"ok": False, "motivo": "sin_credenciales", "log": str(log)}
        yt = {"cliente": youtube}
        try:
            while True:
                estado, status = consultar_video(uploader, yt, video_id)
                if estado == "no_existe":
                    # Studio dice si YouTube lo retiró por ser copia de otro; comparte el perfil
                    # del navegador con la revisión de reclamos.
                    with bloqueo("programar"):
                        retiro = leer_retiro(video_id)
                    if retiro["estado"] == "copia_retirada":
                        return apartar_copia_retirada(video_id, titulo, retiro, log)
                if estado in ("no_existe", "fallo"):
                    return {"ok": False, "motivo": estado, "status": status, "video_id": video_id}
                if estado == "listo":
                    break
                if not esperar:
                    return {"ok": False, "motivo": "procesando", "video_id": video_id}
                time.sleep(ESPERA_ENTRE_CONSULTAS_S)

            if status.get("privacyStatus") != "private":
                return {"ok": False, "motivo": "no_es_privado", "video_id": video_id}

            if reclamo is None:
                revisar_desde = momento_procesado(
                    video_id, datetime.now().astimezone()
                ) + timedelta(minutes=MARGEN_RECLAMO_MIN)
                faltan = (revisar_desde - datetime.now().astimezone()).total_seconds()
                if faltan > 0:
                    if not esperar:
                        return {
                            "ok": False,
                            "motivo": "esperando_margen",
                            "revisar_desde": revisar_desde.isoformat(),
                            "video_id": video_id,
                        }
                    time.sleep(faltan)

            # De a un `programar` por vez: la revisión en Studio comparte el perfil
            # del navegador, y dos procesos a la vez elegirían la misma hora libre.
            with bloqueo("programar"):
                reclamantes = []
                captura_studio = None
                if reclamo is None:
                    revision = revisar(video_id)
                    if revision["estado"] == "desconocido":
                        return {
                            "ok": False,
                            "motivo": "reclamo_desconocido",
                            "detalle": revision.get("motivo"),
                            "video_id": video_id,
                            "titulo": titulo,
                            "captura_studio": revision.get("captura_studio"),
                            "log": str(log),
                        }
                    con_reclamo = revision["estado"] == "con_reclamo"
                    reclamantes = revision.get("reclamantes", [])
                    captura_studio = revision.get("captura_studio")
                else:
                    con_reclamo = reclamo
                # Solo avisa: la fecha no cambia, borrar el video lo decide el usuario.
                en_lista_negra = []
                if con_reclamo and reclamantes:
                    en_lista_negra = reclamantes_en_lista_negra(reclamantes, cargar_lista_negra())

                # Con reclamo el video no genera ingresos: no se intenta activar nada.
                if con_reclamo:
                    monetizacion = {"estado": "omitida", "motivo": "con_reclamo"}
                else:
                    monetizacion = activar_sin_frenar(activar, video_id)
                # La fecha se elige DESPUÉS del paso de Studio, que tarda minutos: una fecha
                # calculada antes podría estar ya en el pasado al enviarla.
                ahora = datetime.now().astimezone().replace(microsecond=0)
                publica_en = elegir_fecha(
                    ahora, con_reclamo, momentos_tomados(uploader, yt, ahora, con_reclamo)
                )
                publicado_ahora = publica_en == ahora and not con_reclamo
                if publicado_ahora:
                    nuevo_status = armar_status_publicado(status)
                else:
                    nuevo_status = armar_status_programado(status, publica_en)
                cuerpo = {"id": video_id, "status": nuevo_status}
                ejecutar_api(
                    uploader,
                    yt,
                    lambda c: c.videos().update(part="status", body=cuerpo).execute(),
                )
                registrar_programado(video_id, titulo, publica_en, con_reclamo)
        except SinCredenciales:
            return {"ok": False, "motivo": "sin_credenciales", "log": str(log)}
        except HttpError as exc:
            print(f"Error YouTube API: {exc}")
            return {"ok": False, "motivo": "error_api", "detalle": str(exc), "log": str(log)}
        except POSIBLES_ERRORES_DE_CONEXION as exc:
            if not es_error_de_conexion(exc):
                raise
            detalle = f"{type(exc).__name__}: {exc}"[:200]
            print(f"Sin conexion con YouTube tras {REINTENTOS_CONEXION} reintentos: {detalle}")
            return {
                "ok": False,
                "motivo": "sin_conexion",
                "video_id": video_id,
                "detalle": detalle,
                "log": str(log),
            }

    return {
        "ok": True,
        "video_id": video_id,
        "titulo": titulo,
        "con_reclamo": con_reclamo,
        "reclamantes": reclamantes,
        "reclamantes_en_lista_negra": en_lista_negra,
        "captura_studio": captura_studio,
        "publica_en": publica_en.isoformat(),
        "publicado_ahora": publicado_ahora,
        "monetizacion": monetizacion,
        "resumen": armar_resumen_programado(
            titulo, publica_en, con_reclamo, reclamantes, monetizacion, publicado_ahora
        ),
        "log": str(log),
    }


# ---------------------------------------------------------------------------
# impugnar
# ---------------------------------------------------------------------------

# Impugnar es una declaración legal a nombre del usuario: solo lo lanza el botón de un video.
DIR_EVIDENCIA_IMPUGNAR = PROJECT_ROOT / "data" / "impugnar_evidencia"
URL_RECLAMOS = "https://studio.youtube.com/video/{video_id}/claims"
ESPERA_NAVEGACION_MS = 30_000
# Una navegación de Studio puede cortar un `evaluate`: se reintenta tras esta pausa.
PAUSA_REINTENTO_NAVEGACION_S = 2.0
INTENTOS_NAVEGACION = 3
# Las esperas por defecto de 10. inpunar_video.py: están calibradas contra Studio.
PAUSA_IMPUGNAR_S = 2.0
ESPERA_ENVIO_IMPUGNAR_S = 6.0
# Studio tarda en reflejar un envío: antes de darlo por no confirmado se recarga /claims unas veces.
INTENTOS_CONFIRMAR_IMPUGNACION = 3
ESPERA_CONFIRMAR_IMPUGNACION_S = 5.0
# Studio a veces deja el formulario abierto tras "Enviar", sin error, y no registra nada: tras
# esta pausa se vuelve a mirar /claims y, si nada cambió, ese reclamo se reintenta una sola vez.
PAUSA_REINTENTO_IMPUGNAR_S = 30.0
# Con ventana, cuánto espera cada reclamo a que el usuario pase "Verifica que eres tú".
ESPERA_VERIFICACION_VENTANA_S = 10 * 60
# Frenan `impugnar_con_ventana`: los videos que siguen fallarían igual en esa ventana (sin
# pasar la verificación, cada uno la esperaría otros 10 minutos).
FRENAN_LA_VENTANA = ("verificacion_de_identidad", "sin_sesion_en_studio", "canal_no_disponible", "error_navegador")


@functools.lru_cache(maxsize=None)
def impugnador():
    return cargar_script("10. inpunar_video.py", "inpunar_video")


def titulo_conocido(video_id):
    """El título con que se registró el video al programarlo; el id si no se encuentra."""
    titulo = None
    if PROGRAMADOS_JSONL.exists():
        with open(PROGRAMADOS_JSONL, "r", encoding="utf-8") as f:
            for linea in f:
                try:
                    registro = json.loads(linea)
                except json.JSONDecodeError:
                    continue
                if registro.get("video_id") == video_id:
                    titulo = registro.get("titulo") or titulo
    return titulo or (buscar_pendiente(video_id) or {}).get("titulo") or video_id


def contar_en_la_pagina(pagina, modulo):
    """Cuántos reclamos se pueden impugnar en la página abierta; None si no se pudo leer."""
    if esta_en_login_de_google(pagina):
        raise NavegadorNoDisponible("sin_sesion_en_studio")
    botones, texto = esperar_resultado_de_reclamos(
        pagina, apelacion_agente().P_VER_DETALLES_BTN, ESPERA_RESULTADO_RECLAMOS_S
    )
    if botones == 0:
        # Sin filas, solo el aviso de Studio de que no hay reclamaciones vale como cero. Una
        # página de error o a medio cargar no dice nada: tomarla por cero daría por hecha una
        # impugnación que Studio no registró.
        return 0 if P_SIN_PROBLEMAS.search(texto) else None
    modulo._scroll_copyright_page_to_bottom(pagina)  # la lista se dibuja a medida que se baja
    # Un índice fuera de rango no hace clic: solo cuenta los botones visibles y habilitados.
    return modulo._click_nth_take_action_in_page(pagina, 10**6)["total"]


def contar_reclamos_impugnables(pagina, video_id, modulo):
    """Cuántos "Tomar medidas" visibles y habilitados tiene /claims, tras recargarla.

    Es lo único que se toma por cierto: cada impugnación que Studio registra deja uno menos.
    Devuelve None si la página no muestra ni la lista ni el aviso de que no hay reclamos.
    Va directo a /claims: /copyright lleva allí desde el navegador, y un `evaluate` que cae
    en esa navegación falla con «Execution context was destroyed».
    """
    from playwright.sync_api import Error as PlaywrightError

    pagina.goto(
        URL_RECLAMOS.format(video_id=video_id),
        wait_until="domcontentloaded",
        timeout=ESPERA_NAVEGACION_MS,
    )
    try:
        pagina.wait_for_load_state("networkidle", timeout=12_000)
    except PlaywrightError:
        pass
    for intento in range(1, INTENTOS_NAVEGACION + 1):
        try:
            return contar_en_la_pagina(pagina, modulo)
        except PlaywrightError as exc:
            # Studio aún puede navegar justo después de cargar: se espera y se vuelve a mirar.
            if "Execution context was destroyed" not in str(exc) or intento == INTENTOS_NAVEGACION:
                raise
            print(f"Studio navegó mientras se contaban los reclamos (intento {intento}): {exc}")
            time.sleep(PAUSA_REINTENTO_NAVEGACION_S)


def contar_filas_de_reclamos(pagina):
    """Cuántos reclamos lista la página ya contada, impugnados o no (un "Ver detalles" por fila)."""
    return pagina.get_by_role("button", name=apelacion_agente().P_VER_DETALLES_BTN).count()


def contar_hasta_que_baje(pagina, video_id, modulo, antes):
    """Recarga /claims hasta ver menos reclamos impugnables que `antes`; devuelve el último conteo."""
    ahora = antes
    for intento in range(INTENTOS_CONFIRMAR_IMPUGNACION):
        if intento:
            time.sleep(ESPERA_CONFIRMAR_IMPUGNACION_S)
        ahora = contar_reclamos_impugnables(pagina, video_id, modulo)
        if ahora is not None and ahora < antes:
            break
    return ahora


def motivo_de_preparacion(estado):
    """El motivo que ve el servicio cuando el formulario de un reclamo no quedó listo para enviar."""
    if estado.startswith("2fa_no_completado"):
        return "verificacion_de_identidad"
    return "formulario_no_disponible"


def guardar_evidencia_de_impugnacion(modulo, pagina, video_id, numero, motivo, detalle=None):
    modulo.dump_dispute_evidence(
        pagina,
        {"video_id": f"{video_id}_reclamo{numero}", "detalle": detalle},
        motivo,
        str(DIR_EVIDENCIA_IMPUGNAR),
    )


def cerrar_pestana(pestana):
    """Cierra la pestaña de un reclamo: el servicio no tiene ventana donde dejarla para revisar."""
    from playwright.sync_api import Error as PlaywrightError

    if pestana is None:
        return
    try:
        if not pestana.is_closed():
            pestana.close()
    except PlaywrightError as exc:
        print(f"No se pudo cerrar la pestaña del reclamo: {exc}")


def capturar_pagina(pagina, destino):
    from playwright.sync_api import Error as PlaywrightError

    try:
        pagina.screenshot(path=str(destino))
    except PlaywrightError as exc:
        print(f"No se pudo capturar Studio: {exc}")
        return None
    return str(destino)


def impugnar_reclamos(pagina, video_id, sin_enviar, modulo, espera_verificacion_s=0.0):
    """Impugna, de a uno, los reclamos que Studio deja impugnar; devuelve el resumen del intento.

    Nunca impugna más reclamos que los que había al empezar. Cada uno cuenta como hecho solo
    si, al recargar /claims, hay exactamente un reclamo impugnable menos. Si el formulario
    quedó abierto tras "Enviar" (`form_persistente`) y /claims no cambió, Studio no lo tomó:
    tras una pausa se vuelve a mirar /claims y, si sigue igual (los mismos impugnables y las
    mismas filas), ese reclamo se reintenta una sola vez; con el formulario del reintento listo
    se mira una última vez, y si algo cambió no se envía. En cualquier otro caso (el reintento
    tampoco sale, la lista no se puede leer, la cuenta no es la esperada) se anota
    `no_confirmado` y se frena. Cualquier otro fallo también frena: no se envió nada de ese
    reclamo y los siguientes fallarían igual (sesión, 2FA, formulario).
    Con `sin_enviar` llena el formulario del primer reclamo y lo cierra sin enviarlo. Con
    `espera_verificacion_s` (solo con ventana) cada reclamo espera ese tiempo a que el usuario
    pase "Verifica que eres tú".
    """
    reclamos = contar_reclamos_impugnables(pagina, video_id, modulo)
    resumen = {"reclamos": reclamos or 0, "impugnados": 0, "fallos": []}
    if sin_enviar:
        resumen.update(sin_enviar=True, listos=0)
    fallos = resumen["fallos"]
    if reclamos is None:
        fallos.append(
            {
                "motivo": "lista_de_reclamos_no_disponible",
                "detalle": "Studio no mostró la lista de reclamos ni el aviso de que no hay",
            }
        )
        return resumen
    total = quedan = resumen["reclamos"]
    args = SimpleNamespace(
        delay=PAUSA_IMPUGNAR_S,
        espera_envio=ESPERA_ENVIO_IMPUGNAR_S,
        # Sin ventana nadie puede pasar un 2FA, así que no se espera a nadie.
        pausa_interactiva_s=espera_verificacion_s,
    )
    try:
        filas = contar_filas_de_reclamos(pagina)
        # El tope es lo que había al empezar: un reclamo que aparece después no se impugna.
        numero, reintento = 1, False
        while numero <= total:
            pestana = None
            try:
                # Siempre el primero habilitado: el que se acaba de impugnar ya no lo está.
                preparado = modulo.preparar_pestana_para_reclamo(pagina.context, video_id, 0, args, {})
                pestana = preparado["page"]
                if preparado["status"] != "ready_to_send":
                    guardar_evidencia_de_impugnacion(
                        modulo, pestana, video_id, numero, "impugnar_formulario", preparado["status"]
                    )
                    fallos.append(
                        {
                            "motivo": motivo_de_preparacion(preparado["status"]),
                            "detalle": preparado["status"][:200],
                        }
                    )
                    break
                if sin_enviar:
                    resumen["listos"] = 1
                    break
                if reintento and (
                    contar_reclamos_impugnables(pagina, video_id, modulo) != quedan
                    or contar_filas_de_reclamos(pagina) != filas
                ):
                    # Studio cambió mientras se abría el reintento (pudo mostrar el primer envío
                    # y esta pestaña tener abierto otro reclamo): no se envía, decide la cuenta.
                    envio = {"ok": False, "razon": "reintento_no_enviado"}
                else:
                    envio = modulo._enviar_una_tab(pestana, args, {}, DIR_EVIDENCIA_IMPUGNAR, numero, total)
                    if not envio["ok"] and not pestana.is_closed():
                        guardar_evidencia_de_impugnacion(
                            modulo, pestana, video_id, numero, "impugnar_no_enviado", envio["razon"]
                        )
            finally:
                cerrar_pestana(pestana)
            # Se mira Studio aunque `envio` diga que falló: Enviar pudo haber salido igual.
            despues = contar_hasta_que_baje(pagina, video_id, modulo, quedan)
            if not reintento and envio["razon"] == "form_persistente" and despues == quedan:
                # El formulario quedó abierto y /claims no cambió: Studio no tomó el envío.
                guardar_evidencia_de_impugnacion(
                    modulo, pagina, video_id, numero, "impugnar_no_registrado", envio["razon"]
                )
                print(
                    f"Studio no registró el reclamo {numero}; "
                    f"miro de nuevo en {PAUSA_REINTENTO_IMPUGNAR_S:.0f} s y lo reintento."
                )
                time.sleep(PAUSA_REINTENTO_IMPUGNAR_S)
                # Antes de reenviar se vuelve a mirar: Studio pudo mostrarlo tarde (uno menos
                # cuenta como hecho), o un reclamo nuevo pudo compensar la cuenta (otra fila).
                despues = contar_reclamos_impugnables(pagina, video_id, modulo)
                if despues == quedan and contar_filas_de_reclamos(pagina) == filas:
                    reintento = True
                    continue
            if despues != quedan - 1:
                guardar_evidencia_de_impugnacion(
                    modulo, pagina, video_id, numero, "impugnar_no_confirmado", envio["razon"]
                )
                reenviado = reintento and envio["razon"] != "reintento_no_enviado"
                tambien = ", también al reintentar" if reenviado else ""
                fallos.append(
                    {
                        "motivo": "no_confirmado",
                        "detalle": f"Studio mostraba {quedan} reclamos impugnables y ahora "
                        f"{'no se pudo leer la lista' if despues is None else despues} "
                        f"(envío: {envio['razon']}{tambien})"[:200],
                    }
                )
                break
            resumen["impugnados"] += 1
            quedan = despues
            if not quedan:
                break
            filas = contar_filas_de_reclamos(pagina)
            numero, reintento = numero + 1, False
    except errores_del_paso_de_studio() as exc:
        detalle = f"{type(exc).__name__}: {exc}"[:200]
        print(f"Falló la impugnación en Studio: {detalle}")
        motivo = exc.motivo if isinstance(exc, NavegadorNoDisponible) else "error_navegador"
        fallos.append({"motivo": motivo, "detalle": detalle})
    return resumen


def impugnar(video_id, sin_enviar=False, abrir_studio=pagina_de_studio, modulo=None):
    """Impugna en Studio los reclamos de un video. Es una declaración legal a nombre del usuario.

    Solo lo lanza el botón «Impugnar» de ese video; `sin_enviar` es la prueba en seco.
    """
    modulo = modulo or impugnador()
    titulo = titulo_conocido(video_id)
    log = ruta_log("impugnar")
    captura = DIR_CAPTURAS / f"impugnar_{video_id}.png"
    base = {"video_id": video_id, "titulo": titulo, "log": str(log)}
    with open(log, "w", encoding="utf-8") as salida, contextlib.redirect_stdout(salida):
        try:
            # El mismo candado que `programar`: los dos usan la copia del perfil de Studio y
            # dos navegadores a la vez sobre ella se pisarían.
            with bloqueo("programar"), abrir_studio() as pagina:
                captura.parent.mkdir(parents=True, exist_ok=True)
                resumen = impugnar_reclamos(pagina, video_id, sin_enviar, modulo)
                resumen["captura"] = capturar_pagina(pagina, captura)
        except errores_del_paso_de_studio() as exc:
            detalle = f"{type(exc).__name__}: {exc}"[:200]
            print(f"No se pudo impugnar en Studio: {detalle}")
            motivo = exc.motivo if isinstance(exc, NavegadorNoDisponible) else "error_navegador"
            return {"ok": False, "motivo": motivo, "detalle": detalle, **base}
    return {**resultado_de_impugnar(resumen, sin_enviar), **base}


def resultado_de_impugnar(resumen, sin_enviar):
    if resumen["reclamos"] == 0 and not resumen["fallos"]:
        return {"ok": False, "motivo": "sin_reclamos_impugnables", **resumen}
    hecho = resumen["listos"] == 1 if sin_enviar else resumen["impugnados"] == resumen["reclamos"]
    return {"ok": hecho and not resumen["fallos"], **resumen}


def impugnar_con_ventana(video_ids, sin_enviar=False, abrir_studio=None, modulo=None):
    """Impugna en una ventana visible, uno detrás del otro, los reclamos de varios videos.

    Para cuando Studio pide "Verifica que eres tú": el usuario la pasa en esa ventana y cada
    reclamo la espera hasta ESPERA_VERIFICACION_VENTANA_S. Pasada una vez, vale para los
    reclamos siguientes de la misma ventana (visto el 2026-11-07). Un fallo de
    FRENAN_LA_VENTANA frena: los demás quedan en `sin_revisar`. Es una declaración legal a
    nombre del usuario: solo la lanza el botón «Impugnar con ventana».
    """
    modulo = modulo or impugnador()
    abrir_studio = abrir_studio or functools.partial(pagina_de_studio, visible=True)
    log = ruta_log("impugnar_con_ventana")
    base = {"ventana": True, "log": str(log)}
    videos, pendientes = [], list(video_ids)

    def quedan():
        return [{"video_id": v, "titulo": titulo_conocido(v)} for v in pendientes]

    with open(log, "w", encoding="utf-8") as salida, contextlib.redirect_stdout(salida):
        try:
            with bloqueo("programar"), abrir_studio() as pagina:
                while pendientes:
                    video_id = pendientes[0]
                    captura = DIR_CAPTURAS / f"impugnar_{video_id}.png"
                    captura.parent.mkdir(parents=True, exist_ok=True)
                    resumen = impugnar_reclamos(
                        pagina, video_id, sin_enviar, modulo, espera_verificacion_s=ESPERA_VERIFICACION_VENTANA_S
                    )
                    resumen["captura"] = capturar_pagina(pagina, captura)
                    videos.append(
                        {"video_id": video_id, "titulo": titulo_conocido(video_id), **resultado_de_impugnar(resumen, sin_enviar)}
                    )
                    pendientes.pop(0)
                    frena = next((f["motivo"] for f in resumen["fallos"] if f["motivo"] in FRENAN_LA_VENTANA), None)
                    if frena:
                        return {**base, "ok": False, "motivo": frena, "videos": videos, "sin_revisar": quedan()}
        except errores_del_paso_de_studio() as exc:
            detalle = f"{type(exc).__name__}: {exc}"[:200]
            print(f"No se pudo impugnar en Studio: {detalle}")
            motivo = exc.motivo if isinstance(exc, NavegadorNoDisponible) else "error_navegador"
            return {**base, "ok": False, "motivo": motivo, "detalle": detalle, "videos": videos, "sin_revisar": quedan()}
    ok = all(v["ok"] or v.get("motivo") == "sin_reclamos_impugnables" for v in videos)
    return {**base, "ok": ok, "videos": videos, "sin_revisar": []}


# ---------------------------------------------------------------------------
# seguir_impugnaciones
# ---------------------------------------------------------------------------

# El último estado leído de cada reclamo de los videos que el servicio impugnó.
IMPUGNACIONES_JSON = DIR_AGENTE / "impugnaciones.json"
# Mientras algún reclamo dice esto, la impugnación sigue abierta y el video se sigue mirando.
P_IMPUGNACION_PENDIENTE = re.compile(r"en proceso", re.IGNORECASE)
# La corrida tiene el candado de Studio: lo que no alcanza a mirar lo mira al día siguiente.
TIEMPO_MAXIMO_SEGUIMIENTO_S = 30 * 60
# Una lista que todavía se dibuja muestra menos filas, o un rato el aviso de "sin reclamos":
# un reclamo que falta, o ninguno, cuenta solo si otra lectura unos segundos después ve lo mismo.
# Una lista vacía espera el triple (30 s): dar por liberado un video con reclamo lo publica solo.
PAUSA_RELECTURA_RECLAMOS_S = 10
ESPERA_EXTRA_LISTA_VACIA = 3
RELECTURAS_DE_RECLAMOS = 2

# Una fila por reclamo de /claims (visto en vivo el 2026-11-04): sus renglones (canción,
# artista, "Derechos de autor", "Audio", ...) y la línea de estado de `#impact-container`
# sin "Ver detalles" ni "Tomar medidas".
JS_FILAS_DE_RECLAMOS = """() => Array.from(document.querySelectorAll('ytcr-video-content-list-row')).map(fila => {
    const renglones = (el) => ((el && el.innerText) || '').split('\\n').map(t => t.trim()).filter(Boolean);
    const impacto = fila.querySelector('#impact-container');
    const botones = new Set(renglones(impacto && impacto.querySelector('.action-buttons')));
    return {renglones: renglones(fila), estado: renglones(impacto).filter(t => !botones.has(t)).join(' ')};
})"""


def leer_impugnaciones():
    if not IMPUGNACIONES_JSON.exists():
        return {}
    with open(IMPUGNACIONES_JSON, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


def guardar_impugnaciones(registro):
    IMPUGNACIONES_JSON.parent.mkdir(parents=True, exist_ok=True)
    temporal = IMPUGNACIONES_JSON.with_name(f"{IMPUGNACIONES_JSON.name}.tmp")
    with open(temporal, "w", encoding="utf-8") as archivo:
        json.dump(registro, archivo, ensure_ascii=False, indent=2)
    temporal.replace(IMPUGNACIONES_JSON)


def impugnados_por_el_servicio():
    """Los videos con algún reclamo impugnado, según el JSON final de cada `servicio_impugnar_*.out`
    (un video) o `servicio_impugnar_con_ventana_*.out` (varios, en `videos`):
    {video_id: {"titulo", "fecha"}}. `fecha` es cuándo se terminó de escribir el log más reciente
    que lo menciona (ISO, al segundo, como `revisado`): la impugnación pudo salir después de la
    última lectura del video."""
    videos = {}
    salidas = sorted(DIR_LOGS.glob("servicio_impugnar_[0-9]*.out"))
    salidas += sorted(DIR_LOGS.glob("servicio_impugnar_con_ventana_*.out"))
    for salida in salidas:
        fecha = datetime.fromtimestamp(salida.stat().st_mtime).isoformat(timespec="seconds")
        lineas = salida.read_text(encoding="utf-8", errors="replace").splitlines()
        for linea in reversed(lineas):
            if not linea.startswith("{"):
                continue
            try:
                resultado = json.loads(linea)
            except json.JSONDecodeError:
                continue
            for video in resultado.get("videos") or [resultado]:
                if video.get("impugnados") and video.get("video_id"):
                    anterior = videos.get(video["video_id"], {})
                    videos[video["video_id"]] = {
                        "titulo": video.get("titulo") or video["video_id"],
                        "fecha": max(fecha, anterior.get("fecha", "")),
                    }
            break
    return videos


def leer_reclamos(pagina, video_id, recargar=True):
    """Los reclamos que muestra /claims: [{"reclamo", "estado"}].

    [] si Studio dice que no hay reclamos; None si la página no muestra ni la lista ni ese
    aviso, o si la lista no tiene la forma conocida (una fila por "Ver detalles"). Con
    `recargar=False` vuelve a leer la página ya abierta, sin navegar. Solo lee.
    """
    from playwright.sync_api import Error as PlaywrightError

    def leer():
        if esta_en_login_de_google(pagina):
            raise NavegadorNoDisponible("sin_sesion_en_studio")
        ver_detalles = apelacion_agente().P_VER_DETALLES_BTN
        botones, texto = esperar_resultado_de_reclamos(pagina, ver_detalles, ESPERA_RESULTADO_RECLAMOS_S)
        if botones == 0:
            # Una página de error o a medio cargar no dice nada: tomarla por vacía daría por
            # retirados reclamos que siguen ahí.
            return [] if P_SIN_PROBLEMAS.search(texto) else None
        apelacion_agente()._scroll_copyright_page_to_bottom(pagina)  # la lista se dibuja al bajar
        filas = pagina.evaluate(JS_FILAS_DE_RECLAMOS)
        if not filas or len(filas) != contar_filas_de_reclamos(pagina):
            return None
        reclamos, repetidos = [], {}
        for fila in filas:
            nombre = " — ".join(fila["renglones"][:2])
            repetidos[nombre] = repetidos.get(nombre, 0) + 1
            if repetidos[nombre] > 1:
                nombre = f"{nombre} ({repetidos[nombre]})"
            reclamos.append({"reclamo": nombre, "estado": fila["estado"]})
        return reclamos

    if recargar:
        pagina.goto(URL_RECLAMOS.format(video_id=video_id), wait_until="domcontentloaded", timeout=ESPERA_NAVEGACION_MS)
        try:
            pagina.wait_for_load_state("networkidle", timeout=12_000)
        except PlaywrightError:
            pass
    for intento in range(1, INTENTOS_NAVEGACION + 1):
        try:
            return leer()
        except PlaywrightError as exc:
            # Studio aún puede navegar justo después de cargar: se espera y se vuelve a mirar.
            if "Execution context was destroyed" not in str(exc) or intento == INTENTOS_NAVEGACION:
                raise
            print(f"Studio navegó mientras se leían los reclamos (intento {intento}): {exc}")
            time.sleep(PAUSA_REINTENTO_NAVEGACION_S)


def confirmar_reclamos(pagina, video_id, antes, reclamos):
    """`reclamos` si están todos los de `antes` y la lista no está vacía; si no, relee la misma
    página tras una pausa (el triple si la lista está vacía) y vale cuando dos lecturas seguidas
    coinciden. None si no coinciden."""
    relecturas = 0
    while reclamos is not None and (
        not reclamos or not set(antes) <= {reclamo["reclamo"] for reclamo in reclamos}
    ):
        if relecturas == RELECTURAS_DE_RECLAMOS:
            return None
        time.sleep(PAUSA_RELECTURA_RECLAMOS_S * (1 if reclamos else ESPERA_EXTRA_LISTA_VACIA))
        otra = leer_reclamos(pagina, video_id, recargar=False)
        relecturas += 1
        if otra == reclamos:
            return reclamos
        print(f"{video_id}: la lista de reclamos cambió entre dos lecturas; la vuelvo a leer")
        reclamos = otra
    return reclamos


def comparar_reclamos(antes, reclamos):
    """Lo leído como {reclamo: estado} y los cambios contra `antes`: un estado distinto o un
    reclamo que ya no está (`ahora` None). La cuenta de días ("Caduca dentro de N días" o
    "1 día") no es un cambio. Sin `antes`, la lectura es la base y no hay cambios."""
    ahora = {reclamo["reclamo"]: reclamo["estado"] for reclamo in reclamos}
    # "1 día" y "N días" son la misma cuenta: se igualan junto con los números.
    sin_numeros = lambda texto: re.sub(r"\bd[ií]as?\b", "día", re.sub(r"\d+", "#", texto))
    cambios = [
        {"reclamo": nombre, "antes": estado, "ahora": ahora.get(nombre)}
        for nombre, estado in antes.items()
        if nombre not in ahora or sin_numeros(ahora[nombre]) != sin_numeros(estado)
    ]
    return ahora, cambios


def seguir_impugnaciones(abrir_studio=pagina_de_studio):
    """Lee en Studio el estado de los reclamos que impugnó el servicio y devuelve lo que cambió.

    Solo abre /claims y lee: nunca pulsa nada. Los videos salen de los `servicio_impugnar_*.out`
    con algún reclamo impugnado. La primera lectura de cada uno es la base; después, cada
    cambio se anota en IMPUGNACIONES_JSON y sale una sola vez en `cambios`. Un video sin
    reclamos "en proceso" deja de seguirse, y en ese momento sale una sola vez en `liberados`
    (Studio ya no muestra reclamos) o en `rechazados` (sigue con alguno: su estado y una
    captura de /claims). Lo lanza el servicio una vez por día, tras `apelar`.
    """
    log = ruta_log("seguir_impugnaciones")
    base = {"log": str(log)}
    registro = leer_impugnaciones()
    for video_id, impugnado in impugnados_por_el_servicio().items():
        datos = registro.setdefault(video_id, {"titulo": impugnado["titulo"]})
        # Un reclamo impugnado después de la última lectura: el video vuelve a seguirse.
        if datos.get("terminado") and impugnado["fecha"] > datos.get("revisado", ""):
            datos["terminado"] = False
    # Primero lo que hace más que no se mira: si se acaba el tiempo, mañana sigue el resto.
    seguidos = sorted(
        (video_id for video_id, datos in registro.items() if not datos.get("terminado")),
        key=lambda video_id: registro[video_id].get("revisado", ""),
    )
    cambios, fallos, liberados, rechazados = [], [], [], []
    inicio = time.monotonic()
    leyendo = None  # el video que se está leyendo; ninguno si falló abrir Studio
    with open(log, "w", encoding="utf-8") as salida, contextlib.redirect_stdout(salida):
        try:
            if seguidos:
                # El mismo candado que `programar`, `impugnar` y `apelar`: comparten el perfil.
                with bloqueo("programar"), abrir_studio() as pagina:
                    DIR_CAPTURAS.mkdir(parents=True, exist_ok=True)
                    for video_id in seguidos:
                        if time.monotonic() - inicio > TIEMPO_MAXIMO_SEGUIMIENTO_S:
                            print("Se acabó el tiempo de la corrida: el resto se mira mañana.")
                            break
                        datos = registro[video_id]
                        leyendo = video_id
                        antes = datos.get("reclamos") or {}
                        reclamos = confirmar_reclamos(pagina, video_id, antes, leer_reclamos(pagina, video_id))
                        if reclamos is None:
                            print(f"{video_id}: no se pudo leer /claims")
                            # Pasa al final de la próxima corrida: uno ilegible no va primero cada día.
                            datos["revisado"] = datetime.now().isoformat(timespec="seconds")
                            guardar_impugnaciones(registro)
                            fallos.append(
                                {"video_id": video_id, "titulo": datos.get("titulo"),
                                 "motivo": "lista_de_reclamos_no_disponible"}
                            )
                            continue
                        ahora, nuevos = comparar_reclamos(antes, reclamos)
                        terminado = not any(P_IMPUGNACION_PENDIENTE.search(e) for e in ahora.values())
                        # La captura va antes de anotar nada: si falla con un error que no es de
                        # Playwright, el video no queda terminado sin que su pregunta haya salido, y
                        # sus cambios se leen de nuevo en la próxima corrida en vez de repetirse.
                        captura = (
                            capturar_pagina(pagina, DIR_CAPTURAS / f"reclamos_{video_id}.png")
                            if terminado and ahora else None
                        )
                        cambios += [{"video_id": video_id, "titulo": datos.get("titulo"), **c} for c in nuevos]
                        datos.update(
                            reclamos=ahora,
                            revisado=datetime.now().isoformat(timespec="seconds"),
                            terminado=terminado,
                        )
                        guardar_impugnaciones(registro)
                        print(f"{video_id}: {len(ahora)} reclamos, {len(nuevos)} cambios")
                        if terminado and not ahora:
                            liberados.append({"video_id": video_id, "titulo": datos.get("titulo")})
                        elif terminado:
                            rechazados.append({
                                "video_id": video_id,
                                "titulo": datos.get("titulo"),
                                "reclamos": [{"reclamo": n, "estado": e} for n, e in ahora.items()],
                                "captura": captura,
                            })
        except errores_del_paso_de_studio() as exc:
            detalle = f"{type(exc).__name__}: {exc}"[:200]
            print(f"No se pudieron leer los reclamos en Studio: {detalle}")
            motivo = exc.motivo if isinstance(exc, NavegadorNoDisponible) else "error_navegador"
            if leyendo and not isinstance(exc, NavegadorNoDisponible):
                # El video en que falló pasa al final de la próxima corrida: si falla siempre, no
                # frena a los demás. Sin navegador o sin sesión falla todo, no un video.
                registro[leyendo]["revisado"] = datetime.now().isoformat(timespec="seconds")
                try:
                    guardar_impugnaciones(registro)
                except OSError as exc_guardar:
                    print(f"No se pudo anotar el video en que falló: {exc_guardar}")
            # Lo ya terminado no se vuelve a mirar: si no sale ahora, no sale nunca.
            return {"ok": False, "motivo": motivo, "detalle": detalle, "cambios": cambios, "fallos": fallos,
                    "liberados": liberados, "rechazados": rechazados, **base}
    return {"ok": not fallos, "seguidos": len(seguidos), "cambios": cambios, "fallos": fallos,
            "liberados": liberados, "rechazados": rechazados, **base}


# ---------------------------------------------------------------------------
# mapear_canal
# ---------------------------------------------------------------------------

# Un video con reclamo se programa a dos años (elegir_fecha): uno privado que sale a más de un
# año está aparcado por un reclamo, lo haya programado este servicio u otro script.
UMBRAL_APARCADO = timedelta(days=365)
# Cuándo se miró por última vez cada video aparcado: la corrida siguiente empieza por el resto.
MAPEO_JSON = DIR_AGENTE / "mapeo_canal.json"
# Tiene el candado de Studio, como el seguimiento: lo que no alcanza sale en `sin_revisar`.
TIEMPO_MAXIMO_MAPEO_S = 30 * 60


def videos_aparcados(uploader, cliente, ahora):
    """Los videos privados programados a más de UMBRAL_APARCADO de `ahora`, entre las últimas
    ESCANEO_PROGRAMADOS subidas del canal: [{"video_id", "titulo", "publica_en"}], en el orden
    de la lista de subidas. Los borrados o retirados por YouTube no aparecen."""
    lista = uploader.get_uploads_playlist_id(cliente)
    if not lista:
        return []
    aparcados, vistos, siguiente = [], 0, None
    while vistos < ESCANEO_PROGRAMADOS:
        respuesta = cliente.playlistItems().list(
            part="snippet", playlistId=lista, maxResults=50, pageToken=siguiente
        ).execute()
        items = respuesta.get("items", [])
        vistos += len(items)
        ids = [
            item["snippet"]["resourceId"]["videoId"]
            for item in items
            if item.get("snippet", {}).get("resourceId", {}).get("videoId")
        ]
        if ids:
            videos = cliente.videos().list(part="snippet,status", id=",".join(ids)).execute()
            por_id = {video["id"]: video for video in videos.get("items", [])}
            for video_id in ids:
                video = por_id.get(video_id, {})
                status = video.get("status", {})
                if status.get("privacyStatus") != "private" or not status.get("publishAt"):
                    continue
                publica_en = datetime.fromisoformat(status["publishAt"]).astimezone()
                if publica_en - ahora > UMBRAL_APARCADO:
                    aparcados.append({
                        "video_id": video_id,
                        "titulo": video.get("snippet", {}).get("title") or video_id,
                        "publica_en": publica_en.isoformat(),
                    })
        siguiente = respuesta.get("nextPageToken")
        if not items or not siguiente:
            break
    return aparcados


def privados_sin_fecha(uploader, yt):
    """Los videos privados sin fecha de publicación (sin `status.publishAt`) entre TODAS las
    subidas del canal que no están en pendientes.json: [{"video_id", "titulo", "subido"}], en el
    orden de la lista de subidas; `subido` es snippet.publishedAt. Solo usa la API. Un video así
    nunca se publica y ningún paso lo esperaba. Recorrer el canal entero cuesta unas 2
    unidades de cuota por cada 50 subidas (una página de la lista y un videos.list). Cada
    llamada pasa sola por `ejecutar_api`: un reintento no vuelve a empezar desde la primera página.
    """
    ids = [video_id for _, video_id in subidas_del_canal(uploader, yt)]
    sin_fecha = []
    for inicio in range(0, len(ids), 50):
        lote = ids[inicio:inicio + 50]
        videos = ejecutar_api(
            uploader, yt, lambda c: c.videos().list(part="snippet,status", id=",".join(lote)).execute()
        )
        por_id = {video["id"]: video for video in videos.get("items", [])}
        for video_id in lote:
            video = por_id.get(video_id, {})
            status = video.get("status", {})
            if status.get("privacyStatus") == "private" and not status.get("publishAt"):
                snippet = video.get("snippet", {})
                sin_fecha.append({
                    "video_id": video_id,
                    "titulo": snippet.get("title") or video_id,
                    "subido": snippet.get("publishedAt"),
                })
    # Pendientes se lee al terminar: un video que se subió mientras se recorría el canal ya
    # figura ahí, y preguntar por él sería preguntar por uno que `programar` está esperando.
    pendientes = {pendiente["video_id"] for pendiente in leer_pendientes()}
    return [video for video in sin_fecha if video["video_id"] not in pendientes]


def leer_mapeo():
    if not MAPEO_JSON.exists():
        return {}
    with open(MAPEO_JSON, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


def guardar_mapeo(mapeo):
    MAPEO_JSON.parent.mkdir(parents=True, exist_ok=True)
    temporal = MAPEO_JSON.with_name(f"{MAPEO_JSON.name}.tmp")
    with open(temporal, "w", encoding="utf-8") as archivo:
        json.dump(mapeo, archivo, ensure_ascii=False, indent=2)
    temporal.replace(MAPEO_JSON)


def mirar_video_aparcado(pagina, video, modulo, leer_reclamantes, lista_negra):
    """En qué lista va un video aparcado, según su /claims; solo lee. Devuelve (lista, datos)."""
    video_id = video["video_id"]
    datos = {"video_id": video_id, "titulo": video["titulo"], "publica_en": video["publica_en"]}
    reclamos = confirmar_reclamos(pagina, video_id, {}, leer_reclamos(pagina, video_id))
    if reclamos is None:
        return "fallos", {**datos, "motivo": "lista_de_reclamos_no_disponible"}
    if not reclamos:
        return "liberados", datos
    impugnables = contar_en_la_pagina(pagina, modulo)
    if impugnables is None:
        return "fallos", {**datos, "motivo": "lista_de_reclamos_no_disponible"}
    datos["reclamos"] = reclamos
    if not impugnables:
        return "en_proceso", datos
    reclamantes = leer_reclamantes(pagina, video_id)
    if not reclamantes:
        # Sin saber quién reclama no se sabe si está en la lista negra: no va a «Impugnar todos».
        return "fallos", {**datos, "motivo": "reclamantes_no_disponibles"}
    datos.update(impugnables=impugnables, reclamantes=reclamantes)
    en_lista_negra = reclamantes_en_lista_negra(reclamantes, lista_negra)
    if not en_lista_negra:
        return "sin_impugnar", datos
    return "lista_negra", {
        **datos,
        "reclamantes_en_lista_negra": en_lista_negra,
        "captura": capturar_pagina(pagina, DIR_CAPTURAS / f"reclamos_{video_id}.png"),
    }


def mapear_canal(uploader=None, abrir_studio=pagina_de_studio, modulo=None, leer_reclamantes=None):
    """Vuelve a mirar en Studio los videos aparcados por un reclamo. Solo lee: nunca pulsa nada.

    Los videos salen de `videos_aparcados`, primero los que hace más que no se miran. Cada uno
    va a una lista: `liberados` (/claims ya no muestra reclamos, en dos lecturas), `sin_impugnar`
    (algún "Tomar medidas" habilitado, con sus reclamantes), `lista_negra` (como el anterior,
    pero algún reclamante está en la lista negra), `en_proceso` (tiene reclamos y ninguno se
    puede impugnar) o `fallos` (no se pudo leer). Lo que no alcanzó a mirarse en
    TIEMPO_MAXIMO_MAPEO_S sale en `sin_revisar`. Además trae `privados_sin_programar`
    (`privados_sin_fecha`): los privados sin fecha de publicación de todo el canal que no están
    en pendientes.json; salen de la API y Studio no los abre. Lo lanza el servicio con
    /mapear_canal.
    """
    uploader = uploader or uploader_real()
    log = ruta_log("mapear_canal")
    base = {"log": str(log)}
    listas = {"liberados": [], "sin_impugnar": [], "lista_negra": [], "en_proceso": [], "fallos": []}
    with open(log, "w", encoding="utf-8") as salida, contextlib.redirect_stdout(salida):
        uploader.cargar_env()
        youtube, *_ = uploader.probar_credenciales_disponibles(prefix="upload")
        if not youtube:
            return {"ok": False, "motivo": "sin_credenciales", **base}
        ahora = datetime.now().astimezone()
        yt = {"cliente": youtube}
        try:
            videos = ejecutar_api(uploader, yt, lambda c: videos_aparcados(uploader, c, ahora))
        except SinCredenciales:
            return {"ok": False, "motivo": "sin_credenciales", **base}
        except HttpError as exc:
            print(f"Error YouTube API: {exc}")
            return {"ok": False, "motivo": "error_api", "detalle": str(exc)[:200], **base}
        except POSIBLES_ERRORES_DE_CONEXION as exc:
            if not es_error_de_conexion(exc):
                raise
            detalle = f"{type(exc).__name__}: {exc}"[:200]
            print(f"Sin conexion con YouTube tras {REINTENTOS_CONEXION} reintentos: {detalle}")
            return {"ok": False, "motivo": "sin_conexion", "detalle": detalle, **base}
        # Aparte de los aparcados: si recorrer todo el canal falla, los aparcados se revisan igual.
        privados, motivo_privados = [], None
        try:
            privados = privados_sin_fecha(uploader, yt)
        except SinCredenciales:
            motivo_privados = "sin_credenciales"
        except HttpError as exc:
            print(f"Error YouTube API al buscar los privados sin fecha: {exc}")
            motivo_privados = "error_api"
        except POSIBLES_ERRORES_DE_CONEXION as exc:
            if not es_error_de_conexion(exc):
                raise
            print(f"Sin conexion con YouTube al buscar los privados sin fecha: {type(exc).__name__}: {exc}")
            motivo_privados = "sin_conexion"
        mapeo = leer_mapeo()
        pendientes = sorted(videos, key=lambda video: mapeo.get(video["video_id"], ""))
        print(f"{len(videos)} videos aparcados por un reclamo")
        print(f"{len(privados)} videos privados sin fecha de publicación fuera de pendientes.json")
        resumen = {
            "aparcados": len(videos),
            "privados_sin_programar": privados,
            **({"motivo_privados": motivo_privados} if motivo_privados else {}),
            **listas,
            "sin_revisar": pendientes,
            **base,
        }
        leyendo = None  # el video que se está leyendo; sin él (el navegador ni abrió) nada se sella
        try:
            if pendientes:
                modulo = modulo or impugnador()
                leer_reclamantes = leer_reclamantes or apelacion_agente().extract_claimants_from_video
                lista_negra = cargar_lista_negra()
                # El mismo candado que `programar`, `impugnar` y `apelar`: comparten el perfil.
                with bloqueo("programar"), abrir_studio() as pagina:
                    DIR_CAPTURAS.mkdir(parents=True, exist_ok=True)
                    inicio = time.monotonic()
                    while pendientes:
                        if len(pendientes) < len(videos) and time.monotonic() - inicio > TIEMPO_MAXIMO_MAPEO_S:
                            print("Se acabó el tiempo de la corrida: el resto va primero la próxima vez.")
                            break
                        video = leyendo = pendientes[0]
                        lista, datos = mirar_video_aparcado(pagina, video, modulo, leer_reclamantes, lista_negra)
                        print(f"{video['video_id']}: {lista}")
                        listas[lista].append(datos)
                        pendientes.pop(0)
                        leyendo = None
                        mapeo[video["video_id"]] = datetime.now().isoformat(timespec="seconds")
                        guardar_mapeo(mapeo)
        except errores_del_paso_de_studio() as exc:
            detalle = f"{type(exc).__name__}: {exc}"[:200]
            print(f"No se pudieron leer los reclamos en Studio: {detalle}")
            if leyendo:
                # El video en que falló pasa al final: si falla siempre, no frena a los demás.
                mapeo[leyendo["video_id"]] = datetime.now().isoformat(timespec="seconds")
                guardar_mapeo(mapeo)
            motivo = exc.motivo if isinstance(exc, NavegadorNoDisponible) else "error_navegador"
            return {**resumen, "ok": False, "motivo": motivo, "detalle": detalle}
    return {**resumen, "ok": not listas["fallos"]}


# ---------------------------------------------------------------------------
# apelar
# ---------------------------------------------------------------------------

# Envíos por corrida (una por día): cientos de apelaciones seguidas pueden parecer spam.
LIMITE_APELACIONES = 10


def leer_apelaciones():
    if not APELACIONES_JSON.exists():
        return {}
    with open(APELACIONES_JSON, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


def guardar_apelaciones(registro):
    APELACIONES_JSON.parent.mkdir(parents=True, exist_ok=True)
    temporal = APELACIONES_JSON.with_name(f"{APELACIONES_JSON.name}.tmp")
    with open(temporal, "w", encoding="utf-8") as archivo:
        json.dump(registro, archivo, ensure_ascii=False, indent=2)
    temporal.replace(APELACIONES_JSON)


def apelar(sin_enviar=False, limite=LIMITE_APELACIONES, abrir_studio=pagina_de_studio):
    """Apela en hebreo la restricción de edad y los anuncios limitados de los videos del canal.

    Lo lanza el servicio una vez por día. Cada video queda en APELACIONES_JSON antes de
    pulsar el envío y nunca se le manda nada otra vez; lo enviado solo se vuelve a mirar para
    anotar qué resolvió YouTube. `sin_enviar` es la prueba en seco.
    """
    import studio_apelaciones

    log = ruta_log("apelar")
    base = {"log": str(log), "sin_enviar": sin_enviar}
    registro = leer_apelaciones()

    def anotar(clave, datos):
        registro[clave] = datos
        guardar_apelaciones(registro)

    def capturar(nombre):
        return capturar_pagina(pagina, DIR_CAPTURAS / f"{nombre}.png")

    with open(log, "w", encoding="utf-8") as salida, contextlib.redirect_stdout(salida):
        try:
            # El mismo candado que `programar` e `impugnar`: comparten la copia del perfil.
            with bloqueo("programar"), abrir_studio() as pagina:
                DIR_CAPTURAS.mkdir(parents=True, exist_ok=True)
                resumen = studio_apelaciones.apelar_videos(
                    pagina, registro, anotar, limite, sin_enviar=sin_enviar, capturar=capturar
                )
        except errores_del_paso_de_studio() as exc:
            detalle = f"{type(exc).__name__}: {exc}"[:200]
            print(f"No se pudo apelar en Studio: {detalle}")
            motivo = exc.motivo if isinstance(exc, NavegadorNoDisponible) else "error_navegador"
            return {"ok": False, "motivo": motivo, "detalle": detalle, "videos": [], "fallos": [], **base}
    return {"ok": not resumen.get("motivo") and not resumen["fallos"], **resumen, **base}


# ---------------------------------------------------------------------------
# eliminar_video
# ---------------------------------------------------------------------------


def eliminar_video(video_id, retirado=False, uploader=None):
    """Borra el video de YouTube. No se puede deshacer: solo lo lanza un botón del usuario.

    Antes de borrar consulta el video: confirma que existe en el canal y devuelve su
    título, que es lo que se informa como borrado. Con `retirado` (YouTube lo retiró por ser
    copia de otro) no lo consulta: la API no lista ese video ni a su dueño, y la consulta
    respondería `no_existe` sin borrarlo. Si YouTube no lo deja borrar, lo dice el error.
    """
    uploader = uploader or uploader_real()
    log = ruta_log("eliminar_video")
    with open(log, "w", encoding="utf-8") as salida, contextlib.redirect_stdout(salida):
        uploader.cargar_env()
        youtube, *_ = uploader.probar_credenciales_disponibles(prefix="upload")
        if not youtube:
            return {"ok": False, "motivo": "sin_credenciales", "video_id": video_id, "log": str(log)}
        yt = {"cliente": youtube}
        try:
            titulo = video_id
            if not retirado:
                respuesta = ejecutar_api(
                    uploader, yt, lambda c: c.videos().list(part="snippet", id=video_id).execute()
                )
                items = respuesta.get("items", [])
                if not items:
                    return {"ok": False, "motivo": "no_existe", "video_id": video_id, "log": str(log)}
                titulo = items[0].get("snippet", {}).get("title") or video_id
            ejecutar_api(uploader, yt, lambda c: c.videos().delete(id=video_id).execute())
        except SinCredenciales:
            return {"ok": False, "motivo": "sin_credenciales", "video_id": video_id, "log": str(log)}
        except HttpError as exc:
            print(f"Error YouTube API: {exc}")
            if getattr(exc.resp, "status", None) == 404:
                # Puede ser un video que ya no existía: el servicio no debe decir que sigue en YouTube.
                return {"ok": False, "motivo": "no_encontrado", "video_id": video_id, "log": str(log)}
            return {
                "ok": False,
                "motivo": "error_api",
                "video_id": video_id,
                "detalle": str(exc)[:200],
                "log": str(log),
            }
        except POSIBLES_ERRORES_DE_CONEXION as exc:
            if not es_error_de_conexion(exc):
                raise
            detalle = f"{type(exc).__name__}: {exc}"[:200]
            print(f"Sin conexion con YouTube tras {REINTENTOS_CONEXION} reintentos: {detalle}")
            return {
                "ok": False,
                "motivo": "sin_conexion",
                "video_id": video_id,
                "detalle": detalle,
                "log": str(log),
            }
    return {"ok": True, "video_id": video_id, "titulo": titulo, "log": str(log)}


# ---------------------------------------------------------------------------
# sembrar_publicados
# ---------------------------------------------------------------------------


def subidas_del_canal(uploader, yt):
    """(título, video_id) de todas las subidas del canal: 50 por llamada, 1 unidad de cuota cada una."""
    lista = ejecutar_api(uploader, yt, uploader.get_uploads_playlist_id)
    if not lista:
        return []
    subidas, pagina = [], None
    while True:
        respuesta = ejecutar_api(
            uploader,
            yt,
            lambda c: c.playlistItems()
            .list(part="snippet", playlistId=lista, maxResults=50, pageToken=pagina)
            .execute(),
        )
        for item in respuesta.get("items", []):
            snippet = item.get("snippet", {})
            video_id = snippet.get("resourceId", {}).get("videoId")
            if video_id:
                subidas.append((snippet.get("title", ""), video_id))
        pagina = respuesta.get("nextPageToken")
        if not pagina:
            return subidas


def resultado_final(salida):
    """El JSON final de un `servicio_*.out`, o None."""
    lineas = salida.read_text(encoding="utf-8", errors="replace").splitlines()
    for linea in reversed(lineas):
        if not linea.startswith("{"):
            continue
        try:
            return json.loads(linea)
        except json.JSONDecodeError:
            continue
    return None


def eliminados_por_el_servicio():
    """Los álbumes que el servicio eliminó y que ↩️ Deshacer no devolvió.

    Según el JSON final de cada `servicio_eliminar_*.out` y `servicio_restaurar_*.out`: cuenta lo
    último que le pasó a cada álbum. El nombre del log lleva la fecha, así que ordena por tiempo.
    """
    eventos = []
    for paso in ("eliminar", "restaurar"):
        # El dígito deja afuera los servicio_eliminar_video_*.out, que borran videos de YouTube.
        for salida in DIR_LOGS.glob(f"servicio_{paso}_[0-9]*.out"):
            resultado = resultado_final(salida) or {}
            if resultado.get("ok") and resultado.get("album"):
                eventos.append((salida.name.removeprefix(f"servicio_{paso}_"), paso, resultado["album"]))
    ultimo = {}
    for _, paso, album in sorted(eventos):
        ultimo[album] = paso
    return [album for album, paso in ultimo.items() if paso == "eliminar"]


def subidos_por_el_servicio():
    try:
        estado = json.loads((DIR_AGENTE / "servicio.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    return list(estado.get("subidos", []))


def sembrar_publicados(uploader=None):
    """Carga en el registro los títulos del canal y lo que el servicio ya eliminó o subió.

    Se puede repetir: saltea los video_id y los pares (origen, nombre) que ya están.
    """
    uploader = uploader or uploader_real()
    log = ruta_log("sembrar_publicados")
    with open(log, "w", encoding="utf-8") as salida, contextlib.redirect_stdout(salida):
        uploader.cargar_env()
        youtube, *_ = uploader.probar_credenciales_disponibles(prefix="upload")
        if not youtube:
            return {"ok": False, "motivo": "sin_credenciales", "log": str(log)}
        try:
            canal = subidas_del_canal(uploader, {"cliente": youtube})
        except SinCredenciales:
            return {"ok": False, "motivo": "sin_credenciales", "log": str(log)}
        except HttpError as exc:
            print(f"Error YouTube API: {exc}")
            return {"ok": False, "motivo": "error_api", "detalle": str(exc)[:200], "log": str(log)}
        except POSIBLES_ERRORES_DE_CONEXION as exc:
            if not es_error_de_conexion(exc):
                raise
            detalle = f"{type(exc).__name__}: {exc}"[:200]
            print(f"Sin conexion con YouTube tras {REINTENTOS_CONEXION} reintentos: {detalle}")
            return {"ok": False, "motivo": "sin_conexion", "detalle": detalle, "log": str(log)}

    eliminados = eliminados_por_el_servicio()
    subidos = subidos_por_el_servicio()
    filas = (
        [(titulo, "canal", video_id) for titulo, video_id in canal]
        + [(nombre, "eliminar", "") for nombre in eliminados]
        + [(nombre, "subir", "") for nombre in subidos]
    )
    with bloqueo("publicados"):
        registro = registro_publicados.leer(REGISTRO_PUBLICADOS)
        video_ids = registro.video_ids()
        vistos = {(f["origen"], f["nombre"]) for f in registro.filas}
        nuevas = []
        for nombre, origen, video_id in filas:
            if (video_id and video_id in video_ids) or (origen, nombre) in vistos:
                continue
            if video_id:
                video_ids.add(video_id)
            vistos.add((origen, nombre))
            nuevas.append((nombre, origen, video_id))
        agregados = registro_publicados.anotar(REGISTRO_PUBLICADOS, nuevas)
    return {
        "ok": True,
        "agregados": agregados,
        "leidos": {"canal": len(canal), "eliminar": len(eliminados), "subir": len(subidos)},
        "registro": str(REGISTRO_PUBLICADOS),
        "log": str(log),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Pasos del pipeline para el agente: renderizar, verificar, subir y programar."
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("renderizar", help="Renderiza el próximo álbum (o uno puntual).")
    p.add_argument("--carpeta", help="Nombre de la carpeta en 01_limpieza_de_impurezas, Album/ o EP/.")

    p = sub.add_parser(
        "buscar", help="Elige el próximo álbum y captura su búsqueda en YouTube, sin renderizarlo."
    )
    p.add_argument("--aleatorio", action="store_true", help="Elige al azar entre las carpetas de origen.")
    p.add_argument(
        "--excluir", action="append", default=[], help="Álbum que se salta (repetible): la lista del servicio."
    )

    p = sub.add_parser("verificar", help="Tests del video renderizado y captura.")
    p.add_argument("--carpeta", required=True, help="Carpeta devuelta por renderizar.")

    p = sub.add_parser("subir", help="Sube como privado y borra la carpeta del álbum.")
    p.add_argument("--carpeta", required=True, help="Carpeta devuelta por renderizar.")
    p.add_argument("--sin-deathgrind", action="store_true", help="No usar la API de DeathGrind.")

    p = sub.add_parser("eliminar", help="Aparta el álbum en la papelera sin subirlo (ya publicado).")
    p.add_argument("--carpeta", required=True, help="Carpeta devuelta por buscar o por renderizar.")

    p = sub.add_parser(
        "a_censura", help="Aparta el álbum en Censura/ para censurar su portada (botón «🔞 Censurar»)."
    )
    p.add_argument("--carpeta", required=True, help="Carpeta devuelta por buscar.")

    p = sub.add_parser(
        "censurar", help="Busca con el modelo local qué tapar de la portada y deja una candidata."
    )
    p.add_argument("--carpeta", required=True, help="Carpeta del álbum en Censura/.")
    p.add_argument("--intento", type=int, default=1, help="1: imagen entera; 2: 2x2 pedazos; 3 o más: 3x3 y la entera.")

    p = sub.add_parser(
        "aceptar_censura", help="Pone la portada censurada en el álbum y lo devuelve a su origen."
    )
    p.add_argument("--carpeta", required=True, help="Carpeta del álbum en Censura/.")

    p = sub.add_parser("restaurar", help="Devuelve a su carpeta un álbum que eliminar apartó.")
    p.add_argument("--papelera", required=True, help="Entrada de la papelera devuelta por eliminar.")

    sub.add_parser("vaciar_papelera", help="Borra lo que lleva más de 24 h en la papelera.")

    p = sub.add_parser("programar", help="Revisa reclamos y programa la fecha de publicación.")
    p.add_argument("--video-id", required=True)
    p.add_argument(
        "--esperar",
        action="store_true",
        help="Espera a que YouTube procese el video y pase el margen de 1 h.",
    )
    p.add_argument(
        "--reclamo",
        choices=("si", "no"),
        help="Saltea la revisión en Studio (cuando el usuario ya respondió).",
    )

    p = sub.add_parser(
        "eliminar_video", help="Borra un video de YouTube (no se puede deshacer)."
    )
    p.add_argument("--video-id", required=True)
    p.add_argument(
        "--retirado",
        action="store_true",
        help="El video lo retiró YouTube por ser copia de otro: la API no lo lista, así que se "
        "borra sin comprobar antes que existe.",
    )

    p = sub.add_parser(
        "impugnar",
        help="Impugna en Studio los reclamos de un video (declaración legal a nombre del usuario).",
    )
    p.add_argument("--video-id", required=True)
    p.add_argument(
        "--sin-enviar",
        action="store_true",
        help="Llena el formulario del primer reclamo y lo cierra sin enviarlo (prueba en seco).",
    )

    p = sub.add_parser(
        "impugnar_con_ventana",
        help="Impugna en una ventana visible los reclamos de uno o varios videos y espera a que "
        "el usuario pase ahí «Verifica que eres tú» (declaración legal a nombre del usuario).",
    )
    p.add_argument("--video-id", action="append", required=True, help="Se repite, uno por video.")
    p.add_argument(
        "--sin-enviar",
        action="store_true",
        help="Llena el formulario del primer reclamo de cada video y lo cierra sin enviarlo (prueba en seco).",
    )

    p = sub.add_parser(
        "apelar",
        help="Apela en hebreo la restricción de edad y los anuncios limitados (a nombre del usuario).",
    )
    p.add_argument(
        "--sin-enviar",
        action="store_true",
        help="Llena el primer formulario de cada tipo y lo cierra sin enviarlo (prueba en seco).",
    )
    p.add_argument("--limite", type=int, default=LIMITE_APELACIONES, help="Envíos como máximo.")

    sub.add_parser(
        "sembrar_publicados",
        help="Carga en el registro de publicados los títulos del canal y lo eliminado o subido.",
    )

    sub.add_parser(
        "seguir_impugnaciones",
        help="Lee en Studio el estado de los reclamos impugnados y dice qué cambió (solo lectura).",
    )

    sub.add_parser(
        "mapear_canal",
        help=(
            "Vuelve a mirar en Studio los videos aparcados por un reclamo y busca los privados "
            "sin fecha de publicación (solo lectura)."
        ),
    )

    p = sub.add_parser(
        "estado", help="Videos sin programar, álbumes sin aprobar y si hay un render corriendo."
    )
    p.add_argument(
        "--esperar-render",
        action="store_true",
        help="Espera a que termine el render en curso (lanzado por otra sesión).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.chdir(PROJECT_ROOT)
    if args.comando == "renderizar":
        return responder(renderizar(args.carpeta))
    if args.comando == "buscar":
        return responder(buscar(aleatorio=args.aleatorio, excluir=args.excluir))
    if args.comando == "verificar":
        return responder(verificar(args.carpeta))
    if args.comando == "subir":
        return responder(subir(args.carpeta, sin_deathgrind=args.sin_deathgrind))
    if args.comando == "eliminar":
        return responder(eliminar(args.carpeta))
    if args.comando == "a_censura":
        return responder(a_censura(args.carpeta))
    if args.comando == "censurar":
        return responder(censurar(args.carpeta, args.intento))
    if args.comando == "aceptar_censura":
        return responder(aceptar_censura(args.carpeta))
    if args.comando == "restaurar":
        return responder(restaurar(args.papelera))
    if args.comando == "vaciar_papelera":
        return responder(vaciar_papelera())
    if args.comando == "programar":
        reclamo = None if args.reclamo is None else args.reclamo == "si"
        return responder(programar(args.video_id, esperar=args.esperar, reclamo=reclamo))
    if args.comando == "eliminar_video":
        return responder(eliminar_video(args.video_id, retirado=args.retirado))
    if args.comando == "impugnar":
        return responder(impugnar(args.video_id, sin_enviar=args.sin_enviar))
    if args.comando == "impugnar_con_ventana":
        return responder(impugnar_con_ventana(args.video_id, sin_enviar=args.sin_enviar))
    if args.comando == "apelar":
        return responder(apelar(sin_enviar=args.sin_enviar, limite=args.limite))
    if args.comando == "sembrar_publicados":
        return responder(sembrar_publicados())
    if args.comando == "seguir_impugnaciones":
        return responder(seguir_impugnaciones())
    if args.comando == "mapear_canal":
        return responder(mapear_canal())
    return responder(estado(esperar_render=args.esperar_render))


if __name__ == "__main__":
    sys.exit(main())
