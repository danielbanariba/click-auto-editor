#!/usr/bin/env python3
"""Pasos del pipeline que ejecuta servicio_publicar.py, de a un álbum.

Cada subcomando termina imprimiendo UNA línea JSON para que el servicio
decida el paso siguiente; el resto de la salida va a data/agente/logs/.

    renderizar   renderiza el próximo álbum y devuelve su carpeta
    verificar    corre los tests del video y genera la captura
    subir        sube el video como privado y borra la carpeta del álbum
    eliminar     borra la carpeta del álbum sin subirlo (ya estaba publicado)
    programar    espera el procesado, revisa reclamos y programa la fecha
    estado       videos sin programar, álbumes sin aprobar y render en curso
"""
import argparse
import contextlib
import fcntl
import functools
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

from config import (
    AUDIO_FORMATS,
    BASE_DIR,
    DIR_AUDIO_SCRIPTS,
    DIR_ERROR404,
    DIR_RENDERIZANDO,
    DIR_UPLOAD,
    INTRO_DURATION,
    MIN_FREE_SPACE_GB,
    PROJECT_ROOT,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)

DIR_AGENTE = PROJECT_ROOT / "data" / "agente"
PENDIENTES_JSON = DIR_AGENTE / "pendientes.json"
PROGRAMADOS_JSONL = DIR_AGENTE / "programados.jsonl"
DIR_CAPTURAS = DIR_AGENTE / "capturas"
DIR_LOGS = DIR_AGENTE / "logs"
RAICES_BORRABLES = (DIR_RENDERIZANDO, DIR_UPLOAD)

TOLERANCIA_DURACION_S = 3.0
VOLUMEN_MEDIO_MINIMO_DB = -45.0
SILENCIO_LARGO_S = 20
ANTICIPACION_MIN = 15
MARGEN_RECLAMO_MIN = 60
ESPERA_ENTRE_CONSULTAS_S = 300
PROCESO_RENDER = "ffmpeg_render.py"
ESPERA_RENDER_MAX_S = 6 * 3600
ESCANEO_PROGRAMADOS = int(os.environ.get("AGENTE_ESCANEO_PROGRAMADOS", "500"))
CAMPOS_STATUS_CONSERVADOS = ("embeddable", "license", "publicStatsViewable")

P_SIN_PROBLEMAS = re.compile(
    r"(no se encontraron problemas|no se encontr[oó] ning[uú]n problema|"
    r"no hay problemas de derechos de autor|no copyright issues|no issues found)",
    re.IGNORECASE,
)


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
    if not directorio.exists():
        return []
    return sorted(p.name for p in directorio.iterdir() if p.is_dir())


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


def elegir_fecha(ahora, con_reclamo, ocupadas):
    if con_reclamo:
        desde = sumar_dos_anios(ahora)
    else:
        desde = ahora + timedelta(minutes=ANTICIPACION_MIN)
    return siguiente_hora_libre(desde, ocupadas)


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


def clasificar_reclamo(botones_detalle, texto_pagina, en_pagina_copyright):
    if not en_pagina_copyright:
        return "desconocido"
    if botones_detalle > 0:
        return "con_reclamo"
    if P_SIN_PROBLEMAS.search(texto_pagina or ""):
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


def render_en_curso():
    """True si corre un render, lo haya lanzado esta sesión, otra o el usuario a mano."""
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argumentos = cmdline.read_bytes().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        # Solo Python cuenta: un editor con el script abierto también lo tiene en sus argumentos.
        if Path(argumentos[0]).name.startswith("python") and any(
            a.endswith(PROCESO_RENDER) for a in argumentos[1:]
        ):
            return True
    return False


def renderizar(carpeta=None):
    # Dos renders 4K a la vez llenan el disco y la GPU.
    if render_en_curso():
        return {"ok": False, "motivo": "render_en_curso"}
    try:
        libre_gb = shutil.disk_usage(BASE_DIR).free / 1024**3
    except OSError as exc:
        return {"ok": False, "motivo": "disco_no_disponible", "detalle": str(exc)}
    if libre_gb < MIN_FREE_SPACE_GB:
        return {"ok": False, "motivo": "sin_espacio", "libre_gb": round(libre_gb, 1)}

    origen_antes = listar_carpetas(DIR_AUDIO_SCRIPTS)
    if not origen_antes:
        return {"ok": False, "motivo": "sin_albumes"}
    if carpeta and carpeta not in origen_antes:
        return {"ok": False, "motivo": "album_no_encontrado", "album": carpeta}

    renderizando_antes = set(listar_carpetas(DIR_RENDERIZANDO))
    error_antes = set(listar_carpetas(DIR_ERROR404))
    comando = [sys.executable, "2. ffmpeg_render.py", "--test"]
    if carpeta:
        comando.append(str(DIR_AUDIO_SCRIPTS / carpeta))

    log = ruta_log("render")
    with open(log, "w", encoding="utf-8") as salida:
        # El render sale con 0 aunque falle: el resultado se lee de las carpetas.
        subprocess.run(
            comando, cwd=PROJECT_ROOT, stdout=salida, stderr=subprocess.STDOUT, check=False
        )

    origen_despues = listar_carpetas(DIR_AUDIO_SCRIPTS)
    nuevas = [c for c in listar_carpetas(DIR_RENDERIZANDO) if c not in renderizando_antes]
    restantes = len(origen_despues)
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
                "log": str(log),
            }

    album = carpeta or next(iter(set(origen_antes) - set(origen_despues)), None)
    con_error = [c for c in listar_carpetas(DIR_ERROR404) if c not in error_antes]
    return {
        "ok": False,
        "motivo": "render_fallido",
        "album": album,
        "movido_a_error404": bool(con_error),
        "restantes": restantes,
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
                pagina = navegador.new_page(viewport={"width": 1280, "height": 1000}, locale="es-419")
                pagina.goto(
                    f"https://www.youtube.com/results?search_query={quote(titulo)}",
                    wait_until="domcontentloaded",
                    timeout=60000,
                )
                try:
                    pagina.wait_for_selector("ytd-video-renderer", timeout=20000)
                except PlaywrightError:
                    pass  # Sin resultados también sirve: se captura lo que haya.
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
    for captura in (
        DIR_CAPTURAS / f"{slug(carpeta.name)}.jpg",
        DIR_CAPTURAS / f"busqueda_{slug(carpeta.name)}.png",
    ):
        captura.unlink(missing_ok=True)
    return bool(borrada)


def eliminar(carpeta, uploader=None, raices=RAICES_BORRABLES):
    """Borra un álbum renderizado que ya estaba publicado, sin subirlo."""
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        return {"ok": False, "motivo": "carpeta_no_existe", "carpeta": str(carpeta)}
    raiz = raiz_borrable(carpeta, raices)
    if raiz is None:
        return {"ok": False, "motivo": "carpeta_fuera_de_raiz", "carpeta": str(carpeta)}
    uploader = uploader or uploader_real()
    log = ruta_log("eliminar")
    with open(log, "w", encoding="utf-8") as salida, contextlib.redirect_stdout(salida):
        borrada = borrar_album(uploader, carpeta, raiz)
    if not borrada:
        return {"ok": False, "motivo": "no_se_pudo_borrar", "album": carpeta.name, "log": str(log)}
    return {"ok": True, "album": carpeta.name, "log": str(log)}


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
    with open(log, "a", encoding="utf-8") as salida, contextlib.redirect_stdout(salida):
        borrada = borrar_album(uploader, carpeta, raiz)
    return {
        "ok": True,
        "video_id": video_id,
        "titulo": metadata["title"],
        "carpeta_borrada": borrada,
        "studio": f"https://studio.youtube.com/video/{video_id}/edit",
        "log": str(log),
    }


# ---------------------------------------------------------------------------
# programar
# ---------------------------------------------------------------------------


def ejecutar_api(uploader, yt, llamada):
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


def revisar_reclamo_en_studio(video_id):
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    apelacion = cargar_script("11. apelacion.py", "apelacion")
    # Copia propia del perfil para no pisar la de 11. apelacion.py si corre a la vez.
    apelacion.DEFAULT_RUNTIME_PROFILE_DIR = (
        PROJECT_ROOT / ".runtime_browser_profiles" / "agente"
    )
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
        return {"estado": "desconocido", "motivo": "perfil_de_navegador"}
    if user_data_dir is None:
        return {"estado": "desconocido", "motivo": "sin_perfil_de_navegador"}
    channel, ejecutable = apelacion.resolve_launch_config(args, fuente)
    headless = os.environ.get("AGENTE_STUDIO_HEADLESS", "1") != "0"
    captura = DIR_CAPTURAS / f"studio_{video_id}.png"
    captura.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        try:
            contexto = apelacion.launch_context(
                playwright,
                user_data_dir,
                headless,
                channel,
                apelacion.profile_name_from_args(args),
                ejecutable,
            )
        except PlaywrightError as exc:
            print(f"No se pudo abrir el navegador: {exc}")
            return {"estado": "desconocido", "motivo": "error_navegador"}
        try:
            pagina = contexto.pages[0] if contexto.pages else contexto.new_page()
            reclamantes = apelacion.extract_claimants_from_video(pagina, video_id)
            en_copyright = f"/video/{video_id}/copyright" in pagina.url
            botones = pagina.get_by_role("button", name=apelacion.P_VER_DETALLES_BTN).count()
            texto = pagina.evaluate("() => (document.body && document.body.innerText) || ''")
            pagina.screenshot(path=str(captura))
        except PlaywrightError as exc:
            print(f"Error revisando Studio: {exc}")
            return {"estado": "desconocido", "motivo": "error_navegador"}
        finally:
            contexto.close()

    return {
        "estado": clasificar_reclamo(botones, texto, en_copyright),
        "reclamantes": reclamantes,
        "captura_studio": str(captura),
    }


def horas_tomadas(uploader, yt):
    programados, _ = ejecutar_api(
        uploader,
        yt,
        lambda c: uploader.fetch_scheduled_publish_times(c, max_items=ESCANEO_PROGRAMADOS),
    )
    return horas_ocupadas(programados) | horas_ocupadas(fechas_programadas_localmente())


def armar_resumen_programado(titulo, publica_en, con_reclamo, reclamantes):
    fecha = publica_en.strftime("%Y-%m-%d %H:%M")
    if con_reclamo:
        quienes = f" ({', '.join(reclamantes)})" if reclamantes else ""
        return f"⚠️ {titulo}\nTiene reclamo{quienes}: programado para {fecha} para apelar."
    return f"✓ {titulo}\nSin reclamos: programado para {fecha}."


def programar(video_id, esperar=False, reclamo=None, revisar=revisar_reclamo_en_studio, uploader=None):
    uploader = uploader or uploader_real()
    titulo = (buscar_pendiente(video_id) or {}).get("titulo", video_id)
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
                else:
                    con_reclamo = reclamo

                publica_en = elegir_fecha(
                    datetime.now().astimezone(), con_reclamo, horas_tomadas(uploader, yt)
                )
                cuerpo = {"id": video_id, "status": armar_status_programado(status, publica_en)}
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

    return {
        "ok": True,
        "video_id": video_id,
        "titulo": titulo,
        "con_reclamo": con_reclamo,
        "reclamantes": reclamantes,
        "publica_en": publica_en.isoformat(),
        "resumen": armar_resumen_programado(titulo, publica_en, con_reclamo, reclamantes),
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
    p.add_argument("--carpeta", help="Nombre de la carpeta en 01_limpieza_de_impurezas.")

    p = sub.add_parser("verificar", help="Tests del video renderizado y captura.")
    p.add_argument("--carpeta", required=True, help="Carpeta devuelta por renderizar.")

    p = sub.add_parser("subir", help="Sube como privado y borra la carpeta del álbum.")
    p.add_argument("--carpeta", required=True, help="Carpeta devuelta por renderizar.")
    p.add_argument("--sin-deathgrind", action="store_true", help="No usar la API de DeathGrind.")

    p = sub.add_parser("eliminar", help="Borra la carpeta del álbum sin subirlo (ya publicado).")
    p.add_argument("--carpeta", required=True, help="Carpeta devuelta por renderizar.")

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
    if args.comando == "verificar":
        return responder(verificar(args.carpeta))
    if args.comando == "subir":
        return responder(subir(args.carpeta, sin_deathgrind=args.sin_deathgrind))
    if args.comando == "eliminar":
        return responder(eliminar(args.carpeta))
    if args.comando == "programar":
        reclamo = None if args.reclamo is None else args.reclamo == "si"
        return responder(programar(args.video_id, esperar=args.esperar, reclamo=reclamo))
    return responder(estado(esperar_render=args.esperar_render))


if __name__ == "__main__":
    sys.exit(main())
