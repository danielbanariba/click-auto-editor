import os
import random
import subprocess
import csv
import ast
import json
import uuid
import requests
import numpy as np
import io
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import time
import sys
import re
import shutil
from mutagen import File as MutagenFile
from effects.sombra import add_shadow

# Importar configuración
sys.path.append(str(Path(__file__).parent))
from config import (
    DIR_AUDIO_SCRIPTS,
    DIR_UPLOAD,
    INTRO_VIDEO,
    MAX_PARALLEL_RENDERS,
    MAX_FOLDERS_TO_PROCESS,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    FPS,
    INTRO_DURATION,
    USE_GPU,
    VIDEO_PRESET_NVENC,
    VIDEO_CQ,
    VIDEO_BITRATE,
    VIDEO_MAXRATE,
    VIDEO_BUFSIZE,
    VIDEO_PRESET_CPU,
    VIDEO_CRF,
    AUDIO_BITRATE,
    AUDIO_SAMPLE_RATE,
    NVENC_EXTRA_OPTS,
    SSD_TEMP_DIR,
    USE_SSD_TEMP,
    DEFAULT_BASE_DIR,
    STAGING_ENABLED,
    STAGING_FAST_BASE_DIR,
    STAGING_BATCH_SIZE,
    STAGING_SHUFFLE,
    USE_CPP_VHS,
    VHS_CPP_BIN,
    VHS_CPP_INTENSITY,
    VHS_CPP_OVERLAY,
    ALLOW_FFMPEG_FALLBACK,
    DISABLE_CPP_ON_CUDA,
    CUDA_FAIL_FAST
)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Rutas principales
MAIN_DIR = DIR_AUDIO_SCRIPTS
OUTPUT_DIR = DIR_AUDIO_SCRIPTS

# Sombra de portada
SHADOW_OPACITY = 255
SHADOW_DIRECTION = 68
SHADOW_DISTANCE = 27
SHADOW_SOFTNESS = 19

# Tracklist overlay
TRACKLIST_MIN_FONT_SIZE = 28
TRACKLIST_MAX_FONT_SIZE = 90
TRACKLIST_TOP_MARGIN_RATIO = 0.06
TRACKLIST_SIDE_MARGIN_RATIO = 0.06
TRACKLIST_RIGHT_MARGIN_RATIO = 0.05
TRACKLIST_GAP_RATIO = 0.03
TRACKLIST_LINE_SPACING = 0.12
TRACKLIST_MIN_LIST_WIDTH_RATIO = 0.25
TRACKLIST_COVER_SCALES = (0.85, 0.8, 0.75, 0.7)
TRACKLIST_MAX_LINES_PER_TRACK = 2
TRACKLIST_PAGE_SIZE = 12
TRACKLIST_TEXT_SHADOW_OFFSET = 4
TRACKLIST_TEXT_SHADOW_BLUR = 5
TRACKLIST_TEXT_SHADOW_ALPHA = 200
TRACKLIST_TEXT_SHADOW_ALPHA_HIGHLIGHT = 230

CUDA_ERROR_FLAG = Path(__file__).parent / ".cuda_cpp_disabled"
ENV_LOADED = False
DEATHGRIND_SESSION = None

BASE_URL = "https://deathgrind.club"
API_URL = f"{BASE_URL}/api"
DELAY_BASE_429 = 30
MAX_RETRIES_429 = int(os.environ.get("DEATHGRIND_MAX_429", "1"))


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

GPU_CHECKED = False
GPU_AVAILABLE = False


def resolve_rel_path(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return Path(path.name)


def list_folders(path: Path):
    if not path.exists():
        return []
    return [p for p in path.iterdir() if p.is_dir()]


def stage_folders_to_fast(slow_audio: Path, fast_audio: Path, batch_size: int, shuffle: bool):
    """
    Mueve un lote de carpetas desde HDD a NVMe para render.
    """
    folders = list_folders(slow_audio)
    if not folders:
        return []

    if shuffle:
        random.shuffle(folders)
    else:
        folders.sort(key=lambda p: p.name.lower())

    batch_size = max(1, batch_size)
    batch = folders[:batch_size]

    fast_audio.mkdir(parents=True, exist_ok=True)

    staged = []
    for folder in batch:
        dest = fast_audio / folder.name
        if dest.exists():
            continue
        try:
            shutil.move(str(folder), str(dest))
            staged.append(dest)
        except Exception as exc:
            print(f"[STAGING] Error moviendo {folder.name}: {exc}")
    return staged


def restore_staged_folder(fast_audio: Path, slow_audio: Path, folder_name: str):
    src = fast_audio / folder_name
    if not src.exists():
        return False
    destination = get_unique_destination(slow_audio, folder_name)
    try:
        shutil.move(str(src), str(destination))
        return True
    except Exception as exc:
        print(f"[STAGING] Error devolviendo {folder_name}: {exc}")
        return False


def prepare_staging_batch():
    """
    Prepara el staging en NVMe y retorna (folders, staging_ctx) o (None, None) si no aplica.
    """
    if not STAGING_ENABLED:
        return None, None

    fast_base = Path(STAGING_FAST_BASE_DIR)
    if not fast_base.exists():
        try:
            fast_base.mkdir(parents=True, exist_ok=True)
            print(f"[STAGING] Carpeta NVMe creada: {fast_base}")
        except PermissionError:
            print(f"[STAGING] Sin permisos para crear {fast_base}. Se omite staging.")
            return None, None
        except Exception as exc:
            print(f"[STAGING] Error creando {fast_base}: {exc}. Se omite staging.")
            return None, None
    elif not fast_base.is_dir():
        print(f"[STAGING] La ruta NVMe no es carpeta: {fast_base}. Se omite staging.")
        return None, None

    audio_rel = resolve_rel_path(DIR_AUDIO_SCRIPTS, DEFAULT_BASE_DIR)
    upload_rel = resolve_rel_path(DIR_UPLOAD, DEFAULT_BASE_DIR)

    slow_audio = DEFAULT_BASE_DIR / audio_rel
    fast_audio = fast_base / audio_rel
    fast_upload = fast_base / upload_rel

    if list_folders(fast_audio) or list_folders(fast_upload):
        print("[STAGING] NVMe no está vacío. Limpia o mueve uploads antes de continuar.")
        return None, None

    batch_size = min(STAGING_BATCH_SIZE, MAX_FOLDERS_TO_PROCESS)
    staged = stage_folders_to_fast(slow_audio, fast_audio, batch_size, STAGING_SHUFFLE)
    if not staged:
        print("[STAGING] No se movieron carpetas al NVMe.")
        return None, None

    folders = [(path, path.name) for path in staged]
    staging_ctx = {
        "fast_audio": fast_audio,
        "slow_audio": slow_audio,
        "staged_names": {path.name for path in staged},
    }
    return folders, staging_ctx


def nvenc_available() -> bool:
    """
    Verifica si NVENC puede inicializarse en este entorno.
    """
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x360:rate=30",
            "-frames:v",
            "1",
            "-vf",
            "format=yuv420p",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace"
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def resolve_gpu_usage() -> bool:
    """
    Decide si se usará GPU realmente (NVENC disponible + config habilitada).
    """
    global GPU_CHECKED, GPU_AVAILABLE, USE_GPU
    if GPU_CHECKED:
        return USE_GPU and GPU_AVAILABLE

    GPU_CHECKED = True
    if not USE_GPU:
        GPU_AVAILABLE = False
        return False

    GPU_AVAILABLE = nvenc_available()
    if not GPU_AVAILABLE:
        print("[GPU] NVENC no disponible en este entorno. Usando CPU (libx264).")
    return USE_GPU and GPU_AVAILABLE


def get_unique_destination(base_dir: Path, folder_name: str) -> Path:
    """
    Devuelve un destino único para evitar sobreescrituras.
    """
    candidate = base_dir / folder_name
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = base_dir / f"{folder_name}__{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def get_unique_file_path(base_dir: Path, filename: str) -> Path:
    """
    Devuelve un nombre de archivo único dentro del directorio.
    """
    candidate = base_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        candidate = base_dir / f"{stem}__{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_folder_to_upload(source_folder: Path, folder_name: str, show_progress: bool) -> Path:
    """
    Mueve la carpeta renderizada a la carpeta de subida.
    """
    DIR_UPLOAD.mkdir(parents=True, exist_ok=True)
    overlays_dir = source_folder / "_track_overlays"
    if overlays_dir.exists():
        try:
            shutil.rmtree(overlays_dir)
        except Exception:
            pass
    destination = get_unique_destination(DIR_UPLOAD, folder_name)

    if show_progress:
        print(f"[UPLOAD] Moviendo carpeta a {destination}...")

    shutil.move(str(source_folder), str(destination))
    return destination


def get_complementary_color(r, g, b):
    """
    Calcula el color complementario de un RGB dado
    (mismo algoritmo que en auto_effects.py)
    """
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn

    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/diff) + 120) % 360
    else:
        h = (60 * ((r-g)/diff) + 240) % 360

    s = 0 if mx == 0 else (diff/mx)
    v = mx

    # Calcular el color complementario (rotar 180°)
    h = (h + 180) % 360

    # Convertir HSV de vuelta a RGB
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r = int((r + m) * 255)
    g = int((g + m) * 255)
    b = int((b + m) * 255)

    return r, g, b


def extract_average_color(image_path):
    """
    Extrae el color promedio de una imagen
    """
    try:
        img = Image.open(image_path)
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Calcular color promedio
        data = np.array(img)
        avg_color = data.mean(axis=(0, 1))

        return tuple(avg_color.astype(int))
    except Exception as e:
        print(f"Error extrayendo color de {image_path}: {e}")
        return (255, 255, 255)  # Blanco por defecto


def is_valid_image(image_path: Path) -> bool:
    """
    Verifica si un archivo de imagen puede abrirse sin errores.
    """
    if not image_path:
        return False
    try:
        if not Path(image_path).exists():
            return False
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def find_base_cover_for_shadow(shadow_path: Path):
    """
    Si la portada es *_shadow.png, intenta encontrar la base sin sombra.
    """
    stem = shadow_path.stem
    if not stem.endswith("_shadow"):
        return None

    base_stem = stem[:-7]
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = shadow_path.with_name(base_stem + ext)
        if candidate.exists():
            return candidate
    return None


def ensure_shadow_cover(cover_path: Path, folder_path: Path, original_folder_path: Path):
    """
    Genera una portada con sombra si no existe y devuelve su ruta.
    """
    if cover_path.name.endswith("_shadow.png"):
        if is_valid_image(cover_path):
            return cover_path
        base_cover = find_base_cover_for_shadow(cover_path)
        if base_cover and is_valid_image(base_cover):
            print(f"[COVER] Sombra corrupta en {cover_path.name}, regenerando...")
            cover_path = base_cover
        else:
            print(f"[COVER] Sombra corrupta y sin base valida: {cover_path.name}")
            return cover_path

    shadow_name = f"{cover_path.stem}_shadow.png"
    shadow_path = folder_path / shadow_name
    if shadow_path.exists() and not is_valid_image(shadow_path):
        try:
            shadow_path.unlink()
        except Exception:
            pass
    if not shadow_path.exists():
        try:
            add_shadow(
                str(cover_path),
                str(shadow_path),
                SHADOW_OPACITY,
                SHADOW_DIRECTION,
                SHADOW_DISTANCE,
                SHADOW_SOFTNESS
            )
        except Exception as e:
            print(f"[COVER] Error creando sombra: {e}")
            return cover_path

    if original_folder_path and original_folder_path != folder_path:
        dest_shadow = original_folder_path / shadow_name
        if not dest_shadow.exists():
            try:
                shutil.copy2(shadow_path, dest_shadow)
            except Exception:
                pass

    return shadow_path


def normalize_track_title(title: str) -> str:
    """
    Normaliza titulos de pistas eliminando prefijos numericos.
    """
    cleaned = re.sub(r"^\s*\d+\s*[-._)\]]\s*", "", title)
    cleaned = re.sub(r"^\s*\d+\s+", "", cleaned)
    return cleaned.strip()


def normalize_compare(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def strip_album_suffix(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\s*[\[(]\s*\d{4}\s*[\])]\s*$", "", cleaned)
    cleaned = re.sub(
        r"\s*[\[(]\s*(ep|album|demo|single|split|compilation|live|boxset)\s*[\])]\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def parse_band_album_from_folder(folder_name: str):
    if " - " in folder_name:
        band, album = folder_name.split(" - ", 1)
        return band.strip(), strip_album_suffix(album)
    return folder_name.strip(), None


def clean_track_title(title: str, band_name, album_name, fallback_title: str) -> str:
    cleaned = normalize_track_title(title)
    if not cleaned:
        cleaned = normalize_track_title(fallback_title) or fallback_title

    band = band_name or ""
    album = album_name or ""

    if band and album:
        patterns = [
            rf"^\s*{re.escape(band)}\s*[-–:|]+\s*{re.escape(album)}\s*[-–:|]+\s*",
            rf"^\s*{re.escape(album)}\s*[-–:|]+\s*{re.escape(band)}\s*[-–:|]+\s*",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    if band:
        cleaned = re.sub(rf"^\s*{re.escape(band)}\s*[-–:|]+\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"\s*[-–:|]+\s*{re.escape(band)}\s*$", "", cleaned, flags=re.IGNORECASE)
    if album:
        cleaned = re.sub(rf"^\s*{re.escape(album)}\s*[-–:|]+\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"\s*[-–:|]+\s*{re.escape(album)}\s*$", "", cleaned, flags=re.IGNORECASE)

    cleaned = cleaned.strip()
    if not cleaned:
        cleaned = normalize_track_title(fallback_title) or fallback_title

    band_norm = normalize_compare(band) if band else ""
    album_norm = normalize_compare(album) if album else ""
    title_norm = normalize_compare(cleaned)
    if title_norm in {band_norm, album_norm, f"{band_norm} {album_norm}", f"{album_norm} {band_norm}"}:
        cleaned = normalize_track_title(fallback_title) or fallback_title
    return cleaned


def cargar_env(env_path=".env"):
    global ENV_LOADED
    if ENV_LOADED:
        return
    ENV_LOADED = True
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                os.environ[key] = val


def crear_sesion_autenticada():
    global DEATHGRIND_SESSION
    if DEATHGRIND_SESSION is not None:
        return DEATHGRIND_SESSION

    email = os.environ.get("DEATHGRIND_EMAIL")
    password = os.environ.get("DEATHGRIND_PASSWORD")
    if not email or not password:
        print("[TRACKLIST] Credenciales DeathGrind no encontradas, se usa audio local.")
        return None

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json",
    })

    try:
        session.get(f"{BASE_URL}/auth/sign-in", timeout=30)
        csrf_token = session.cookies.get("csrfToken", "")
        headers = {"x-csrf-token": csrf_token, "x-uuid": uuid.uuid4().hex}
        response = session.post(
            f"{API_URL}/auth/login",
            json={"login": email, "password": password},
            headers=headers,
            timeout=30,
        )
    except requests.RequestException as exc:
        print(f"[TRACKLIST] Error de conexion DeathGrind: {exc}")
        return None

    if response.status_code not in (200, 202):
        print(f"[TRACKLIST] Error login DeathGrind: {response.status_code}")
        return None

    csrf_token = session.cookies.get("csrfToken", "")
    session.headers.update({"x-csrf-token": csrf_token, "x-uuid": headers["x-uuid"]})
    DEATHGRIND_SESSION = session
    return session


def api_get(session, endpoint, params=None, max_retries=5):
    if session is None:
        return None
    url = f"{API_URL}{endpoint}"
    retries_429 = 0
    retries_error = 0

    while True:
        try:
            response = session.get(url, params=params, timeout=30)
            if response.status_code == 429:
                retries_429 += 1
                if MAX_RETRIES_429 is not None and retries_429 >= MAX_RETRIES_429:
                    print("[TRACKLIST] Rate limit DeathGrind, se omite API.")
                    return None
                wait_time = DELAY_BASE_429 * retries_429
                print(f"[TRACKLIST] Rate limit DeathGrind, esperando {wait_time}s...")
                time.sleep(wait_time)
                continue
            if response.status_code != 200:
                retries_error += 1
                if retries_error >= max_retries:
                    return None
                time.sleep(2)
                continue
            return response.json()
        except requests.RequestException:
            retries_error += 1
            if retries_error >= max_retries:
                return None
            time.sleep(2)


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def normalize_album_name(value: str) -> str:
    return normalize_name(strip_album_suffix(value))


def get_repertorio_csv_path():
    csv_path = os.environ.get("DEATHGRIND_REPERTORIO_CSV")
    if csv_path:
        return csv_path
    default_csv = Path("/home/banar/Desktop/scrapper-deathgrind/data/bandas_completo.csv")
    if default_csv.exists():
        return str(default_csv)
    return None


def buscar_post_id_en_csv(csv_path, band_name, album_name):
    if not csv_path:
        return None
    path = Path(csv_path)
    if not path.exists():
        return None

    norm_band = normalize_name(band_name)
    norm_album = normalize_album_name(album_name) if album_name else ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                album = row.get("album") or ""
                title = row.get("title") or ""
                if not album and title and " - " in title:
                    album = title.split(" - ", 1)[1]
                if norm_album and normalize_album_name(album) != norm_album:
                    continue
                bands_raw = row.get("bands") or row.get("band") or ""
                try:
                    bands = ast.literal_eval(bands_raw) if bands_raw else []
                except Exception:
                    bands = [bands_raw]
                bands = [band for band in bands if band]
                if bands and not any(normalize_name(band) == norm_band for band in bands):
                    continue
                post_id = row.get("post_id")
                if post_id:
                    try:
                        return int(post_id)
                    except ValueError:
                        return None
    except Exception:
        return None
    return None


def match_post(post, band_name, album_name):
    post_album = post.get("album") or post.get("title") or ""
    bands = post.get("bands", [])
    band_match = False
    for band in bands:
        name = band.get("name") if isinstance(band, dict) else str(band)
        if normalize_name(name) == normalize_name(band_name):
            band_match = True
            break
    if not band_match:
        return False
    if album_name:
        return normalize_album_name(post_album) == normalize_album_name(album_name)
    return True


def buscar_post_id_en_api(session, band_name, album_name):
    album_search = strip_album_suffix(album_name or "")
    query = f"{band_name} {album_search}".strip()
    data = api_get(session, "/posts/filter", params={"search": query})
    if not data:
        return None
    posts = data.get("posts", [])
    for post in posts:
        if match_post(post, band_name, album_name):
            return post.get("postId") or post.get("id")

    offset = data.get("offset")
    while data.get("hasMore") and offset is not None:
        data = api_get(session, "/posts/filter", params={"search": query, "offset": offset})
        if not data:
            break
        posts = data.get("posts", [])
        for post in posts:
            if match_post(post, band_name, album_name):
                return post.get("postId") or post.get("id")
        offset = data.get("offset")
    return None


TRACKLIST_KEYS = (
    "tracklist",
    "trackList",
    "tracks",
    "track",
    "songs",
    "songList",
    "track_listing",
    "trackListing",
)


def find_tracklist_in_data(data, depth=0, max_depth=3):
    if depth > max_depth:
        return None
    if isinstance(data, dict):
        for key in TRACKLIST_KEYS:
            if key in data:
                return data.get(key)
        for value in data.values():
            if isinstance(value, (dict, list)):
                found = find_tracklist_in_data(value, depth + 1, max_depth)
                if found is not None:
                    return found
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                found = find_tracklist_in_data(item, depth + 1, max_depth)
                if found is not None:
                    return found
    return None


def parse_track_number(value):
    if not value:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"\d+", str(value))
    if match:
        return int(match.group(0))
    return None


def parse_track_title_item(item):
    if isinstance(item, dict):
        for key in ("title", "name", "track", "song", "trackTitle", "track_name"):
            value = item.get(key)
            if value:
                return normalize_track_title(str(value))
        return None
    if isinstance(item, str):
        return normalize_track_title(item)
    return None


def parse_track_number_item(item):
    if isinstance(item, dict):
        for key in ("trackNumber", "track_number", "track", "position", "index", "number"):
            value = item.get(key)
            track_number = parse_track_number(value)
            if track_number is not None:
                return track_number
    if isinstance(item, str):
        return parse_track_number(item)
    return None


def parse_tracklist_raw(raw):
    if raw is None:
        return []
    if isinstance(raw, dict):
        for key in TRACKLIST_KEYS:
            if key in raw:
                return parse_tracklist_raw(raw.get(key))
        return []

    entries = []
    if isinstance(raw, str):
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            title = normalize_track_title(line)
            track_number = parse_track_number(line)
            if title:
                entries.append((track_number, idx, title))
    elif isinstance(raw, list):
        for idx, item in enumerate(raw):
            title = parse_track_title_item(item)
            if not title:
                continue
            track_number = parse_track_number_item(item)
            entries.append((track_number, idx, title))

    if not entries:
        return []

    if any(track_number is not None for track_number, _idx, _title in entries):
        entries.sort(key=lambda item: (item[0] if item[0] is not None else 9999, item[1]))
    return [title for _track_number, _idx, title in entries]


def obtener_tracklist_api(session, post_id):
    if not session or not post_id:
        return []
    data = api_get(session, f"/posts/{post_id}")
    if not data:
        return []
    raw = find_tracklist_in_data(data)
    return parse_tracklist_raw(raw)


def obtener_tracklist_deathgrind(folder_name, audio_count):
    cargar_env()
    session = crear_sesion_autenticada()
    if not session:
        return []

    band_name, album_name = parse_band_album_from_folder(folder_name)
    csv_path = get_repertorio_csv_path()
    post_id = buscar_post_id_en_csv(csv_path, band_name, album_name)
    if not post_id:
        post_id = buscar_post_id_en_api(session, band_name, album_name)
    if not post_id:
        return []

    titles = obtener_tracklist_api(session, post_id)
    if titles and len(titles) == audio_count:
        return titles
    return []


def build_tracklist(audio_files, folder_name=None, api_titles=None):
    """
    Construye una lista de pistas con tiempos de inicio/fin.
    """
    band_name = None
    album_name = None
    if folder_name:
        band_name, album_name = parse_band_album_from_folder(folder_name)

    use_api = False
    clean_api_titles = []
    if api_titles and len(api_titles) == len(audio_files):
        clean_api_titles = [normalize_track_title(title) for title in api_titles]
        if all(title for title in clean_api_titles):
            use_api = True

    tracks = []
    current = 0.0
    for idx, audio_file in enumerate(audio_files, start=1):
        duration = get_audio_duration(audio_file)
        if duration <= 0:
            duration = 0.1
        fallback = audio_file.stem
        if use_api:
            title = clean_api_titles[idx - 1]
        else:
            title = clean_track_title(audio_file.stem, band_name, album_name, fallback)
        if not title:
            title = normalize_track_title(fallback) or fallback
        start = current
        end = current + duration
        tracks.append({
            "index": idx,
            "title": title,
            "start": start,
            "end": end,
        })
        current = end
    return tracks


def find_font_path(bold: bool = False):
    """
    Encuentra una fuente TTF disponible en el sistema.
    """
    if bold:
        candidates = [
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_font(font_size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """
    Carga una fuente con fallback al default de PIL.
    """
    font_path = find_font_path(bold=bold)
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            pass
    return ImageFont.load_default()


def measure_text(draw: ImageDraw.ImageDraw, text: str, font) -> float:
    """
    Mide el ancho del texto con el font dado.
    """
    try:
        return draw.textlength(text, font=font)
    except Exception:
        try:
            return font.getlength(text)
        except Exception:
            return font.getsize(text)[0]


def truncate_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    """
    Recorta el texto para que entre en el ancho disponible.
    """
    if measure_text(draw, text, font) <= max_width:
        return text
    ellipsis = "..."
    trimmed = text
    while trimmed and measure_text(draw, trimmed + ellipsis, font) > max_width:
        trimmed = trimmed[:-1]
    return (trimmed + ellipsis) if trimmed else text


def font_line_height(font, fallback_size: int) -> int:
    try:
        ascent, descent = font.getmetrics()
        if ascent or descent:
            return ascent + descent
    except Exception:
        pass
    try:
        bbox = font.getbbox("HgÁÉqy")
        return bbox[3] - bbox[1]
    except Exception:
        return fallback_size


def wrap_text_lines(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, max_lines: int):
    """
    Divide un texto en varias lineas sin exceder el ancho.
    """
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = ""
    idx = 0
    while idx < len(words):
        word = words[idx]
        if not current:
            if measure_text(draw, word, font) > max_width:
                return None
            current = word
            idx += 1
            continue

        candidate = f"{current} {word}"
        if measure_text(draw, candidate, font) <= max_width:
            current = candidate
            idx += 1
            continue

        lines.append(current)
        current = ""
        if len(lines) >= max_lines:
            return None

    if current:
        lines.append(current)

    for line in lines:
        if measure_text(draw, line, font) > max_width:
            return None

    return lines


def invert_color(avg_color):
    """
    Invierte un color RGB promedio (contraparte directa).
    """
    if not avg_color:
        return (255, 255, 255)
    r, g, b = avg_color
    return (255 - int(r), 255 - int(g), 255 - int(b))


def generate_tracklist_overlays(
    cover_overlay_path: Path,
    text_color,
    tracks,
    output_dir: Path,
    width: int,
    height: int
):
    """
    Genera overlays PNG con portada izquierda y lista de canciones.
    """
    if not cover_overlay_path or not cover_overlay_path.exists():
        return None, None
    if not tracks:
        return None, None

    cover_image = Image.open(cover_overlay_path).convert("RGBA")
    aspect = cover_image.width / cover_image.height

    left_margin = int(width * TRACKLIST_SIDE_MARGIN_RATIO)
    right_margin = int(width * TRACKLIST_RIGHT_MARGIN_RATIO)
    gap = int(width * TRACKLIST_GAP_RATIO)
    top_margin = int(height * TRACKLIST_TOP_MARGIN_RATIO)
    bottom_margin = top_margin

    cover_height = None
    cover_width = None
    list_width = None
    list_x = None

    for scale in TRACKLIST_COVER_SCALES:
        candidate_height = int(height * scale)
        candidate_width = int(candidate_height * aspect)
        candidate_list_x = left_margin + candidate_width + gap
        candidate_list_width = width - candidate_list_x - right_margin
        if candidate_list_width >= int(width * TRACKLIST_MIN_LIST_WIDTH_RATIO):
            cover_height = candidate_height
            cover_width = candidate_width
            list_width = candidate_list_width
            list_x = candidate_list_x
            break

    if cover_height is None or list_width is None:
        return None, None

    total_tracks = len(tracks)
    page_size = TRACKLIST_PAGE_SIZE or total_tracks
    page_size = max(1, min(page_size, total_tracks))

    max_font_size = min(TRACKLIST_MAX_FONT_SIZE, int(height * 0.05))
    available_height = height - top_margin - bottom_margin
    raw_font_size = int(available_height / (page_size * (1.0 + TRACKLIST_LINE_SPACING)))
    font_size = min(max_font_size, raw_font_size)
    if font_size < TRACKLIST_MIN_FONT_SIZE:
        return None, None

    base_font = None
    highlight_font = None
    base_line_height = None
    highlight_line_height = None
    line_texts_by_track = None
    max_total_height = None
    while font_size >= TRACKLIST_MIN_FONT_SIZE:
        base_font = load_font(font_size, bold=True)
        highlight_size = int(round(font_size * 1.25))
        if highlight_size <= font_size:
            highlight_size = font_size + 6
        highlight_size = min(highlight_size, font_size + 18)
        highlight_font = load_font(highlight_size, bold=True)
        base_text_height = font_line_height(base_font, font_size)
        highlight_text_height = font_line_height(highlight_font, highlight_size)

        base_line_height = int(base_text_height * (1.0 + TRACKLIST_LINE_SPACING))
        highlight_line_height = int(highlight_text_height * (1.0 + TRACKLIST_LINE_SPACING))

        dummy = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy)

        line_texts_by_track = []
        total_lines = 0
        max_lines_in_track = 1
        for track in tracks:
            line_text = f"{track['index']:02d}. {track['title']}"
            wrapped = wrap_text_lines(
                draw,
                line_text,
                highlight_font,
                list_width,
                TRACKLIST_MAX_LINES_PER_TRACK,
            )
            if not wrapped:
                line_texts_by_track = None
                break
            line_texts_by_track.append(wrapped)
            total_lines += len(wrapped)
            max_lines_in_track = max(max_lines_in_track, len(wrapped))

        if not line_texts_by_track:
            font_size -= 1
            continue

        if page_size >= total_tracks:
            max_page_lines = total_lines
        else:
            max_page_lines = 0
            for start in range(0, total_tracks, page_size):
                end = min(start + page_size, total_tracks)
                page_lines = sum(len(line_texts_by_track[i]) for i in range(start, end))
                if page_lines > max_page_lines:
                    max_page_lines = page_lines

        base_total_height = max_page_lines * base_line_height
        max_extra = (highlight_line_height - base_line_height) * max_lines_in_track
        max_total_height = base_total_height + max_extra
        if max_total_height <= available_height:
            break
        font_size -= 1

    if (
        font_size < TRACKLIST_MIN_FONT_SIZE
        or base_font is None
        or highlight_font is None
        or base_line_height is None
        or highlight_line_height is None
        or not line_texts_by_track
        or max_total_height is None
    ):
        return None, None

    cover_resized = cover_image.resize((cover_width, cover_height), Image.LANCZOS)
    cover_y = int((height - cover_height) / 2)

    text_color = text_color or (255, 255, 255)
    base_luminance = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]
    shadow_rgb = (20, 20, 20) if base_luminance > 128 else (240, 240, 240)
    base_alpha = 200
    highlight_alpha = 255

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    page_line_counts = []
    for start in range(0, total_tracks, page_size):
        end = min(start + page_size, total_tracks)
        page_line_counts.append(sum(len(line_texts_by_track[i]) for i in range(start, end)))

    for highlight_idx in range(total_tracks):
        page_index = highlight_idx // page_size
        page_start = page_index * page_size
        page_end = min(page_start + page_size, total_tracks)
        page_lines = page_line_counts[page_index] if page_line_counts else 0
        highlight_lines = len(line_texts_by_track[highlight_idx])
        page_height = (
            page_lines * base_line_height
            + (highlight_line_height - base_line_height) * highlight_lines
        )
        start_y = max(top_margin, int((height - page_height) / 2))
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        overlay.paste(cover_resized, (left_margin, cover_y), cover_resized)
        draw = ImageDraw.Draw(overlay)

        shadow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)

        y = start_y
        for idx in range(page_start, page_end):
            lines = line_texts_by_track[idx]
            is_highlight = idx == highlight_idx
            font_to_use = highlight_font if is_highlight else base_font
            line_height = highlight_line_height if is_highlight else base_line_height
            shadow_alpha = (
                TRACKLIST_TEXT_SHADOW_ALPHA_HIGHLIGHT
                if is_highlight
                else TRACKLIST_TEXT_SHADOW_ALPHA
            )
            shadow_fill = (shadow_rgb[0], shadow_rgb[1], shadow_rgb[2], shadow_alpha)

            for line_text in lines:
                shadow_draw.text(
                    (list_x + TRACKLIST_TEXT_SHADOW_OFFSET, y + TRACKLIST_TEXT_SHADOW_OFFSET),
                    line_text,
                    font=font_to_use,
                    fill=shadow_fill,
                )
                y += line_height

        if TRACKLIST_TEXT_SHADOW_BLUR > 0:
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(TRACKLIST_TEXT_SHADOW_BLUR))
        overlay = Image.alpha_composite(overlay, shadow_layer)
        draw = ImageDraw.Draw(overlay)

        y = start_y
        for idx in range(page_start, page_end):
            lines = line_texts_by_track[idx]
            is_highlight = idx == highlight_idx
            alpha = highlight_alpha if is_highlight else base_alpha
            fill = (text_color[0], text_color[1], text_color[2], alpha)
            font_to_use = highlight_font if is_highlight else base_font
            line_height = highlight_line_height if is_highlight else base_line_height

            for line_text in lines:
                draw.text((list_x, y), line_text, font=font_to_use, fill=fill)
                y += line_height

        out_path = output_dir / f"track_{highlight_idx:03d}.png"
        overlay.save(out_path, "PNG")

    tracklist_path = output_dir / "tracklist.tsv"
    with open(tracklist_path, "w", encoding="utf-8") as f:
        for idx, track in enumerate(tracks):
            f.write(f"{idx}\t{track['start']:.3f}\t{track['end']:.3f}\t{track['title']}\n")

    return tracklist_path, output_dir


def extract_cover_bytes_from_audio(audio_path: Path):
    """
    Extrae bytes de portada embebida en metadata (mp3/flac/m4a).
    """
    try:
        audio = MutagenFile(str(audio_path), easy=False)
    except Exception:
        return None

    if audio is None:
        return None

    # FLAC: pictures
    try:
        if hasattr(audio, "pictures") and audio.pictures:
            return audio.pictures[0].data
    except Exception:
        pass

    # MP4/M4A: covr
    try:
        if hasattr(audio, "tags") and audio.tags:
            covr = audio.tags.get("covr")
            if covr:
                return bytes(covr[0])
    except Exception:
        pass

    # MP3: ID3 APIC
    try:
        if hasattr(audio, "tags") and audio.tags:
            if hasattr(audio.tags, "getall"):
                apic = audio.tags.getall("APIC")
                if apic:
                    return apic[0].data
            for key in audio.tags.keys():
                if str(key).startswith("APIC"):
                    return audio.tags[key].data
    except Exception:
        pass

    return None


def save_cover_bytes(image_bytes: bytes, output_path: Path) -> bool:
    """
    Guarda bytes de imagen como archivo JPEG.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(output_path, format="JPEG", quality=95)
        return True
    except Exception as e:
        print(f"[COVER] Error guardando portada embebida: {e}")
        return False


def extract_cover_from_audio_files(audio_files, folder_path, original_folder_path=None):
    """
    Busca portada embebida en metadata y la guarda como cover.jpg.
    """
    for audio_file in audio_files:
        image_bytes = extract_cover_bytes_from_audio(audio_file)
        if not image_bytes:
            continue

        cover_path = get_unique_file_path(folder_path, "cover.jpg")
        if save_cover_bytes(image_bytes, cover_path):
            if original_folder_path and original_folder_path != folder_path:
                try:
                    dest_path = get_unique_file_path(original_folder_path, "cover.jpg")
                    shutil.copy2(cover_path, dest_path)
                except Exception:
                    pass
            return cover_path

    return None


def analyze_audio_amplitude(audio_path, fps=30):
    """
    Analiza la amplitud del audio usando FFmpeg (más rápido que librosa)
    Retorna valores normalizados para escala reactiva al audio
    """
    try:
        # Usar FFmpeg para extraer volumen RMS
        # Esto es MÁS RÁPIDO que librosa y ya tienes FFmpeg instalado
        cmd = [
            'ffmpeg',
            '-i', str(audio_path),
            '-af', f'astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-',
            '-f', 'null',
            '-'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            stderr=subprocess.DEVNULL
        )

        # Parsear salida (simplificado - usaremos valores fijos para velocidad)
        # Para 3 videos diarios, pre-calcular audio es overhead innecesario
        # Usamos expresión dinámica de FFmpeg directamente
        return None

    except Exception as e:
        print(f"Error analizando audio {audio_path}: {e}")
        return None


def generate_wiggle_expression(seed):
    """
    Genera valores aleatorios para simular wiggle(3,3) de After Effects
    Usa un seed para reproducibilidad
    """
    np.random.seed(seed)
    # Generar ruido aleatorio que cambia cada ~0.33 segundos (3 veces por segundo)
    return np.random.uniform(-3, 3)


def get_audio_duration(audio_path):
    """
    Obtiene la duración del audio en segundos usando ffprobe
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        return float(result.stdout.strip())
    except Exception:
        return 0


def get_audio_bitrate_kbps(audio_path):
    """
    Obtiene el bitrate del audio en kbps usando ffprobe.
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=bit_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        bitrate_str = result.stdout.strip()
        if not bitrate_str:
            return 0
        return max(0, int(int(bitrate_str) / 1000))
    except Exception:
        return 0


def pick_output_audio_bitrate(audio_files):
    """
    Define un bitrate de salida respetando la calidad de origen.
    """
    bitrates = [get_audio_bitrate_kbps(af) for af in audio_files]
    max_bitrate = max(bitrates) if bitrates else 0

    if max_bitrate <= 0:
        return AUDIO_BITRATE

    target = min(384, max_bitrate)
    target = max(128, target)
    return f"{target}k"


def format_time(seconds):
    """
    Formatea segundos en formato HH:MM:SS
    """
    if seconds < 0:
        return "00:00:00"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_progress_bar(current_time, total_duration, elapsed_time, folder_name, bar_width=25):
    """
    Imprime una barra de progreso que se actualiza en la misma línea
    """
    if total_duration <= 0:
        return

    percentage = min(100, (current_time / total_duration) * 100)
    filled = int(bar_width * percentage / 100)
    bar = '█' * filled + '░' * (bar_width - filled)

    # Calcular tiempo restante estimado
    if current_time > 0 and elapsed_time > 0:
        speed = current_time / elapsed_time
        remaining_video_time = total_duration - current_time
        eta_seconds = remaining_video_time / speed if speed > 0 else 0
    else:
        eta_seconds = 0

    eta_str = format_time(eta_seconds)

    # \033[K limpia el resto de la línea para evitar basura
    sys.stdout.write(f'\r\033[K{folder_name[:35]} | {bar} {percentage:5.1f}% | ETA: {eta_str}')
    sys.stdout.flush()


def run_ffmpeg_command(cmd, show_progress, total_duration, folder_name, start_time):
    """
    Ejecuta FFmpeg con barra de progreso opcional.
    """
    if show_progress:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            bufsize=1,
            universal_newlines=True
        )

        stderr_output = ""
        time_pattern = re.compile(r'time=(\d+):(\d+):(\d+\.?\d*)')

        while True:
            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break

            stderr_output += line

            match = time_pattern.search(line)
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                seconds = float(match.group(3))
                current_time = hours * 3600 + minutes * 60 + seconds

                elapsed = time.time() - start_time
                print_progress_bar(current_time, total_duration, elapsed, folder_name)

        process.wait()
        print()
        return process.returncode == 0, stderr_output

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace"
    )
    return result.returncode == 0, result.stderr


def run_cpp_command(cmd, show_progress, extra_env=None):
    """
    Ejecuta el render C++ y retorna (returncode, stderr_text).
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    if show_progress:
        result = subprocess.run(cmd, env=env)
        return result.returncode, ""

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        env=env
    )
    return result.returncode, result.stderr


def is_cuda_failure(stderr_text: str) -> bool:
    if not stderr_text:
        return False
    patterns = (
        "illegal memory access",
        "CUDA_ERROR_ILLEGAL_ADDRESS",
        "out of memory",
    )
    return any(pat in stderr_text for pat in patterns)

def is_tracking_failure(stderr_text: str) -> bool:
    if not stderr_text:
        return False
    lowered = stderr_text.lower()
    return "tracking_errors.cu" in lowered or "tracking errors" in lowered


def render_video_with_cpp(
    folder_path,
    original_folder_path,
    folder_name,
    audio_files,
    cover_main,
    cover_overlay,
    tracklist_path,
    track_overlays_path,
    audio_duration,
    total_duration,
    show_progress,
    start_time,
    temp_folder_path,
    use_gpu
):
    """
    Genera el video completo en C++/CUDA y deja el audio al final con FFmpeg.
    Retorna (success, error_kind) donde error_kind puede ser "cuda" u otro motivo.
    """
    if audio_duration <= 0:
        print(f"[ERROR] Duración de audio inválida en {folder_name}")
        return False, "audio"

    if not VHS_CPP_BIN.exists():
        print(f"[ERROR] No se encontró el renderer C++ en {VHS_CPP_BIN}")
        return False, "missing_bin"

    video_only = folder_path / f"{folder_name}__video.mp4"
    audio_only = folder_path / f"{folder_name}__audio.m4a"
    final_video = folder_path / f"{folder_name}.mp4"

    num_audios = len(audio_files)
    if num_audios == 1:
        audio_concat_line = "[1:a]acopy[all_music]"
    else:
        audio_inputs = "".join([f"[{i+1}:a]" for i in range(num_audios)])
        audio_concat_line = f"{audio_inputs}concat=n={num_audios}:v=0:a=1[all_music]"

    audio_inputs_args = []
    for audio_file in audio_files:
        audio_inputs_args.extend(['-i', str(audio_file)])

    if show_progress:
        print("[AUDIO] Generando mezcla...")
    else:
        print(f"[AUDIO] {folder_name} generando mezcla...")

    audio_filter = f"""
{audio_concat_line};
[0:a]atrim=0:{INTRO_DURATION}[intro_audio];
[all_music]adelay=delays={int(INTRO_DURATION*1000)}:all=1[music_delayed];
[intro_audio][music_delayed]amix=inputs=2:duration=longest[outa]
""".strip()

    output_audio_bitrate = pick_output_audio_bitrate(audio_files)

    audio_cmd = [
        'ffmpeg',
        '-y',
        '-threads', '0',
        '-i', str(INTRO_VIDEO),
        *audio_inputs_args,
        '-filter_complex', audio_filter,
        '-map', '[outa]',
        '-c:a', 'aac',
        '-profile:a', 'aac_low',
        '-b:a', output_audio_bitrate,
        '-ar', str(AUDIO_SAMPLE_RATE),
        '-ac', '2',
        '-movflags', '+faststart',
        str(audio_only)
    ]

    audio_success, audio_stderr = run_ffmpeg_command(
        audio_cmd, False, 0, folder_name, start_time
    )
    if not audio_success:
        print(f"\n[ERROR] FFmpeg falló generando audio en {folder_name}")
        print(f"STDERR: {audio_stderr[-500:]}")
        return False, "audio"

    if show_progress:
        print("[VHS GPU] Generando video con CUDA...")
    else:
        print(f"[VHS GPU] {folder_name} generando video con CUDA...")

    base_cpp_cmd = [
        str(VHS_CPP_BIN),
        '--intro', str(INTRO_VIDEO),
        '--cover', str(cover_main),
        '--duration', str(audio_duration),
        '--output', str(video_only),
        '--intensity', str(VHS_CPP_INTENSITY),
        '--cq', str(VIDEO_CQ),
        '--preset', VIDEO_PRESET_NVENC
    ]

    if cover_overlay and Path(cover_overlay).exists():
        base_cpp_cmd.extend(['--cover-overlay', str(cover_overlay)])
    if tracklist_path and track_overlays_path:
        base_cpp_cmd.extend(['--tracklist', str(tracklist_path)])
        base_cpp_cmd.extend(['--track-overlays', str(track_overlays_path)])

    cpp_cmd = base_cpp_cmd.copy()

    if VHS_CPP_OVERLAY.exists():
        cpp_cmd.extend(['--vhs-overlay', str(VHS_CPP_OVERLAY)])
    else:
        print(f"[VHS GPU] Overlay no encontrado: {VHS_CPP_OVERLAY}")

    cpp_returncode, cpp_stderr = run_cpp_command(cpp_cmd, show_progress)
    cpp_cmd_no_overlay = base_cpp_cmd.copy()

    if cpp_returncode != 0:
        if not show_progress:
            print(f"[ERROR] VHS GPU falló en {folder_name}")
            if cpp_stderr:
                print(f"STDERR: {cpp_stderr[-500:]}")

        # Reintento sin overlay si hay error CUDA
        if is_cuda_failure(cpp_stderr) and CUDA_FAIL_FAST:
            return False, "cuda"
        if is_cuda_failure(cpp_stderr) and VHS_CPP_OVERLAY.exists():
            print(f"[VHS GPU] {folder_name} reintentando sin overlay por error CUDA...")
            retry_code, retry_stderr = run_cpp_command(cpp_cmd_no_overlay, show_progress)
            if retry_code == 0:
                print(f"[VHS GPU] {folder_name} renderizado sin overlay")
            else:
                if not show_progress and retry_stderr:
                    print(f"STDERR: {retry_stderr[-500:]}")
                cpp_stderr = retry_stderr
                cpp_returncode = retry_code
        else:
            return False, "cpp"

    if cpp_returncode != 0 and is_cuda_failure(cpp_stderr):
        if CUDA_FAIL_FAST:
            return False, "cuda"
        if not show_progress:
            if is_tracking_failure(cpp_stderr):
                print(f"[VHS GPU] {folder_name} reintentando sin tracking errors por error CUDA...")
            else:
                print(f"[VHS GPU] {folder_name} reintentando en modo seguro (sin tracking errors)...")
        tracking_env = {"VHS_DISABLE_TRACKING_ERRORS": "1"}
        retry_cmd = cpp_cmd_no_overlay if VHS_CPP_OVERLAY.exists() else cpp_cmd
        retry_code, retry_stderr = run_cpp_command(retry_cmd, show_progress, tracking_env)
        if retry_code == 0:
            if not show_progress:
                print(f"[VHS GPU] {folder_name} renderizado sin tracking errors")
        else:
            if not show_progress and retry_stderr:
                print(f"STDERR: {retry_stderr[-500:]}")
            cpp_stderr = retry_stderr
            cpp_returncode = retry_code

    if cpp_returncode != 0 and is_cuda_failure(cpp_stderr):
        if CUDA_FAIL_FAST:
            return False, "cuda"
        if not show_progress:
            print(f"[VHS GPU] {folder_name} reintentando con NVENC sin hwframes...")
        hwframes_env = {"VHS_NVENC_NO_HWFRAMES": "1"}
        retry_cmd = cpp_cmd_no_overlay if VHS_CPP_OVERLAY.exists() else cpp_cmd
        retry_code, retry_stderr = run_cpp_command(retry_cmd, show_progress, hwframes_env)
        if retry_code == 0:
            if not show_progress:
                print(f"[VHS GPU] {folder_name} renderizado sin hwframes")
        else:
            if not show_progress and retry_stderr:
                print(f"STDERR: {retry_stderr[-500:]}")
            cpp_stderr = retry_stderr
            cpp_returncode = retry_code

    if cpp_returncode != 0 and is_cuda_failure(cpp_stderr):
        if CUDA_FAIL_FAST:
            return False, "cuda"
        if not show_progress:
            print(f"[VHS GPU] {folder_name} reintentando en modo seguro (sin color bleeding/noise)...")
        safe_env = {"VHS_SAFE_MODE": "1"}
        retry_cmd = cpp_cmd_no_overlay if VHS_CPP_OVERLAY.exists() else cpp_cmd
        retry_code, retry_stderr = run_cpp_command(retry_cmd, show_progress, safe_env)
        if retry_code == 0:
            if not show_progress:
                print(f"[VHS GPU] {folder_name} renderizado en modo seguro")
        else:
            if not show_progress and retry_stderr:
                print(f"STDERR: {retry_stderr[-500:]}")
            return False, "cuda"

    if show_progress:
        print("[MUX] Pegando audio al video final...")
    else:
        print(f"[MUX] {folder_name} pegando audio al video final...")

    mux_cmd = [
        'ffmpeg',
        '-y',
        '-i', str(video_only),
        '-i', str(audio_only),
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:v', 'copy',
        '-c:a', 'copy',
        '-movflags', '+faststart',
        '-shortest',
        str(final_video)
    ]

    mux_success, mux_stderr = run_ffmpeg_command(
        mux_cmd, False, 0, folder_name, start_time
    )
    if not mux_success:
        print(f"[ERROR] Mux de audio falló en {folder_name}")
        print(f"STDERR: {mux_stderr[-500:]}")
        return False, "mux"

    if not USE_SSD_TEMP:
        for temp_file in (video_only, audio_only):
            if temp_file.exists():
                temp_file.unlink()

    if USE_SSD_TEMP and temp_folder_path:
        output_video = final_video
        final_dest = original_folder_path / f"{folder_name}.mp4"
        if output_video.exists():
            if show_progress:
                print("[SSD] Moviendo video final al disco destino...")
            shutil.move(str(output_video), str(final_dest))
        if temp_folder_path.exists():
            shutil.rmtree(temp_folder_path)

    if track_overlays_path:
        try:
            shutil.rmtree(track_overlays_path)
        except Exception:
            pass

    destination_folder = move_folder_to_upload(original_folder_path, folder_name, show_progress)
    elapsed = time.time() - start_time
    if show_progress:
        print(f"\n[ÉXITO] {folder_name} renderizado en {format_time(elapsed)}")
        print(f"[UPLOAD] Carpeta movida a: {destination_folder}")
    else:
        print(f"[ÉXITO] {folder_name} renderizado en {elapsed:.1f}s")
        print(f"[UPLOAD] Carpeta movida a: {destination_folder}")

    return True, None


def render_video(folder_path, folder_name, show_progress=False):
    """
    Renderiza un video completo usando FFmpeg
    show_progress: Si True, muestra barra de progreso en tiempo real (solo para modo single)
    """
    original_folder_path = folder_path
    temp_folder_path = None
    use_gpu = resolve_gpu_usage()

    try:
        if not show_progress:
            print(f"[INICIO] Procesando: {folder_name}")
        start_time = time.time()

        # ================================================================
        # PASO 1: Copiar a SSD si está habilitado
        # ================================================================
        if USE_SSD_TEMP:
            SSD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
            temp_folder_path = SSD_TEMP_DIR / folder_name

            if show_progress:
                print(f"[SSD] Copiando a SSD local...")

            # Copiar solo archivos necesarios (no el video si ya existe)
            if temp_folder_path.exists():
                shutil.rmtree(temp_folder_path)
            temp_folder_path.mkdir(parents=True)

            for file in os.listdir(folder_path):
                if file.lower().endswith(('.mp3', '.flac', '.wav', '.m4a', '.png', '.jpg', '.jpeg', '.txt')):
                    shutil.copy2(folder_path / file, temp_folder_path / file)

            # Usar la carpeta temporal para el renderizado
            folder_path = temp_folder_path

            if show_progress:
                print(f"[SSD] Listo - renderizando desde NVMe")

        # ================================================================
        # PASO 2: Buscar archivos necesarios
        # ================================================================
        audio_files = []
        cover_file = None

        for file in os.listdir(folder_path):
            file_lower = file.lower()
            if file_lower.endswith(('.mp3', '.flac', '.wav', '.m4a')):
                audio_files.append(folder_path / file)
            elif file_lower.endswith(('.png', '.jpg', '.jpeg')):
                # Priorizar: cover.* > *_shadow.png > cualquier imagen
                if file_lower.startswith('cover.'):
                    cover_file = folder_path / file
                elif cover_file is None or (file.endswith('_shadow.png') and not str(cover_file).endswith('_shadow.png')):
                    if cover_file is None or file.endswith('_shadow.png'):
                        cover_file = folder_path / file

        # Ordenar audios por nombre (01. xxx, 02. xxx, etc.)
        audio_files.sort(key=lambda x: x.name.lower())

        if cover_file and not is_valid_image(cover_file):
            print(f"[COVER] Portada dañada: {cover_file.name}, intentando otra...")
            cover_file = None

        if not cover_file and audio_files:
            if show_progress:
                print("[COVER] Portada no encontrada, buscando en metadata...")
            else:
                print(f"[COVER] {folder_name} portada no encontrada, buscando en metadata...")

            cover_file = extract_cover_from_audio_files(
                audio_files,
                folder_path,
                original_folder_path
            )
            if cover_file:
                if show_progress:
                    print(f"[COVER] Portada extraida: {cover_file.name}")
                else:
                    print(f"[COVER] {folder_name} portada extraida: {cover_file.name}")

        if cover_file and not is_valid_image(cover_file):
            print(f"[COVER] Portada invalida: {cover_file.name}")
            cover_file = None

        if not audio_files or not cover_file:
            print(f"[ERROR] Archivos faltantes en {folder_name} (audios: {len(audio_files)}, cover: {cover_file is not None})")
            return False

        if show_progress and len(audio_files) > 1:
            print(f"[INFO] {len(audio_files)} pistas de audio encontradas, concatenando en orden...")

        cover_main = cover_file
        cover_overlay = None

        if cover_file and cover_file.name.endswith("_shadow.png"):
            base_cover = find_base_cover_for_shadow(cover_file)
            if base_cover:
                cover_main = base_cover
                cover_overlay = cover_file

        if cover_overlay is None or not is_valid_image(cover_overlay):
            cover_overlay = ensure_shadow_cover(
                cover_main,
                folder_path,
                original_folder_path
            )

        cover_bg = cover_main

        # Extraer color promedio e invertir para usar en texto
        avg_color = extract_average_color(cover_bg)
        inv_color = invert_color(avg_color)
        comp_r, comp_g, comp_b = get_complementary_color(*avg_color)
        spectrum_color = f"0x{comp_r:02x}{comp_g:02x}{comp_b:02x}"

        tracklist_path = None
        track_overlays_path = None
        if USE_CPP_VHS:
            api_titles = obtener_tracklist_deathgrind(folder_name, len(audio_files))
            if api_titles:
                print(f"[TRACKLIST] {folder_name} titulos obtenidos desde DeathGrind.")
            else:
                print(f"[TRACKLIST] {folder_name} usando titulos de audio local.")
            tracks = build_tracklist(audio_files, folder_name=folder_name, api_titles=api_titles)
            if tracks and cover_overlay:
                overlays_dir = folder_path / "_track_overlays"
                tracklist_path, track_overlays_path = generate_tracklist_overlays(
                    cover_overlay_path=Path(cover_overlay),
                    text_color=inv_color,
                    tracks=tracks,
                    output_dir=overlays_dir,
                    width=VIDEO_WIDTH,
                    height=VIDEO_HEIGHT
                )

        # Obtener duración total de todos los audios (para la barra de progreso)
        audio_duration = sum(get_audio_duration(af) for af in audio_files)
        total_duration = INTRO_DURATION + audio_duration  # Duración total del video

        output_audio_bitrate = pick_output_audio_bitrate(audio_files)

        if show_progress:
            print(f"\n[RENDERIZANDO] {folder_name}")
            print(f"Duración estimada: {format_time(total_duration)}")
            print("")

        if USE_CPP_VHS:
            if DISABLE_CPP_ON_CUDA and CUDA_ERROR_FLAG.exists():
                if show_progress:
                    print("[COVER] C++ desactivado por error CUDA previo, usando FFmpeg...")
                else:
                    print(f"[COVER] {folder_name} usando FFmpeg por error CUDA previo...")
            else:
                cpp_success, cpp_error = render_video_with_cpp(
                    folder_path=folder_path,
                    original_folder_path=original_folder_path,
                    folder_name=folder_name,
                    audio_files=audio_files,
                    cover_main=cover_main,
                    cover_overlay=cover_overlay,
                    tracklist_path=tracklist_path,
                    track_overlays_path=track_overlays_path,
                    audio_duration=audio_duration,
                    total_duration=total_duration,
                    show_progress=show_progress,
                    start_time=start_time,
                    temp_folder_path=temp_folder_path,
                    use_gpu=use_gpu
                )
                if cpp_success:
                    return True
                if cpp_error == "cuda" and DISABLE_CPP_ON_CUDA:
                    try:
                        CUDA_ERROR_FLAG.touch()
                    except Exception:
                        pass
                if cpp_error != "cuda" or not ALLOW_FFMPEG_FALLBACK:
                    return False
                if show_progress:
                    print("[FALLBACK] CUDA falló, usando FFmpeg para este álbum...")
                else:
                    print(f"[FALLBACK] {folder_name} usando FFmpeg por error CUDA...")
                for suffix in ("__video.mp4", "__audio.m4a"):
                    tmp_file = folder_path / f"{folder_name}{suffix}"
                    if tmp_file.exists():
                        tmp_file.unlink()
                # Continuar con FFmpeg
                pass
                # continuar

        cover_height = max(1, int(VIDEO_HEIGHT * 1.02))

        # Ruta de salida
        output_file = folder_path / f"{folder_name}.mp4"

        # ====================================================================
        # FILTRO COMPLEJO DE FFMPEG CON VHS OVERLAY
        # ====================================================================

        # Flujo optimizado:
        # [0] = Intro video
        # [1] = Portada (para fondo difuminado + overlay)
        # [2] = Video VHS noise (loop de 60s para overlay auténtico)
        # [3+] = Archivos de audio
        #
        # El video VHS (content/vhs_noise.mp4) contiene:
        # - Jitter irregular
        # - Tracking errors
        # - Noise granulado
        # - Scanlines
        # Se mezcla con el fondo usando blend para efecto auténtico

        # ================================================================
        # Construir filtro de concatenación de audio dinámico
        # ================================================================
        num_audios = len(audio_files)

        # Índices de audio empiezan en 3 (después de intro, cover, vhs_noise)
        if num_audios == 1:
            audio_concat_filter = f"[3:a]"
            audio_concat_output = "[all_music]"
            audio_concat_line = f"{audio_concat_filter}acopy{audio_concat_output}"
        else:
            audio_inputs = "".join([f"[{i+3}:a]" for i in range(num_audios)])
            audio_concat_filter = f"{audio_inputs}concat=n={num_audios}:v=0:a=1"
            audio_concat_output = "[all_music]"
            audio_concat_line = f"{audio_concat_filter}{audio_concat_output}"

        # VHS noise video path
        vhs_noise_video = Path(__file__).parent / "content" / "vhs_noise.mp4"

        # ============================================================
        # FILTRO VHS CON VIDEO OVERLAY - Mucho más rápido
        # ============================================================
        # 1. Fondo: Portada difuminada con color bleeding y grading VHS
        # 2. VHS Overlay: Video de ruido VHS en loop mezclado con blend
        # 3. Portada: Cover limpio sin efectos (nítido)
        # 4. Transición: Fade de intro a contenido
        # ============================================================

        # Determinar si usar filtros GPU (scale_cuda, overlay_cuda)
        # Nota: Algunos filtros VHS no tienen equivalente CUDA, requieren CPU
        if use_gpu:
            # ============================================================
            # FILTRO GPU OPTIMIZADO
            # Pipeline: CUDA decode → hwdownload → CPU filters → hwupload → CUDA overlay → NVENC
            #
            # Los filtros VHS (color bleeding, eq, colorlevels) no tienen CUDA
            # pero scale_cuda y overlay_cuda sí están disponibles
            # ============================================================
            filter_complex = f"""
[0:v]hwdownload,format=nv12,scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:flags=lanczos,fps={FPS}[intro];
[1:v]scale={int(VIDEO_WIDTH*1.2)}:-1:flags=lanczos,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},fps={FPS},
format=yuv444p,
split=3[luma][chroma1][chroma2];
[luma]extractplanes=y[y];
[chroma1]extractplanes=u,gblur=sigma=12:sigmaV=0[u];
[chroma2]extractplanes=v,gblur=sigma=12:sigmaV=0[v];
[y][u][v]mergeplanes=0x001020:yuv444p,
format=yuv420p,
gblur=sigma=20,
rgbashift=rh=-4:bh=4:rv=1:bv=-1:edge=wrap,
eq=saturation=0.5:contrast=1.2:brightness=0.02:gamma=1.15,
colorlevels=rimax=0.9:gimax=0.9:bimax=0.9:romin=0.05:gomin=0.05:bomin=0.05,
colortemperature=temperature=5500[bg_base];
[2:v]scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:flags=bilinear,fps={FPS},
format=yuv420p,loop=-1:size=1800,setpts=N/{FPS}/TB[vhs_loop];
[bg_base][vhs_loop]blend=all_mode=softlight:all_opacity=0.55[bg];
[1:v]scale=-1:{cover_height}:flags=lanczos,fps={FPS},format=yuva420p,unsharp=5:5:1.2:5:5:0.0[cover];
[bg][cover]overlay=(W-w)/2:(H-h)/2[content];
[intro][content]xfade=transition=fade:duration=1:offset={INTRO_DURATION-1}[outv];
{audio_concat_line};
[0:a]atrim=0:{INTRO_DURATION}[intro_audio];
[all_music]adelay=delays={int(INTRO_DURATION*1000)}:all=1[music_delayed];
[intro_audio][music_delayed]amix=inputs=2:duration=longest[outa]
""".strip()
        else:
            # ============================================================
            # FILTRO CPU (fallback)
            # ============================================================
            filter_complex = f"""
[0:v]scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:flags=lanczos,fps={FPS}[intro];
[1:v]scale={int(VIDEO_WIDTH*1.2)}:-1:flags=lanczos,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},fps={FPS},
format=yuv444p,
split=3[luma][chroma1][chroma2];
[luma]extractplanes=y[y];
[chroma1]extractplanes=u,gblur=sigma=12:sigmaV=0[u];
[chroma2]extractplanes=v,gblur=sigma=12:sigmaV=0[v];
[y][u][v]mergeplanes=0x001020:yuv444p,
format=yuv420p,
gblur=sigma=20,
rgbashift=rh=-4:bh=4:rv=1:bv=-1:edge=wrap,
eq=saturation=0.5:contrast=1.2:brightness=0.02:gamma=1.15,
colorlevels=rimax=0.9:gimax=0.9:bimax=0.9:romin=0.05:gomin=0.05:bomin=0.05,
colortemperature=temperature=5500[bg_base];
[2:v]scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:flags=bilinear,fps={FPS},
format=yuv420p,loop=-1:size=1800,setpts=N/{FPS}/TB[vhs_loop];
[bg_base][vhs_loop]blend=all_mode=softlight:all_opacity=0.55[bg];
[1:v]scale=-1:{cover_height}:flags=lanczos,fps={FPS},format=yuva420p,unsharp=5:5:1.2:5:5:0.0[cover];
[bg][cover]overlay=(W-w)/2:(H-h)/2[content];
[intro][content]xfade=transition=fade:duration=1:offset={INTRO_DURATION-1}[outv];
{audio_concat_line};
[0:a]atrim=0:{INTRO_DURATION}[intro_audio];
[all_music]adelay=delays={int(INTRO_DURATION*1000)}:all=1[music_delayed];
[intro_audio][music_delayed]amix=inputs=2:duration=longest[outa]
""".strip()

        # Construir lista de inputs de audio
        audio_inputs_args = []
        for audio_file in audio_files:
            audio_inputs_args.extend(['-i', str(audio_file)])

        # Comando FFmpeg con VHS overlay
        # Inputs:
        #   [0] = Intro video
        #   [1] = Portada (para fondo difuminado + cover limpio)
        #   [2] = Video VHS noise (loop)
        #   [3+] = Archivos de audio

        if use_gpu:
            # Pipeline GPU optimizado:
            # NVDEC (decode) → CUDA filters → NVENC (encode)
            # Mantiene frames en VRAM evitando transferencias PCIe
            cmd = [
                'ffmpeg',
                '-y',
                '-hwaccel', 'cuda',                    # Usar NVDEC para decode
                '-hwaccel_output_format', 'cuda',     # Mantener frames en GPU
                '-threads', '0',
                '-i', str(INTRO_VIDEO),               # [0] Intro (decode GPU)
                '-loop', '1', '-t', '9999', '-i', str(cover_main),  # [1] Portada
                '-stream_loop', '-1', '-i', str(vhs_noise_video),   # [2] VHS noise (decode CPU)
                *audio_inputs_args,                   # [3+] Audio
                '-filter_complex', filter_complex,
                '-map', '[outv]',
                '-map', '[outa]',
                '-c:v', 'h264_nvenc',
                '-preset', VIDEO_PRESET_NVENC,
                '-tune', 'hq',
                '-cq', str(VIDEO_CQ),
                *NVENC_EXTRA_OPTS,
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-profile:a', 'aac_low',
                '-b:a', output_audio_bitrate,
                '-ar', str(AUDIO_SAMPLE_RATE),
                '-ac', '2',
                '-shortest',
                str(output_file)
            ]
        else:
            cmd = [
                'ffmpeg',
                '-y',
                '-threads', '0',
                '-i', str(INTRO_VIDEO),           # [0] Intro
                '-loop', '1', '-t', '9999', '-i', str(cover_main),  # [1] Portada
                '-stream_loop', '-1', '-i', str(vhs_noise_video),   # [2] VHS noise (loop infinito)
                *audio_inputs_args,               # [3+] Audio
                '-filter_complex', filter_complex,
                '-map', '[outv]',
                '-map', '[outa]',
                '-c:v', 'libx264',
                '-preset', VIDEO_PRESET_CPU,
                '-crf', str(VIDEO_CRF),
                '-tune', 'fastdecode',
                '-profile:v', 'high',
                '-bf', '2',
                '-g', str(FPS // 2),
                '-x264-params', 'open-gop=0',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-profile:a', 'aac_low',
                '-b:a', output_audio_bitrate,
                '-ar', str(AUDIO_SAMPLE_RATE),
                '-ac', '2',
                '-movflags', '+faststart',
                '-shortest',
                str(output_file)
            ]

        # Ejecutar FFmpeg
        if show_progress:
            # Modo con barra de progreso (capturar stderr en tiempo real)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                bufsize=1,
                universal_newlines=True
            )

            # Leer stderr en tiempo real para obtener progreso
            stderr_output = ""
            time_pattern = re.compile(r'time=(\d+):(\d+):(\d+\.?\d*)')

            while True:
                # Leer una línea del stderr
                line = process.stderr.readline()
                if not line and process.poll() is not None:
                    break

                stderr_output += line

                # Buscar el tiempo actual en la salida
                match = time_pattern.search(line)
                if match:
                    hours = int(match.group(1))
                    minutes = int(match.group(2))
                    seconds = float(match.group(3))
                    current_time = hours * 3600 + minutes * 60 + seconds

                    elapsed = time.time() - start_time
                    print_progress_bar(current_time, total_duration, elapsed, folder_name)

            process.wait()
            print()  # Nueva línea después de la barra de progreso

            if process.returncode == 0:
                # Mover video de SSD a disco original si aplica
                if USE_SSD_TEMP and temp_folder_path:
                    output_video = temp_folder_path / f"{folder_name}.mp4"
                    final_video = original_folder_path / f"{folder_name}.mp4"
                    if output_video.exists():
                        if show_progress:
                            print(f"[SSD] Moviendo video a disco destino...")
                        shutil.move(str(output_video), str(final_video))
                        shutil.rmtree(temp_folder_path)  # Limpiar temporal
                        if show_progress:
                            print(f"[SSD] Limpieza completada")

                elapsed = time.time() - start_time
                destination_folder = move_folder_to_upload(original_folder_path, folder_name, show_progress)
                print(f"\n[ÉXITO] {folder_name} renderizado en {format_time(elapsed)}")
                print(f"[UPLOAD] Carpeta movida a: {destination_folder}")
                return True
            else:
                # Limpiar temporales en caso de error
                if USE_SSD_TEMP and temp_folder_path and temp_folder_path.exists():
                    shutil.rmtree(temp_folder_path)
                print(f"\n[ERROR] FFmpeg falló en {folder_name}")
                print(f"STDERR: {stderr_output[-500:]}")
                return False
        else:
            # Modo sin progreso (para renderizado paralelo)
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace"
            )

            if result.returncode == 0:
                # Mover video de SSD a disco original si aplica
                if USE_SSD_TEMP and temp_folder_path:
                    output_video = temp_folder_path / f"{folder_name}.mp4"
                    final_video = original_folder_path / f"{folder_name}.mp4"
                    if output_video.exists():
                        shutil.move(str(output_video), str(final_video))
                        shutil.rmtree(temp_folder_path)

                elapsed = time.time() - start_time
                destination_folder = move_folder_to_upload(original_folder_path, folder_name, show_progress)
                print(f"[ÉXITO] {folder_name} renderizado en {elapsed:.1f}s")
                print(f"[UPLOAD] Carpeta movida a: {destination_folder}")
                return True
            else:
                # Limpiar temporales en caso de error
                if USE_SSD_TEMP and temp_folder_path and temp_folder_path.exists():
                    shutil.rmtree(temp_folder_path)
                print(f"[ERROR] FFmpeg falló en {folder_name}")
                print(f"STDERR: {result.stderr[-500:]}")
                return False

    except Exception as e:
        # Limpiar temporales en caso de excepción
        if USE_SSD_TEMP and temp_folder_path and temp_folder_path.exists():
            shutil.rmtree(temp_folder_path)
        print(f"[EXCEPCIÓN] Error procesando {folder_name}: {e}")
        return False


def process_folders_parallel(folders_override=None, staging_ctx=None):
    """
    Procesa carpetas en paralelo (hasta MAX_PARALLEL_RENDERS simultáneos)
    """
    if folders_override is not None:
        folders = folders_override
    else:
        # Recoger todas las carpetas
        folders = [
            (MAIN_DIR / folder_name, folder_name)
            for folder_name in os.listdir(MAIN_DIR)
            if (MAIN_DIR / folder_name).is_dir()
        ]

    # Mezclar aleatoriamente
    random.shuffle(folders)

    # Limitar cantidad
    folders = folders[:MAX_FOLDERS_TO_PROCESS]

    print(f"\n{'='*60}")
    print(f"INICIANDO RENDERIZADO PARALELO")
    print(f"Carpetas a procesar: {len(folders)}")
    print(f"Renders paralelos: {MAX_PARALLEL_RENDERS}")
    print(f"{'='*60}\n")

    # Procesar en paralelo
    successful = 0
    failed = 0

    executor = ProcessPoolExecutor(max_workers=MAX_PARALLEL_RENDERS)
    future_to_folder = {
        executor.submit(render_video, folder_path, folder_name): folder_name
        for folder_path, folder_name in folders
    }

    cancelled = False
    try:
        # Procesar conforme terminan
        for future in as_completed(future_to_folder):
            folder_name = future_to_folder[future]
            try:
                success = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                    if staging_ctx:
                        restore_staged_folder(
                            staging_ctx["fast_audio"],
                            staging_ctx["slow_audio"],
                            folder_name
                        )
            except Exception as e:
                print(f"[EXCEPCIÓN] Error en {folder_name}: {e}")
                failed += 1
                if staging_ctx:
                    restore_staged_folder(
                        staging_ctx["fast_audio"],
                        staging_ctx["slow_audio"],
                        folder_name
                    )
    except KeyboardInterrupt:
        cancelled = True
        print("\n\nRenderizado cancelado por el usuario.")
        for future in future_to_folder:
            future.cancel()
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        # Terminar procesos para liberar VRAM
        try:
            for proc in getattr(executor, "_processes", {}).values():
                if proc.is_alive():
                    proc.terminate()
            for proc in getattr(executor, "_processes", {}).values():
                proc.join(timeout=5)
        except Exception:
            pass
    finally:
        if not cancelled:
            try:
                executor.shutdown(wait=True)
            except Exception:
                pass
        else:
            cleanup_temp()
            if staging_ctx:
                for name in staging_ctx.get("staged_names", set()):
                    restore_staged_folder(
                        staging_ctx["fast_audio"],
                        staging_ctx["slow_audio"],
                        name
                    )

    if cancelled:
        return

    print(f"\n{'='*60}")
    print(f"RENDERIZADO COMPLETADO")
    print(f"Exitosos: {successful}")
    print(f"Fallidos: {failed}")
    print(f"{'='*60}\n")


def process_folders_sequential():
    """
    Procesa carpetas secuencialmente (1 a la vez) con barra de progreso
    """
    # Recoger todas las carpetas
    folders = [
        (MAIN_DIR / folder_name, folder_name)
        for folder_name in os.listdir(MAIN_DIR)
        if (MAIN_DIR / folder_name).is_dir()
    ]

    # Mezclar aleatoriamente
    random.shuffle(folders)

    # Limitar cantidad
    folders = folders[:MAX_FOLDERS_TO_PROCESS]

    print(f"\n{'='*60}")
    print(f"INICIANDO RENDERIZADO SECUENCIAL")
    print(f"Carpetas a procesar: {len(folders)}")
    print(f"{'='*60}")

    # Procesar secuencialmente
    successful = 0
    failed = 0
    total_start_time = time.time()

    for i, (folder_path, folder_name) in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] ", end="")
        success = render_video(folder_path, folder_name, show_progress=True)

        if success:
            successful += 1
        else:
            failed += 1

        # Mostrar estadísticas parciales
        total_elapsed = time.time() - total_start_time
        avg_time = total_elapsed / i
        remaining = len(folders) - i
        eta_total = remaining * avg_time

        print(f"\nProgreso total: {i}/{len(folders)} videos")
        print(f"Exitosos: {successful} | Fallidos: {failed}")
        print(f"Tiempo promedio por video: {format_time(avg_time)}")
        print(f"ETA para completar todos: {format_time(eta_total)}")
        print(f"{'='*60}")

    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"RENDERIZADO COMPLETADO")
    print(f"Tiempo total: {format_time(total_elapsed)}")
    print(f"Exitosos: {successful}")
    print(f"Fallidos: {failed}")
    print(f"{'='*60}\n")


def render_single_video(specific_folder=None):
    """
    Renderiza un solo video con barra de progreso (modo prueba)
    specific_folder: Path a carpeta específica (opcional, si None usa aleatoria)
    """
    if specific_folder:
        # Usar carpeta específica
        folder_path = Path(specific_folder)
        folder_name = folder_path.name
        if not folder_path.exists():
            print(f"ERROR: La carpeta no existe: {folder_path}")
            return
    else:
        # Recoger todas las carpetas
        folders = [
            (MAIN_DIR / folder_name, folder_name)
            for folder_name in os.listdir(MAIN_DIR)
            if (MAIN_DIR / folder_name).is_dir()
        ]

        if not folders:
            print("ERROR: No hay carpetas para procesar")
            return

        # Seleccionar una carpeta aleatoria
        folder_path, folder_name = random.choice(folders)

    print(f"\n{'='*60}")
    print(f"MODO PRUEBA - RENDERIZADO ÚNICO")
    print(f"{'='*60}")

    # Renderizar con barra de progreso
    success = render_video(folder_path, folder_name, show_progress=True)

    print(f"\n{'='*60}")
    if success:
        print(f"PRUEBA COMPLETADA EXITOSAMENTE")
        print(f"Carpeta final: {DIR_UPLOAD} (revisa [UPLOAD] para el nombre final)")
    else:
        print(f"PRUEBA FALLIDA")
    print(f"{'='*60}\n")


# ============================================================================
# MAIN
# ============================================================================

def cleanup_temp():
    """Limpia el directorio temporal del SSD"""
    if USE_SSD_TEMP and SSD_TEMP_DIR.exists():
        for item in SSD_TEMP_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)


if __name__ == "__main__":
    try:
        # Verificar que el intro existe
        if not INTRO_VIDEO.exists():
            print(f"ERROR: No se encuentra el video de intro en {INTRO_VIDEO}")
            exit(1)

        # Verificar que el directorio principal existe
        if not MAIN_DIR.exists():
            print(f"ERROR: No se encuentra el directorio {MAIN_DIR}")
            exit(1)

        # Verificar argumentos
        if len(sys.argv) > 1:
            if sys.argv[1] == "--test" or sys.argv[1] == "-t":
                # Modo prueba: renderizar un solo video con barra de progreso
                if len(sys.argv) > 2:
                    # Carpeta específica proporcionada
                    render_single_video(sys.argv[2])
                else:
                    # Carpeta aleatoria
                    render_single_video()
            elif sys.argv[1] == "--parallel" or sys.argv[1] == "-p":
                # Modo paralelo: renderizar múltiples videos simultáneamente (sin barra de progreso)
                process_folders_parallel()
            elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
                print("\nUso:")
                print(f"  python '{Path(__file__).name}'                        # Renderizado paralelo ({MAX_PARALLEL_RENDERS} videos simultáneos)")
                print(f"  python '{Path(__file__).name}' --test                 # Prueba con 1 video aleatorio + barra de progreso")
                print(f"  python '{Path(__file__).name}' --test /ruta/carpeta   # Prueba con carpeta específica")
                print(f"  python '{Path(__file__).name}' -t                     # Igual que --test")
                print(f"  python '{Path(__file__).name}' --parallel             # Renderizado paralelo ({MAX_PARALLEL_RENDERS} videos simultáneos)")
                print(f"  python '{Path(__file__).name}' -p                     # Igual que --parallel")
                print("")
            else:
                print(f"Argumento desconocido: {sys.argv[1]}")
                print("Usa --help para ver opciones")
        else:
            # Modo normal: renderizado paralelo (con staging opcional)
            folders_override, staging_ctx = prepare_staging_batch()
            process_folders_parallel(folders_override=folders_override, staging_ctx=staging_ctx)

    except KeyboardInterrupt:
        print("\n\nRenderizado cancelado por el usuario.")
        exit(0)
