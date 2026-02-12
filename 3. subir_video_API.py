#!/usr/bin/env python3
import argparse
import ast
import csv
import errno
import json
import os
import random
import re
import shutil
import socket
import ssl
import sys
import time
import unicodedata
import webbrowser
from urllib.parse import quote, unquote
import uuid
from datetime import datetime, timedelta, time as dtime, timezone
from pathlib import Path

import requests
import httplib2
import numpy as np
from PIL import Image
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from mutagen import File as MutagenFile

from config import AUDIO_FORMATS, DIR_UPLOAD, DIR_YA_SUBIDOS, RANDOMIZE_VIDEO_SELECTION
from subir_video.authenticate import authenticate, authenticate_next
from limpieza.censura import censor_profanity


class QuotaExceededError(RuntimeError):
    """Error lanzado cuando se agota la cuota diaria de la YouTube API."""

    pass


class UploadLimitExceededError(RuntimeError):
    """Error lanzado cuando el canal de YouTube alcanza su limite de subidas diarias."""

    pass


def _get_error_reasons(exc: HttpError) -> list[str]:
    """Extrae las razones de error de un HttpError."""
    try:
        content = (
            exc.content.decode("utf-8")
            if isinstance(exc.content, bytes)
            else exc.content
        )
        data = json.loads(content)
        errors = data.get("error", {}).get("errors", [])
        return [e.get("reason", "") for e in errors]
    except (json.JSONDecodeError, AttributeError):
        return []


def is_quota_error(exc: HttpError) -> bool:
    """Detecta si un HttpError es por cuota agotada de la API."""
    if exc.resp is None:
        return False
    if exc.resp.status not in (403, 429):
        return False
    reasons = _get_error_reasons(exc)
    return any(r in {"quotaExceeded", "dailyLimitExceeded"} for r in reasons)


def is_upload_limit_error(exc: HttpError) -> bool:
    """Detecta si un HttpError es por limite de subidas del canal."""
    if exc.resp is None:
        return False
    if exc.resp.status != 400:
        return False
    reasons = _get_error_reasons(exc)
    return "uploadLimitExceeded" in reasons


BASE_URL = "https://deathgrind.club"
API_URL = f"{BASE_URL}/api"
INTRO_SECONDS = 8

DELAY_ENTRE_PAGINAS = 1.0
DELAY_BASE_429 = 30
MAX_RETRIES_ERROR = 5
DEATHGRIND_DISABLED = False
PYCOUNTRY_WARNED = False
BABEL_WARNED = False
DEATHGRIND_API_CACHE = {}
DEATHGRIND_LAST_REQUEST_TS = 0.0
METADATA_RETRY_DELAY = int(os.environ.get("DEATHGRIND_METADATA_RETRY_DELAY", "45"))
METADATA_MAX_WAIT_MINUTES = int(
    os.environ.get("DEATHGRIND_METADATA_MAX_WAIT_MINUTES", "30")
)

DEFAULT_SCHEDULE_HOURS = [8, 12]
DEFAULT_PLAYLIST_PRIVACY = os.environ.get("YOUTUBE_PLAYLIST_PRIVACY", "public")

TIPOS_DISCO = {
    1: "Album",
    2: "EP",
    3: "Demo",
    4: "Single",
    5: "Split",
    6: "Compilation",
    7: "Live",
    8: "Boxset",
    9: "EP",
}

FULL_TIPO_MAP = {
    "album": "Full Album",
    "ep": "Full EP",
    "demo": "Full Demo",
    "single": "Full Single",
    "split": "Full Split",
    "compilation": "Full Compilation",
    "live": "Full Live",
    "boxset": "Full Boxset",
}

STREAM_SERVICES = [
    "bandcamp",
    "spotify",
    "youtube_music",
    "apple_music",
    "deezer",
    "amazon",
]

FOLLOW_SERVICES = [
    "official_site",
    "facebook",
    "instagram",
    "youtube",
    "tiktok",
    "twitter",
    "metal_archives",
    "spirit_of_metal",
    "discogs",
    "vk",
]

SERVICE_LABELS = {
    "bandcamp": "Bandcamp",
    "spotify": "Spotify",
    "youtube_music": "YouTube Music",
    "apple_music": "Apple Music",
    "deezer": "Deezer",
    "amazon": "Amazon Music",
    "official_site": "Official Site",
    "facebook": "Facebook",
    "instagram": "Instagram",
    "youtube": "YouTube",
    "tiktok": "TikTok",
    "twitter": "X/Twitter",
    "metal_archives": "Metal Archives",
    "spirit_of_metal": "Spirit of Metal",
    "discogs": "Discogs",
    "vk": "VK",
}

SERVICE_MATCHERS = {
    "bandcamp": ["bandcamp.com"],
    "spotify": ["spotify.com"],
    "youtube_music": ["music.youtube.com"],
    "apple_music": ["music.apple.com", "itunes.apple.com"],
    "deezer": ["deezer.com"],
    "amazon": ["music.amazon.", "amazon.com/music", "music.amazon.com"],
    "facebook": ["facebook.com"],
    "instagram": ["instagram.com"],
    "youtube": ["youtube.com", "youtu.be"],
    "tiktok": ["tiktok.com"],
    "twitter": ["twitter.com", "x.com"],
    "metal_archives": ["metal-archives.com"],
    "spirit_of_metal": ["spirit-of-metal.com"],
    "discogs": ["discogs.com"],
    "vk": ["vk.com"],
}

PLAYLIST_LINKS_CACHE_PATH = Path(
    os.environ.get(
        "YOUTUBE_PLAYLIST_LINKS_CACHE",
        str(Path(__file__).resolve().parent / "data" / "playlist_links_cache.json"),
    )
)


def cargar_env(env_path=".env"):
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                os.environ[key] = val


def crear_sesion_autenticada():
    email = os.environ.get("DEATHGRIND_EMAIL")
    password = os.environ.get("DEATHGRIND_PASSWORD")
    if not email or not password:
        print("No hay credenciales de DeathGrind en .env, se omite la API.")
        return None

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }
    )

    session.get(f"{BASE_URL}/auth/sign-in", timeout=30)
    csrf_token = session.cookies.get("csrfToken", "")
    uuid_header = uuid.uuid4().hex
    headers = {"x-csrf-token": csrf_token, "x-uuid": uuid_header}
    response = session.post(
        f"{API_URL}/auth/login",
        json={"login": email, "password": password},
        headers=headers,
        timeout=30,
    )

    if response.status_code not in (200, 202):
        raise ConnectionError(f"Error de login DeathGrind: {response.status_code}")

    csrf_token = session.cookies.get("csrfToken", "")
    session.headers.update({"x-csrf-token": csrf_token, "x-uuid": uuid_header})
    return session


def deathgrind_block_on_429():
    raw = os.environ.get("DEATHGRIND_BLOCK_ON_429", "1").strip().lower()
    return raw not in {"0", "false", "no"}


def deathgrind_max_retries_429():
    raw = os.environ.get("DEATHGRIND_MAX_429", "3").strip()
    try:
        value = int(raw)
    except ValueError:
        return 3
    return value if value > 0 else None


def deathgrind_search_page_limit():
    raw = os.environ.get("DEATHGRIND_SEARCH_PAGES", "3").strip()
    try:
        value = int(raw)
    except ValueError:
        return 3
    return value if value > 0 else None


def deathgrind_genre_page_limit():
    raw = os.environ.get("DEATHGRIND_GENRE_PAGES", "2").strip()
    try:
        value = int(raw)
    except ValueError:
        return 2
    return value if value > 0 else None


def deathgrind_min_interval():
    raw = os.environ.get("DEATHGRIND_MIN_INTERVAL", "0").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, value)


def deathgrind_cache_enabled():
    raw = os.environ.get("DEATHGRIND_CACHE", "1").strip().lower()
    return raw not in {"0", "false", "no"}


def deathgrind_cache_limit():
    raw = os.environ.get("DEATHGRIND_CACHE_MAX", "2000").strip()
    try:
        value = int(raw)
    except ValueError:
        return 2000
    return value if value > 0 else None


def make_deathgrind_cache_key(endpoint, params):
    if not params:
        return (endpoint, ())
    items = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, (list, tuple)):
            value = tuple(str(v) for v in value)
        else:
            value = str(value)
        items.append((key, value))
    return (endpoint, tuple(items))


def api_get(session, endpoint, params=None, max_retries=MAX_RETRIES_ERROR):
    global DEATHGRIND_DISABLED, DEATHGRIND_LAST_REQUEST_TS
    if DEATHGRIND_DISABLED or session is None:
        return None
    url = f"{API_URL}{endpoint}"
    retries_429 = 0
    retries_error = 0
    block_on_429 = deathgrind_block_on_429()
    max_retries_429 = deathgrind_max_retries_429()
    min_interval = deathgrind_min_interval()
    cache_key = None
    if deathgrind_cache_enabled():
        cache_key = make_deathgrind_cache_key(endpoint, params)
        if cache_key in DEATHGRIND_API_CACHE:
            return DEATHGRIND_API_CACHE[cache_key]

    while True:
        try:
            if min_interval > 0:
                elapsed = time.time() - DEATHGRIND_LAST_REQUEST_TS
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            DEATHGRIND_LAST_REQUEST_TS = time.time()
            response = session.get(url, params=params, timeout=30)
            if response.status_code == 429:
                retries_429 += 1
                wait_time = DELAY_BASE_429 * retries_429
                if max_retries_429 is not None and retries_429 >= max_retries_429:
                    DEATHGRIND_DISABLED = True
                    print(
                        "Rate limit DeathGrind, se desactiva la API en esta ejecucion."
                    )
                    return None
                if not block_on_429:
                    print(
                        "Rate limit DeathGrind, se omite la llamada por esta ejecucion."
                    )
                    return None
                print(f"Rate limit DeathGrind, esperando {wait_time}s...")
                time.sleep(wait_time)
                continue
            if response.status_code != 200:
                retries_error += 1
                if retries_error >= max_retries:
                    return None
                time.sleep(5)
                continue
            data = response.json()
            if cache_key is not None:
                cache_limit = deathgrind_cache_limit()
                if cache_limit is not None and len(DEATHGRIND_API_CACHE) >= cache_limit:
                    DEATHGRIND_API_CACHE.pop(next(iter(DEATHGRIND_API_CACHE)))
                DEATHGRIND_API_CACHE[cache_key] = data
            return data
        except requests.RequestException:
            retries_error += 1
            if retries_error >= max_retries:
                return None
            time.sleep(5)


def cargar_generos(archivo="generos_activos.txt"):
    env_path = os.environ.get("DEATHGRIND_GENEROS_FILE")
    if env_path:
        archivo = env_path
    generos = []
    generos_lookup = {}
    if not os.path.exists(archivo):
        return generos, generos_lookup
    with open(archivo, "r", encoding="utf-8") as handle:
        lines = handle.readlines()[1:]
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                try:
                    genre_id = int(parts[0])
                except ValueError:
                    continue
                name = parts[2].strip()
                generos.append((genre_id, name))
                generos_lookup[genre_id] = name
    return generos, generos_lookup


def cargar_repertorio(archivo=Path("data/repertorio.json")):
    env_path = os.environ.get("DEATHGRIND_REPERTORIO_FILE")
    if env_path:
        archivo = Path(env_path)
    if not archivo.exists():
        return []
    try:
        with open(archivo, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return []


def deathgrind_smoke_test(session, generos):
    if not session:
        return False
    if not generos:
        print("No se encontro generos_activos.txt, se omite prueba DeathGrind.")
        return True

    genre_id, genre_name = generos[0]
    data = api_get(session, "/posts/filter", params={"genres": genre_id})
    if not data:
        print("Prueba DeathGrind fallo al consultar posts.")
        return False
    print(f"Prueba DeathGrind /posts/filter?genres={genre_id} ({genre_name}) ok.")
    posts = data.get("posts") or []
    print(f"Prueba DeathGrind posts: {len(posts)} | hasMore: {data.get('hasMore')}")
    if not posts:
        return False

    first_post = posts[0]
    post_id = first_post.get("postId")
    band_id = None
    bands = first_post.get("bands") or []
    if bands and isinstance(bands[0], dict):
        band_id = bands[0].get("bandId")

    if band_id:
        discography = api_get(session, f"/bands/{band_id}/discography")
        if discography is None:
            print(f"Prueba DeathGrind /bands/{band_id}/discography fallo.")
            return False
        disc_posts = discography.get("posts") or []
        print(f"Prueba DeathGrind discography posts: {len(disc_posts)}")
    else:
        print("Prueba DeathGrind: no se encontro bandId en el primer post.")

    if post_id:
        links_data = api_get(session, f"/posts/{post_id}/links")
        if links_data is None:
            print(f"Prueba DeathGrind /posts/{post_id}/links fallo.")
            return False
        links = links_data.get("links") if isinstance(links_data, dict) else links_data
        links = links or []
        print(f"Prueba DeathGrind links: {len(links)}")
    else:
        print("Prueba DeathGrind: no se encontro postId en el primer post.")

    return True


def normalize_name(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def limpiar_genero_texto(value):
    if value is None:
        return None
    text = str(value).strip()
    text = re.sub(r"^\(\d+\)", "", text).strip()
    text = re.sub(r"^\d+\s*", "", text).strip()
    return text if text else None


def split_genres(value):
    if not value:
        return []
    if isinstance(value, list):
        parts = value
    else:
        parts = re.split(r"[,/;]+", str(value))
    cleaned = []
    seen = set()
    for part in parts:
        item = limpiar_genero_texto(part)
        if not item:
            continue
        key = item.lower()
        if key == "unknown":
            continue
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


def format_genre_text(value):
    parts = split_genres(value)
    return "/".join(parts) if parts else ""


def normalize_lookup_text(value):
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", text)


def tokenize_lookup_text(value):
    text = normalize_lookup_text(value)
    if not text:
        return []
    return [token for token in text.split() if len(token) > 2]


def country_flag_emoji(country_code):
    if not country_code or len(country_code) != 2:
        return ""
    code = country_code.upper()
    if not code.isalpha():
        return ""
    return "".join(chr(127397 + ord(char)) for char in code)


def resolve_country_meta(country_name):
    code = None
    normalized_name = country_name
    if not country_name:
        return {
            "name": None,
            "code": None,
            "flag": "",
        }

    text = str(country_name).strip()
    if len(text) == 2 and text.isalpha():
        code = text.upper()
    else:
        try:
            import pycountry

            match = pycountry.countries.lookup(text)
            if match and getattr(match, "alpha_2", None):
                code = match.alpha_2
                normalized_name = match.name
        except Exception:
            pass

    return {
        "name": normalized_name,
        "code": code,
        "flag": country_flag_emoji(code),
    }


def load_playlist_links_cache():
    path = PLAYLIST_LINKS_CACHE_PATH
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        items = payload.get("items") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            return []
        normalized = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue
            normalized.append(
                {
                    "title": title,
                    "url": url,
                    "title_key": normalize_lookup_text(item.get("title_key") or title),
                }
            )
        return normalized
    except Exception:
        return []


def find_best_playlist_link(playlist_cache, query_texts, required_texts=None):
    if not playlist_cache:
        return None

    query_tokens = []
    for query in query_texts or []:
        query_tokens.extend(tokenize_lookup_text(query))
    if not query_tokens:
        return None
    required_tokens = []
    for text in required_texts or []:
        required_tokens.extend(tokenize_lookup_text(text))

    best_item = None
    best_score = -1
    for item in playlist_cache:
        title_key = item.get("title_key") or ""
        if not title_key:
            continue
        if required_tokens and not all(token in title_key for token in required_tokens):
            continue
        score = 0
        for token in query_tokens:
            if token in title_key:
                score += 3
        for token in required_tokens:
            if token in title_key:
                score += 5
        if title_key == normalize_lookup_text(" ".join(query_texts or [])):
            score += 10
        if score > best_score:
            best_score = score
            best_item = item
    return best_item


def build_dynamic_hashtags(genres, country_name, year_value, tipo_full):
    tags = []
    seen = set()

    def add_tag(raw):
        if not raw:
            return
        slug = normalize_lookup_text(raw).replace(" ", "")
        if not slug or slug in seen:
            return
        seen.add(slug)
        tags.append(f"#{slug}")

    for genre in genres or []:
        add_tag(genre)

    if genres:
        add_tag(genres[0])
        primary_tokens = tokenize_lookup_text(genres[0])
        for token in primary_tokens[:4]:
            add_tag(token)

    if country_name:
        add_tag(f"{country_name} metal")
    if year_value:
        add_tag(year_value)
    if tipo_full:
        add_tag(tipo_full)
    add_tag("underground metal")
    add_tag("extreme metal")

    return tags[:14]


def strip_album_suffix(value):
    cleaned = str(value)
    patterns = [
        re.compile(r"\s*[\[(]\s*\d{4}\s*[\])]\s*$"),
        re.compile(
            r"\s*[\[(]\s*(ep|album|demo|single|split|compilation|live|boxset)\s*[\])]\s*$",
            re.IGNORECASE,
        ),
    ]
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            new = pattern.sub("", cleaned).strip()
            if new != cleaned:
                cleaned = new
                changed = True
    return cleaned.strip()


def normalize_album_name(value):
    return normalize_name(strip_album_suffix(value))


def album_names_match(name1, name2):
    """Compara nombres de album con tolerancia a truncamiento."""
    if not name1 or not name2:
        return False
    norm1 = normalize_album_name(name1)
    norm2 = normalize_album_name(name2)
    if not norm1 or not norm2:
        return False
    # Match exacto
    if norm1 == norm2:
        return True
    # Match parcial: uno contiene al otro (para nombres truncados)
    if norm1.startswith(norm2) or norm2.startswith(norm1):
        return True
    return False


def elegir_tipo(type_value):
    if isinstance(type_value, list) and type_value:
        first = type_value[0]
        if isinstance(first, int):
            return TIPOS_DISCO.get(first)
        return str(first)
    if isinstance(type_value, int):
        return TIPOS_DISCO.get(type_value)
    if isinstance(type_value, str):
        return type_value
    return None


def formatear_tipo_full(tipo):
    if not tipo:
        return "Full Album"
    key = tipo.lower().strip()
    return FULL_TIPO_MAP.get(key, f"Full {tipo}")


def extraer_genero(post, generos_lookup):
    raw = post.get("genres") or post.get("genre") or post.get("genre_ids")
    if not raw:
        return None
    generos = []
    if isinstance(raw, dict):
        raw = [raw]
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                raw = list(parsed)
        except Exception:
            raw = [int(val) for val in re.findall(r"\d+", raw)] or [raw]
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                genre_id = item.get("id") or item.get("genreId")
                name = item.get("name") or item.get("genre")
                if not name and genre_id in generos_lookup:
                    name = generos_lookup.get(genre_id)
                if name:
                    cleaned = limpiar_genero_texto(name)
                    if cleaned:
                        generos.append(cleaned)
                elif genre_id and genre_id in generos_lookup:
                    generos.append(generos_lookup.get(genre_id))
            else:
                if isinstance(item, int) and item in generos_lookup:
                    generos.append(generos_lookup[item])
                else:
                    cleaned = limpiar_genero_texto(item)
                    if cleaned and not cleaned.isdigit():
                        generos.append(cleaned)
    elif isinstance(raw, str):
        cleaned = limpiar_genero_texto(raw)
        if cleaned:
            generos.append(cleaned)
    if generos:
        return ", ".join(generos)

    # Fallback: algunos posts traen el genero en bands[].genre
    bands = post.get("bands") or post.get("band")
    if isinstance(bands, dict):
        bands = [bands]
    if isinstance(bands, list):
        fallback = []
        seen = set()
        for band in bands:
            if not isinstance(band, dict):
                continue
            genre_value = band.get("genre") or band.get("genres")
            for item in split_genres(genre_value):
                key = item.lower().strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                fallback.append(item)
        if fallback:
            return ", ".join(fallback)

    return None


def extraer_anio(post):
    release_date = post.get("releaseDate") or post.get("year")
    if isinstance(release_date, list) and release_date:
        return release_date[0]
    if isinstance(release_date, (str, int)):
        return release_date
    return None


def extraer_anio_de_texto(texto):
    if not texto:
        return None
    match = re.search(r"(19|20)\d{2}", str(texto))
    return match.group(0) if match else None


def ordenar_carpetas_por_anio(folders, shuffle_within_year=True):
    if not folders:
        return folders

    buckets = {}
    for folder in folders:
        year = extraer_anio_de_texto(folder.name)
        if year is not None:
            try:
                year = int(year)
            except ValueError:
                year = None
        buckets.setdefault(year, []).append(folder)

    years = sorted([y for y in buckets.keys() if y is not None], reverse=True)
    if None in buckets:
        years.append(None)

    ordered = []
    for year in years:
        items = buckets[year]
        if shuffle_within_year:
            random.shuffle(items)
        else:
            items.sort(key=lambda item: item.name.lower())
        ordered.extend(items)
    return ordered


def ordenar_carpetas_para_subida(folders):
    if not folders:
        return folders

    if RANDOMIZE_VIDEO_SELECTION:
        randomized = list(folders)
        random.shuffle(randomized)
        return randomized

    return ordenar_carpetas_por_anio(folders, shuffle_within_year=False)


def unwrap_deathgrind_payload(data):
    if isinstance(data, dict):
        for key in ("post", "band"):
            value = data.get(key)
            if isinstance(value, dict):
                return value
    return data


def extraer_pais_valor(value):
    if value is None:
        return None
    if isinstance(value, dict):
        for key in (
            "name",
            "title",
            "label",
            "value",
            "country",
            "countryName",
            "country_name",
        ):
            parsed = extraer_pais_valor(value.get(key))
            if parsed:
                return parsed
        return None
    if isinstance(value, list):
        for item in value:
            parsed = extraer_pais_valor(item)
            if parsed:
                return parsed
        return None
    text = str(value).strip()
    return text if text else None


def inferir_pais_desde_location(location_text):
    if not location_text:
        return None
    text = str(location_text).strip()
    if not text:
        return None

    # Ejemplos: "Bogota, Colombia", "Madrid, Spain"
    if "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if parts:
            return parts[-1]

    # Fallback simple si viene con separadores alternos
    for sep in (" / ", " - ", " | "):
        if sep in text:
            parts = [part.strip() for part in text.split(sep) if part.strip()]
            if parts:
                return parts[-1]

    return text


def extraer_pais_desde_data(data, band_id=None, band_name=None):
    data = unwrap_deathgrind_payload(data)
    if not isinstance(data, dict):
        return None
    for key in ("country", "pais", "origin", "countryName", "country_name"):
        parsed = extraer_pais_valor(data.get(key))
        if parsed:
            return parsed

    # Location suele venir como "Ciudad, Pais"
    location = extraer_pais_valor(data.get("location"))
    if location:
        inferred = inferir_pais_desde_location(location)
        if inferred:
            return inferred

    bands = data.get("bands") or data.get("band")
    if isinstance(bands, dict):
        return extraer_pais_desde_data(bands, band_id=band_id, band_name=band_name)
    if isinstance(bands, list):
        for band in bands:
            if not isinstance(band, dict):
                continue
            if (
                band_id
                and band.get("bandId")
                and str(band.get("bandId")) != str(band_id)
            ):
                continue
            if (
                band_name
                and band.get("name")
                and normalize_name(band.get("name")) != normalize_name(band_name)
            ):
                continue
            parsed = extraer_pais_desde_data(band)
            if parsed:
                return parsed
    return None


def normalizar_pais(pais):
    if not pais:
        return None
    text = str(pais).strip()
    if not text:
        return None
    code = None
    lookup_candidates = [text]
    inferred_from_location = inferir_pais_desde_location(text)
    if inferred_from_location and inferred_from_location not in lookup_candidates:
        lookup_candidates.insert(0, inferred_from_location)

    for candidate in lookup_candidates:
        if len(candidate) == 2 and candidate.isalpha():
            code = candidate.upper()
            break
        try:
            import pycountry

            match = pycountry.countries.lookup(candidate)
            if match and getattr(match, "alpha_2", None):
                code = match.alpha_2
                text = match.name
                break
        except Exception:
            global PYCOUNTRY_WARNED
            if not PYCOUNTRY_WARNED:
                print(
                    "Instala pycountry para detectar el codigo del pais (pip install pycountry)."
                )
                PYCOUNTRY_WARNED = True

    if code:
        name_en = None
        name_es = None
        try:
            from babel import Locale

            name_en = Locale.parse("en").territories.get(code)
            name_es = Locale.parse("es").territories.get(code)
        except Exception:
            global BABEL_WARNED
            if not BABEL_WARNED:
                print(
                    "Instala babel para mostrar el pais en ingles y espanol (pip install babel)."
                )
                BABEL_WARNED = True
        if not name_en:
            try:
                import pycountry

                country = pycountry.countries.get(alpha_2=code)
                if country and country.name:
                    name_en = country.name
            except Exception:
                pass
        if name_en and name_es:
            if name_en == name_es:
                return name_en
            return name_en
        return name_en or name_es or code
    return text


def obtener_band_id(post, band_name):
    bands = post.get("bands", [])
    for band in bands:
        if isinstance(band, dict):
            name = band.get("name", "")
            if normalize_name(name) == normalize_name(band_name):
                return band.get("bandId")
        else:
            if normalize_name(str(band)) == normalize_name(band_name):
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
        return album_names_match(post_album, album_name)
    return True


def buscar_release_en_cache(repertorio, band_name, album_name):
    if not repertorio:
        return None
    norm_band = normalize_name(band_name)
    for release in repertorio:
        if normalize_name(release.get("band", "")) != norm_band:
            continue
        if album_name and not album_names_match(release.get("album", ""), album_name):
            continue
        return release
    return None


def buscar_release_en_csv(csv_path, band_name, album_name):
    if not csv_path:
        return None
    path = Path(csv_path)
    if not path.exists():
        return None
    norm_band = normalize_name(band_name)

    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                album = row.get("album") or ""
                title = row.get("title") or ""
                if not album and title and " - " in title:
                    album = title.split(" - ", 1)[1]

                if album_name and not album_names_match(album, album_name):
                    continue

                bands_raw = row.get("bands") or row.get("band") or ""
                try:
                    bands = ast.literal_eval(bands_raw) if bands_raw else []
                except Exception:
                    bands = [bands_raw]
                bands = [band for band in bands if band]
                if bands and not any(
                    normalize_name(band) == norm_band for band in bands
                ):
                    continue

                genre_ids = []
                genre_raw = row.get("genre_ids") or ""
                if genre_raw:
                    try:
                        genre_ids = ast.literal_eval(genre_raw)
                        if isinstance(genre_ids, int):
                            genre_ids = [genre_ids]
                    except Exception:
                        genre_ids = [
                            int(val) for val in re.findall(r"\d+", str(genre_raw))
                        ]
                type_id = row.get("type_id")
                if type_id:
                    try:
                        parsed = ast.literal_eval(type_id)
                        if isinstance(parsed, (list, tuple)):
                            type_id = parsed[0] if parsed else None
                        elif isinstance(parsed, int):
                            type_id = parsed
                    except Exception:
                        match = re.search(r"\d+", str(type_id))
                        type_id = int(match.group(0)) if match else None
                year = row.get("year")
                try:
                    year = int(year) if year else None
                except ValueError:
                    year = None

                country = None
                country_raw = row.get("country")
                if country_raw:
                    try:
                        parsed = ast.literal_eval(country_raw)
                        if isinstance(parsed, (list, tuple)):
                            country = parsed[0] if parsed else None
                        elif isinstance(parsed, str):
                            country = parsed
                        else:
                            country = str(parsed)
                    except Exception:
                        country = str(country_raw).strip() or None

                return {
                    "band": bands[0] if bands else band_name,
                    "album": album or album_name,
                    "post_id": int(row.get("post_id")) if row.get("post_id") else None,
                    "type_id": type_id,
                    "genres": genre_ids,
                    "year": year,
                    "post_url": row.get("post_url"),
                    "country": country,
                }
    except Exception:
        return None
    return None


def buscar_release_en_api(
    session, band_name, album_name, generos, allow_genre_fallback=False
):
    if DEATHGRIND_DISABLED:
        return None
    album_search = strip_album_suffix(album_name)
    query = f"{band_name} {album_search}".strip()
    max_pages = deathgrind_search_page_limit()
    page_count = 0
    data = api_get(session, "/posts/filter", params={"search": query})
    if data:
        page_count += 1
        posts = data.get("posts", [])
        for post in posts:
            if match_post(post, band_name, album_name):
                return post

        offset = data.get("offset")
        seen_offsets = set()
        while data.get("hasMore") and offset is not None:
            if max_pages is not None and page_count >= max_pages:
                break
            if offset in seen_offsets:
                break
            seen_offsets.add(offset)
            data = api_get(
                session, "/posts/filter", params={"search": query, "offset": offset}
            )
            if not data:
                break
            page_count += 1
            posts = data.get("posts", [])
            for post in posts:
                if match_post(post, band_name, album_name):
                    return post
            offset = data.get("offset")

    if not allow_genre_fallback or not generos:
        return None

    max_genre_pages = deathgrind_genre_page_limit()
    for genre_id, _genre_name in generos:
        offset = None
        page_count = 0
        seen_offsets = set()
        while True:
            if max_genre_pages is not None and page_count >= max_genre_pages:
                break
            params = {"genres": genre_id}
            if offset is not None:
                params["offset"] = offset
            data = api_get(session, "/posts/filter", params=params)
            if not data:
                break
            page_count += 1
            posts = data.get("posts", [])
            for post in posts:
                if match_post(post, band_name, album_name):
                    return post
            if not data.get("hasMore"):
                break
            offset = data.get("offset")
            if offset is None or offset in seen_offsets:
                break
            seen_offsets.add(offset)
            time.sleep(DELAY_ENTRE_PAGINAS)
    return None


def esperar_release_en_api(
    session, band_name, album_name, generos, allow_genre_fallback
):
    if DEATHGRIND_DISABLED or not session:
        return None

    waited = 0
    delay = max(10, METADATA_RETRY_DELAY)
    max_wait_seconds = (
        METADATA_MAX_WAIT_MINUTES * 60 if METADATA_MAX_WAIT_MINUTES > 0 else None
    )

    while True:
        post = buscar_release_en_api(
            session,
            band_name,
            album_name,
            generos,
            allow_genre_fallback,
        )
        if post:
            return post

        if max_wait_seconds is not None and waited >= max_wait_seconds:
            print(
                f"No se encontro release en API tras esperar {METADATA_MAX_WAIT_MINUTES} min: "
                f"{band_name} - {album_name}"
            )
            return None

        print(
            f"API sin metadata completa para {band_name} - {album_name}. "
            f"Esperando {delay}s para reintentar..."
        )
        time.sleep(delay)
        waited += delay


def iter_link_items(data):
    if isinstance(data, dict):
        if any(key in data for key in ("url", "link", "href", "value")):
            yield data
            return
    if isinstance(data, list):
        for item in data:
            yield item
        return
    if not isinstance(data, dict):
        return
    for key in (
        "links",
        "stream",
        "download",
        "social",
        "socials",
        "follow",
        "relatedLinks",
        "related_links",
    ):
        value = data.get(key)
        if isinstance(value, (list, dict)):
            for item in iter_link_items(value):
                yield item


def get_link_value(item, key_list):
    for key in key_list:
        value = item.get(key)
        if value:
            return value
    return None


def normalizar_link(item):
    if not isinstance(item, dict):
        return None, None
    url = get_link_value(item, ["url", "link", "href", "value"])
    name = get_link_value(item, ["name", "type", "title", "label", "text"])
    if not url:
        return None, None
    url = str(url).strip()
    url_lower = url.lower()
    service_key = None
    for key, patterns in SERVICE_MATCHERS.items():
        if any(pattern in url_lower for pattern in patterns):
            service_key = key
            break
    if not service_key and name:
        name_lower = name.lower()
        if any(token in name_lower for token in ["official", "website", "site", "web"]):
            service_key = "official_site"
        for key in SERVICE_MATCHERS.keys():
            if key.replace("_", " ") in name_lower:
                # Evitar etiquetas incorrectas (ej: Bandcamp apuntando a Mega).
                if any(
                    pattern in url_lower for pattern in SERVICE_MATCHERS.get(key, [])
                ):
                    service_key = key
                break
    if not service_key and name:
        lowered = str(name).strip().lower()
        if lowered in {"official", "official site", "website", "site", "web"}:
            service_key = "official_site"
    return service_key, url


def extraer_links_desde_data(data):
    links = {"stream": {}, "follow": {}}
    data = unwrap_deathgrind_payload(data)
    if not data:
        return links
    for item in iter_link_items(data):
        service_key, url = normalizar_link(item)
        if not service_key:
            continue
        if service_key in STREAM_SERVICES:
            links["stream"].setdefault(service_key, url)
        elif service_key in FOLLOW_SERVICES:
            links["follow"].setdefault(service_key, url)
    return links


def merge_links(base, new_links):
    for category in ("stream", "follow"):
        base.setdefault(category, {})
        for key, value in new_links.get(category, {}).items():
            if key not in base[category]:
                base[category][key] = value
    return base


def extraer_links_deathgrind(session, post_id):
    if DEATHGRIND_DISABLED:
        return {"stream": {}, "follow": {}}
    if not session or not post_id:
        return {"stream": {}, "follow": {}}
    data = api_get(session, f"/posts/{post_id}/links")
    return extraer_links_desde_data(data)


def first_value(values):
    if not values:
        return None
    if isinstance(values, list):
        return values[0]
    return values


def parse_track_number(value):
    if not value:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"\d+", str(value))
    if match:
        return int(match.group(0))
    return None


def limpiar_titulo(nombre):
    titulo = re.sub(r"^\s*\d{1,3}\s*[-._)\]]*\s*", "", nombre).strip()
    return titulo if titulo else nombre


TITLE_MAX_CHARS = 100
DESCRIPTION_MAX_BYTES = 5000
FULL_ALBUM_MARKER = "full album"

CENSORED_DIR = Path(
    "/run/media/banar/Entretenimiento/01_edicion_automatizada/Censurado"
)


def mostrar_progreso_busqueda(actual, total, ancho=30):
    """Muestra barra de progreso para busqueda de metadata."""
    progreso = actual / total if total > 0 else 0
    lleno = int(round(ancho * max(0.0, min(1.0, progreso))))
    barra = "#" * lleno + "-" * (ancho - lleno)
    sys.stdout.write(f"\rBuscando metadata [{barra}] {actual}/{total}")
    sys.stdout.flush()
    if actual >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def truncar_titulo(title, max_len=TITLE_MAX_CHARS, marker=FULL_ALBUM_MARKER):
    if not title:
        return title
    texto = re.sub(r"\s+", " ", str(title)).strip()
    if len(texto) <= max_len:
        return texto

    marker = (marker or "").lower().strip()
    lower = texto.lower()
    ellipsis = "..."

    if marker and marker in lower:
        idx = lower.rfind(marker)
        suffix = texto[idx:]
        if len(suffix) >= max_len:
            phrase = texto[idx : idx + len(marker)]
            return phrase[:max_len]
        available = max_len - len(suffix)
        if available <= len(ellipsis):
            return suffix[-max_len:]
        prefix = texto[: available - len(ellipsis)].rstrip()
        if not prefix:
            return suffix[-max_len:]
        return f"{prefix}{ellipsis}{suffix}"

    if max_len <= len(ellipsis):
        return texto[:max_len]
    return texto[: max_len - len(ellipsis)].rstrip() + ellipsis


def build_public_title(country_flag, band, album, year_text, genre_display, tipo_caps):
    flag_prefix = f"{country_flag} " if country_flag else ""
    base = f"{flag_prefix}{band} - {album}"
    if year_text:
        base += f" ({year_text})"
    if genre_display:
        base += f" • [{genre_display}]"
    if tipo_caps:
        base += f" ⟨{tipo_caps}⟩"
    return base


def compress_title_to_limit(
    country_flag, band, album, year_text, genre_text, tipo_full
):
    genres = split_genres(genre_text)
    tipo_caps = (tipo_full or "").upper().strip()

    def make_title(active_flag, active_genres, active_tipo):
        return build_public_title(
            active_flag,
            band,
            album,
            year_text,
            "/".join(active_genres) if active_genres else "",
            active_tipo,
        )

    title = make_title(country_flag, genres, tipo_caps)
    if len(title) <= TITLE_MAX_CHARS:
        return title

    # 1) Si hay multiples generos, quitar progresivamente el mas largo.
    reduced_genres = list(genres)
    while len(reduced_genres) > 1:
        longest = max(reduced_genres, key=len)
        reduced_genres.remove(longest)
        title = make_title(country_flag, reduced_genres, tipo_caps)
        if len(title) <= TITLE_MAX_CHARS:
            return title

    # 2) Quitar bandera.
    title = make_title("", reduced_genres, tipo_caps)
    if len(title) <= TITLE_MAX_CHARS:
        return title

    # 3) Acortar tipo.
    short_tipo = tipo_caps.replace("FULL ", "") if tipo_caps else ""
    title = make_title("", reduced_genres, short_tipo)
    if len(title) <= TITLE_MAX_CHARS:
        return title

    # 4) Quitar bloque de tipo.
    title = build_public_title("", band, album, year_text, "/".join(reduced_genres), "")
    if len(title) <= TITLE_MAX_CHARS:
        return title

    return truncar_titulo(title)


def truncar_descripcion(texto, max_bytes=DESCRIPTION_MAX_BYTES, suffix="..."):
    if texto is None:
        return ""
    texto = str(texto).strip()
    raw = texto.encode("utf-8")
    if len(raw) <= max_bytes:
        return texto
    suffix_bytes = suffix.encode("utf-8")
    if len(suffix_bytes) >= max_bytes:
        return raw[:max_bytes].decode("utf-8", errors="ignore")
    limit = max_bytes - len(suffix_bytes)
    truncated = raw[:limit].decode("utf-8", errors="ignore").rstrip()
    return truncated + suffix


def ensure_description_limit(full_description):
    if len(full_description.encode("utf-8")) <= DESCRIPTION_MAX_BYTES:
        return full_description

    trimmed = full_description
    steps = [
        ("💼 LABELS: Promotional Packages Available", "Eliminar bloque LABELS"),
        ("🎸 BANDS: Submit Your Album (FREE)", "Eliminar bloque BANDS"),
        ("╔════════════════════════════════════════════╗", "Eliminar bloque SUBSCRIBE"),
        ("🔥 MORE SLAM", "Eliminar bloque MORE SLAM"),
        ("🔗 FOLLOW", "Eliminar bloque FOLLOW"),
        ("🎧 STREAM & DOWNLOAD", "Eliminar bloque STREAM"),
    ]

    for header, _label in steps:
        trimmed = remove_section_by_header(trimmed, header)
        if len(trimmed.encode("utf-8")) <= DESCRIPTION_MAX_BYTES:
            return trimmed

    return truncar_descripcion(trimmed, max_bytes=DESCRIPTION_MAX_BYTES)


def remove_section_by_header(text, header_prefix):
    lines = text.splitlines()
    divider = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    i = 0
    while i < len(lines):
        if (
            lines[i].strip() == divider
            and i + 2 < len(lines)
            and lines[i + 2].strip() == divider
            and lines[i + 1].strip().startswith(header_prefix)
        ):
            j = i + 3
            while j < len(lines):
                if (
                    lines[j].strip() == divider
                    and j + 2 < len(lines)
                    and lines[j + 2].strip() == divider
                ):
                    break
                j += 1
            del lines[i:j]
            break
        i += 1
    return "\n".join(lines).strip()


def extraer_audio_metadata(audio_path):
    meta = {
        "title": None,
        "artist": None,
        "album": None,
        "genre": None,
        "year": None,
        "track_number": None,
        "duration": None,
    }

    audio_easy = MutagenFile(audio_path, easy=True)
    if audio_easy:
        meta["duration"] = getattr(audio_easy.info, "length", None)
        tags = audio_easy.tags or {}
        meta["title"] = first_value(tags.get("title"))
        meta["artist"] = first_value(tags.get("artist"))
        meta["album"] = first_value(tags.get("album"))
        meta["genre"] = first_value(tags.get("genre"))
        meta["track_number"] = parse_track_number(
            first_value(tags.get("tracknumber") or tags.get("track"))
        )
        date_value = first_value(
            tags.get("date") or tags.get("year") or tags.get("originaldate")
        )
        if date_value:
            match = re.search(r"\d{4}", str(date_value))
            if match:
                meta["year"] = match.group(0)

    if meta["duration"] is None:
        audio = MutagenFile(audio_path)
        if audio:
            meta["duration"] = getattr(audio.info, "length", None)
    if meta["duration"] is None:
        try:
            from pydub import AudioSegment

            meta["duration"] = AudioSegment.from_file(audio_path).duration_seconds
        except Exception:
            meta["duration"] = 0

    return meta


def collect_audio_tracks(folder_path):
    audio_exts = {ext.lower() for ext in AUDIO_FORMATS}
    tracks = []
    context = {"band": None, "album": None, "genre": None, "year": None}
    folder_name = folder_path.name.lower()

    for audio_path in folder_path.iterdir():
        if not audio_path.is_file():
            continue
        if audio_path.suffix.lower() not in audio_exts:
            continue
        if audio_path.stem.lower() == folder_name:
            continue
        meta = extraer_audio_metadata(str(audio_path))
        title = meta["title"] or limpiar_titulo(audio_path.stem)
        track_number = meta["track_number"]
        duration = int(meta["duration"] or 0)
        tracks.append(
            {
                "title": title,
                "track_number": track_number,
                "duration": duration,
                "filename": audio_path.name,
            }
        )
        if not context["band"] and meta["artist"]:
            context["band"] = meta["artist"]
        if not context["album"] and meta["album"]:
            context["album"] = meta["album"]
        if not context["genre"] and meta["genre"]:
            context["genre"] = meta["genre"]
        if not context["year"] and meta["year"]:
            context["year"] = meta["year"]

    if tracks and all(track["track_number"] is not None for track in tracks):
        tracks.sort(key=lambda item: item["track_number"])
    else:
        tracks.sort(key=lambda item: item["filename"].lower())

    return tracks, context


def format_time(seconds):
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


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


def parse_track_title_item(item):
    if isinstance(item, dict):
        for key in ("title", "name", "track", "song", "trackTitle", "track_name"):
            value = item.get(key)
            if value:
                return limpiar_titulo(str(value))
        return None
    if isinstance(item, str):
        return limpiar_titulo(item)
    return None


def parse_track_number_item(item):
    if isinstance(item, dict):
        for key in (
            "trackNumber",
            "track_number",
            "track",
            "position",
            "index",
            "number",
        ):
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
            title = limpiar_titulo(line)
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
        entries.sort(
            key=lambda item: (item[0] if item[0] is not None else 9999, item[1])
        )
    return [title for _track_number, _idx, title in entries]


def obtener_tracklist_api(session, post, post_id):
    if DEATHGRIND_DISABLED:
        return []
    raw = find_tracklist_in_data(post) if post else None
    titles = parse_tracklist_raw(raw)
    if titles:
        return titles
    if session and post_id:
        data = api_get(session, f"/posts/{post_id}")
        if data:
            raw = find_tracklist_in_data(data)
            titles = parse_tracklist_raw(raw)
            if titles:
                return titles
    return []


def build_tracklist(tracks, api_titles=None):
    lines = ["0 - Intro (00:00)"]
    total_duration = 0
    for index, track in enumerate(tracks, 1):
        if index == 1:
            total_duration += INTRO_SECONDS
        title = track["title"]
        if api_titles and index - 1 < len(api_titles):
            title = api_titles[index - 1]
        title = censor_profanity(title)
        lines.append(f"{index} - {title} ({format_time(total_duration)})")
        total_duration += track["duration"]
    return lines


def parse_rfc3339(value):
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def normalize_slot(dt):
    return dt.replace(second=0, microsecond=0)


def format_rfc3339(dt):
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def format_local_datetime(dt):
    if not dt:
        return "N/A"
    return dt.strftime("%Y-%m-%d %H:%M")


def parse_user_date(value):
    if not value:
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def max_videos_por_dia(gap_hours):
    if gap_hours <= 0:
        return 0
    return max(1, int(24 // gap_hours))


def slots_respect_gap(slots, taken_slots, gap_minutes):
    if gap_minutes <= 0 or not slots or not taken_slots:
        return True
    gap_seconds = gap_minutes * 60
    for slot in slots:
        for existing in taken_slots:
            if abs((slot - existing).total_seconds()) < gap_seconds:
                return False
    return True


def build_batch_schedule_for_date(date_value, count, gap_hours, taken_slots, tz_local):
    if count <= 0:
        return []
    gap_minutes = max(1, int(round(gap_hours * 60)))
    max_span = gap_minutes * (count - 1)
    if max_span >= 24 * 60:
        return []

    max_offset = 24 * 60 - max_span - 1
    day_start = datetime.combine(date_value, dtime(0, 0), tzinfo=tz_local)
    attempts = min(60, max_offset + 1)

    for _ in range(attempts):
        offset = random.randint(0, max_offset)
        slots = [
            normalize_slot(day_start + timedelta(minutes=offset + gap_minutes * idx))
            for idx in range(count)
        ]
        if not slots_respect_gap(slots, taken_slots, gap_minutes):
            continue
        return slots

    for offset in range(max_offset + 1):
        slots = [
            normalize_slot(day_start + timedelta(minutes=offset + gap_minutes * idx))
            for idx in range(count)
        ]
        if not slots_respect_gap(slots, taken_slots, gap_minutes):
            continue
        return slots

    return []


def find_next_batch_schedule(
    start_date, count, gap_hours, taken_slots, counts_by_date, tz_local, max_days=365
):
    max_per_day = max_videos_por_dia(gap_hours)
    current_date = start_date
    for _ in range(max_days):
        existing = counts_by_date.get(current_date, 0)
        if existing + count > max_per_day:
            current_date += timedelta(days=1)
            continue
        slots = build_batch_schedule_for_date(
            current_date,
            count,
            gap_hours,
            taken_slots,
            tz_local,
        )
        if slots and len(slots) == count:
            return current_date, slots
        current_date += timedelta(days=1)
    return None, []


def parse_schedule_hours():
    raw = os.environ.get("YOUTUBE_SCHEDULE_HOURS")
    if raw:
        hours = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                hour = int(part)
            except ValueError:
                continue
            if 0 <= hour <= 23:
                hours.append(hour)
        if hours:
            return sorted(set(hours))
    return DEFAULT_SCHEDULE_HOURS


def get_uploads_playlist_id(youtube):
    response = youtube.channels().list(part="contentDetails", mine=True).execute()
    items = response.get("items", [])
    if not items:
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def fetch_scheduled_publish_times(youtube, max_items=200):
    playlist_id = get_uploads_playlist_id(youtube)
    if not playlist_id:
        return set(), {}
    tz_local = datetime.now().astimezone().tzinfo
    scheduled = set()
    counts_by_date = {}
    fetched = 0
    page_token = None

    while True:
        resp = (
            youtube.playlistItems()
            .list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=page_token,
            )
            .execute()
        )
        items = resp.get("items", [])
        if not items:
            break

        video_ids = []
        for item in items:
            resource = item.get("snippet", {}).get("resourceId", {})
            video_id = resource.get("videoId")
            if video_id:
                video_ids.append(video_id)

        for i in range(0, len(video_ids), 50):
            chunk = video_ids[i : i + 50]
            vids = (
                youtube.videos()
                .list(
                    part="status",
                    id=",".join(chunk),
                )
                .execute()
            )
            for video in vids.get("items", []):
                status = video.get("status", {})
                publish_at = status.get("publishAt")
                if not publish_at:
                    continue
                dt = parse_rfc3339(publish_at)
                if not dt:
                    continue
                local_dt = dt.astimezone(tz_local)
                if local_dt < datetime.now(tz_local):
                    continue
                slot = normalize_slot(local_dt)
                scheduled.add(slot)
                date_key = slot.date()
                counts_by_date[date_key] = counts_by_date.get(date_key, 0) + 1

        fetched += len(items)
        page_token = resp.get("nextPageToken")
        if max_items is not None and fetched >= max_items:
            if page_token:
                print(
                    "Aviso: se alcanzo el limite de escaneo de programados. Podrian faltar fechas."
                )
            break
        if not page_token:
            break

    return scheduled, counts_by_date


def fetch_scheduled_publish_times_safe(youtube, max_items=200, retries=3):
    for attempt in range(1, retries + 1):
        try:
            return fetch_scheduled_publish_times(youtube, max_items=max_items)
        except (HttpError, OSError) as exc:
            if attempt >= retries:
                print(
                    f"No se pudo consultar videos programados: {exc}. Se continua sin evitar choques."
                )
                return set(), {}
            wait_time = min(30, 2**attempt)
            print(
                f"Error consultando programados ({exc}). Reintentando en {wait_time}s..."
            )
            time.sleep(wait_time)


def resolve_schedule_scan_limit(max_per_day, max_days):
    raw = os.environ.get("YOUTUBE_SCHEDULE_SCAN")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value <= 0:
            return None
        return value
    if max_per_day <= 0 or max_days <= 0:
        return None
    return max(200, max_per_day * max_days)


def prompt_confirm_schedule_date(schedule_date, latest_slot):
    if not sys.stdin.isatty():
        return schedule_date, False
    if latest_slot:
        print(f"Ultima programacion detectada: {format_local_datetime(latest_slot)}")
    else:
        print("No se detectaron videos programados para comparar.")
    prompt = (
        f"Fecha propuesta para el lote: {schedule_date.isoformat()} "
        "¿Es correcta? [S/n] o escribe una fecha (YYYY-MM-DD): "
    )
    while True:
        resp = input(prompt).strip()
        if not resp or resp.lower() in ("s", "si", "sí", "y", "yes"):
            return schedule_date, False
        if resp.lower() in ("n", "no"):
            resp = input("Indica fecha (YYYY-MM-DD): ").strip()
            if not resp:
                print("Fecha no ingresada. Se mantiene la fecha propuesta.")
                return schedule_date, False
        date_override = parse_user_date(resp)
        if date_override:
            return date_override, True
        print("Formato invalido. Usa YYYY-MM-DD (ej: 2026-01-31).")


def build_daily_slot(date_value, hour, tz_local, taken_slots):
    minute_choices = [0, 15, 30, 45]
    for _ in range(60):
        minute = random.choice(minute_choices)
        candidate = datetime.combine(date_value, dtime(hour, minute), tzinfo=tz_local)
        slot = normalize_slot(candidate)
        if slot not in taken_slots:
            return slot
    for minute in minute_choices:
        candidate = datetime.combine(date_value, dtime(hour, minute), tzinfo=tz_local)
        slot = normalize_slot(candidate)
        if slot not in taken_slots:
            return slot
    return normalize_slot(datetime.combine(date_value, dtime(hour, 0), tzinfo=tz_local))


def find_next_publish_slot(start_dt, taken_slots, counts_by_date, hours):
    tz_local = start_dt.tzinfo
    min_start = start_dt + timedelta(minutes=5)
    max_per_day = min(len(hours), 2)
    current_date = start_dt.date()

    while True:
        if counts_by_date.get(current_date, 0) >= max_per_day:
            current_date += timedelta(days=1)
            continue

        day_slots = []
        for hour in hours:
            day_slots.append(
                build_daily_slot(current_date, hour, tz_local, taken_slots)
            )
        day_slots.sort()

        for slot in day_slots:
            if slot < min_start:
                continue
            if slot in taken_slots:
                continue
            if counts_by_date.get(current_date, 0) >= max_per_day:
                break
            taken_slots.add(slot)
            counts_by_date[current_date] = counts_by_date.get(current_date, 0) + 1
            return slot

        current_date += timedelta(days=1)


def clean_url(url):
    if not url:
        return None
    url = unquote(url)
    url = re.sub(r"%2F", "/", url)
    url = re.sub(r"%3A", ":", url)
    if "spotify.com" in url:
        if "/album/" in url or "/artist/" in url:
            return url.split("?")[0].split("#")[0]
    if "youtube.com" in url:
        if "watch?v=" in url:
            video_id = url.split("watch?v=")[1].split("&")[0]
            return f"https://www.youtube.com/watch?v={video_id}"
        if "/channel/" in url or "/c/" in url or "/user/" in url:
            return url.split("?")[0].split("#")[0]
    return url.split("?")[0].split("#")[0]


def extraer_links_band_api(session, band_id, band_data=None):
    links = {"stream": {}, "follow": {}}
    if DEATHGRIND_DISABLED:
        return links
    if not session or not band_id:
        return links

    if band_data:
        band_data = unwrap_deathgrind_payload(band_data)
        links = merge_links(links, extraer_links_desde_data(band_data))
    if not band_data:
        data = api_get(session, f"/bands/{band_id}")
        if data:
            links = merge_links(links, extraer_links_desde_data(data))

    data = api_get(session, f"/bands/{band_id}/links")
    if data:
        links = merge_links(links, extraer_links_desde_data(data))

    tiene_links = any(links["stream"].values()) or any(links["follow"].values())
    if not tiene_links:
        data = api_get(session, f"/bands/{band_id}/discography")
        if data:
            links = merge_links(links, extraer_links_desde_data(data))
    return links


def extraer_links_post_html(session, post_id):
    links = {"stream": {}, "follow": {}}
    if DEATHGRIND_DISABLED:
        return links
    if not session or not post_id:
        return links
    try:
        resp = session.get(f"{BASE_URL}/posts/{post_id}", timeout=30)
    except requests.RequestException:
        return links
    if resp.status_code != 200:
        return links
    html = resp.text or ""
    for href in re.findall(r'href=["\']([^"\']+)["\']', html):
        if not href.startswith("http"):
            continue
        cleaned = clean_url(href)
        service_key, url = normalizar_link({"url": cleaned})
        if not service_key or not url:
            continue
        if service_key in STREAM_SERVICES:
            links["stream"].setdefault(service_key, url)
        elif service_key in FOLLOW_SERVICES:
            links["follow"].setdefault(service_key, url)
    return links


def extraer_links_band_html(session, band_id):
    links = {"stream": {}, "follow": {}}
    if DEATHGRIND_DISABLED:
        return links
    if not session or not band_id:
        return links
    try:
        resp = session.get(f"{BASE_URL}/bands/{band_id}", timeout=30)
    except requests.RequestException:
        return links
    if resp.status_code != 200:
        return links
    html = resp.text or ""
    for href in re.findall(r'href=["\']([^"\']+)["\']', html):
        if not href.startswith("http"):
            continue
        cleaned = clean_url(href)
        service_key, url = normalizar_link({"url": cleaned})
        if not service_key or not url:
            continue
        if service_key in STREAM_SERVICES:
            links["stream"].setdefault(service_key, url)
        elif service_key in FOLLOW_SERVICES:
            links["follow"].setdefault(service_key, url)
    return links


def parse_band_album_from_folder(folder_name):
    if " - " in folder_name:
        band, album = folder_name.split(" - ", 1)
        return band.strip(), album.strip()
    return folder_name, folder_name


def get_repertorio_csv_path():
    csv_path = os.environ.get("DEATHGRIND_REPERTORIO_CSV")
    if csv_path:
        return csv_path
    default_csv = Path(
        "/home/banar/Desktop/scrapper-deathgrind/data/bandas_completo.csv"
    )
    if default_csv.exists():
        return str(default_csv)
    return None


def build_preview_title(folder_path, repertorio):
    band, album = parse_band_album_from_folder(folder_path.name)
    album = strip_album_suffix(album)

    release = None
    csv_path = get_repertorio_csv_path()
    if csv_path:
        release = buscar_release_en_csv(csv_path, band, album)
    if release is None and repertorio:
        release = buscar_release_en_cache(repertorio, band, album)

    release_tipo = None
    if release:
        release_tipo = elegir_tipo(release.get("type") or release.get("type_id"))
    tipo_full = formatear_tipo_full(release_tipo)
    return f"{band} - {album} ({tipo_full})"


def abrir_busqueda_youtube(titulo):
    if not titulo:
        return
    query = quote(titulo)
    url = f"https://www.youtube.com/results?search_query={query}"
    webbrowser.open(url)
    print(f"Abriendo busqueda en YouTube: {titulo}")


def safe_delete_folder(folder_path, root_path):
    try:
        folder_resolved = folder_path.resolve()
        root_resolved = Path(root_path).resolve()
    except FileNotFoundError:
        print(f"No se encontro la carpeta para eliminar: {folder_path}")
        return False

    if folder_resolved == root_resolved or root_resolved not in folder_resolved.parents:
        print(f"Se omite borrar carpeta fuera de {root_resolved}: {folder_resolved}")
        return False

    try:
        shutil.rmtree(folder_resolved)
        return True
    except Exception as exc:
        print(f"No se pudo eliminar {folder_resolved}: {exc}")
    return False


def mover_carpeta_subida(folder_path, destino_root):
    try:
        folder_resolved = folder_path.resolve()
        upload_root = DIR_UPLOAD.resolve()
    except FileNotFoundError:
        print(f"No se encontro la carpeta para mover: {folder_path}")
        return None

    if upload_root not in folder_resolved.parents:
        print(f"Se omite mover carpeta fuera de {upload_root}: {folder_resolved}")
        return None

    destino_root.mkdir(parents=True, exist_ok=True)
    destino = destino_root / folder_resolved.name
    if destino.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destino = destino_root / f"{folder_resolved.name}__{stamp}"

    try:
        shutil.move(str(folder_resolved), str(destino))
        return destino
    except Exception as exc:
        print(f"No se pudo mover {folder_resolved}: {exc}")
        return None


def verificacion_manual(folder_path, repertorio):
    titulo = build_preview_title(folder_path, repertorio)
    abrir_busqueda_youtube(titulo)
    respuesta = input("¿Ya esta subido? [Y/n]: ")
    normalizada = re.sub(r"[^a-z]", "", respuesta.strip().lower())
    if normalizada.startswith(("y", "s")):
        if safe_delete_folder(folder_path, DIR_UPLOAD):
            print(f"Carpeta eliminada: {folder_path}")
        return True
    if normalizada and normalizada not in {"n", "no"}:
        print("Entrada no valida, se continua con la subida.")
    return False


def format_tracklist_for_description(tracklist):
    lines = []
    pattern = re.compile(r"^\s*\d+\s*-\s*(.*?)\s*\((\d{2}:\d{2})\)\s*$")
    for row in tracklist:
        text = str(row).strip()
        match = pattern.match(text)
        if match:
            title = match.group(1).strip()
            timestamp = match.group(2).strip()
            lines.append(f"[{timestamp}] ► {title}")
            continue
        lines.append(f"► {text}")
    return lines


def construir_descripcion(
    band,
    genre,
    year,
    links,
    tracklist,
    country_name,
    country_flag,
    tipo_full,
    playlist_cache,
):
    genre_display = format_genre_text(genre)
    country_display = country_name or ""
    year_display = str(year) if year else ""
    tipo_display = (tipo_full or "Full Album").upper()

    genre_parts = split_genres(genre_display)
    genre_playlist = find_best_playlist_link(playlist_cache, [genre_display])
    regional_playlist = None
    if country_display:
        regional_playlist = find_best_playlist_link(
            playlist_cache,
            [genre_display, f"{country_display} metal"],
            required_texts=[country_display],
        )

    regional_playlist_url = regional_playlist["url"] if regional_playlist else None
    regional_playlist_title = (
        regional_playlist["title"] if regional_playlist else country_display
    )

    hashtags = build_dynamic_hashtags(
        split_genres(genre_display),
        country_display or None,
        year_display or None,
        tipo_display,
    )

    tracklist_lines = format_tracklist_for_description(tracklist)

    divider = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    follow_lines = []
    follow_map = [
        ("📘", "Facebook", "facebook"),
        ("📸", "Instagram", "instagram"),
        ("🗂️", "Metal Archives", "metal_archives"),
        ("🌐", "Official Site", "official_site"),
    ]
    for icon, label, key in follow_map:
        value = links.get("follow", {}).get(key)
        if value:
            follow_lines.append(f"{icon} {label}: {value}")

    stream_lines = []
    stream_map = [
        ("🟢", "Spotify", "spotify", False),
        ("🔵", "Bandcamp", "bandcamp", True),
        ("🔴", "Apple Music", "apple_music", False),
        ("⚫", "YouTube Music", "youtube_music", False),
        ("🟠", "Deezer", "deezer", False),
    ]
    for icon, label, key, support_band in stream_map:
        value = links.get("stream", {}).get(key)
        if not value:
            continue
        line = f"{icon} {label}: {value}"
        if support_band:
            line += " ← SUPPORT THE BAND!"
        stream_lines.append(line)

    more_slam_lines = ["▸ Browse 1000+ albums: danielbanariba.com/metal-archive"]
    used_playlist_urls = set()
    for part in genre_parts:
        part_playlist = find_best_playlist_link(playlist_cache, [part])
        if not part_playlist:
            continue
        url = part_playlist.get("url")
        if not url or url in used_playlist_urls:
            continue
        used_playlist_urls.add(url)
        more_slam_lines.append(f"▸ {part} playlist: {url}")

    if not genre_parts and genre_playlist and genre_playlist.get("url"):
        url = genre_playlist.get("url")
        if url not in used_playlist_urls:
            used_playlist_urls.add(url)
            more_slam_lines.append(f"▸ {genre_display} playlist: {url}")

    if regional_playlist_url and regional_playlist_url not in used_playlist_urls:
        more_slam_lines.append(f"▸ {regional_playlist_title}: {regional_playlist_url}")

    lines = [
        divider,
        f"🔗 FOLLOW {band.upper()}",
        divider,
    ]
    if follow_lines:
        lines.extend(["", *follow_lines, ""])
    else:
        lines.append("")

    if stream_lines:
        lines.extend(
            [
                divider,
                "🎧 STREAM & DOWNLOAD",
                divider,
                "",
                *stream_lines,
                "",
            ]
        )

    lines.extend(
        [
            divider,
            "📀 ALBUM INFO",
            divider,
            "",
        ]
    )
    if year_display:
        lines.append(f"📅 Year: {year_display}")
    if country_display:
        lines.append(
            f"🌍 Country: {country_display}{' ' + country_flag if country_flag else ''}"
        )
    if genre_display:
        lines.append(f"⚡ Genre: {genre_display}")
    lines.extend(
        [
            "",
            divider,
            "⏱️ TRACKLIST",
            divider,
            "",
        ]
    )
    lines.extend(tracklist_lines)
    lines.extend(
        [
            "",
            divider,
            "🔥 MORE SLAM",
            divider,
            "",
            *more_slam_lines,
            "",
            "╔════════════════════════╗",
            "║       💀 SUBSCRIBE FOR MORE 💀        ║",
            "╚════════════════════════╝",
            "",
            "🔔 Daily underground extreme metal uploads",
            "👍 LIKE if this crushes your skull",
            "💬 COMMENT your favorite track",
            "🔄 SHARE with metalheads",
            "",
            divider,
            "🎸 BANDS: Submit Your Album (FREE)",
            divider,
            "",
            "Get featured on our channel (13,000+ monthly listeners)",
            "→ danielbanariba.com/metal-archive/submit",
            "",
            divider,
            "💼 LABELS: Promotional Packages Available",
            divider,
            "",
            "Reach highly engaged extreme metal audience",
            "→ danielbanariba.com/metal-archive/promo",
            "",
            divider,
            "",
            " ".join(hashtags),
        ]
    )
    return "\n".join(lines)


def encontrar_video(folder_path):
    candidates = [
        path
        for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    ]
    if not candidates:
        return None
    folder_name = folder_path.name.lower()
    for candidate in candidates:
        if candidate.stem.lower() == folder_name:
            return candidate
    return max(candidates, key=lambda item: item.stat().st_size)


RETRIABLE_STATUS_CODES = {500, 502, 503, 504}
RETRIABLE_EXCEPTIONS = (
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
    TimeoutError,
    ssl.SSLError,
    socket.gaierror,
    httplib2.error.ServerNotFoundError,
)
RETRIABLE_ERRNOS = {
    errno.ECONNABORTED,
    errno.ECONNRESET,
    errno.ECONNREFUSED,
    errno.ENETDOWN,
    errno.ENETUNREACH,
    errno.EHOSTUNREACH,
    errno.ETIMEDOUT,
    errno.EPIPE,
}
if hasattr(socket, "EAI_AGAIN"):
    RETRIABLE_ERRNOS.add(socket.EAI_AGAIN)


def subir_video(
    youtube,
    video_path,
    title,
    description,
    privacy="private",
    category_id="10",
    publish_at=None,
    label=None,
):
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": category_id,
        },
        "status": {},
    }
    status = {"privacyStatus": privacy}
    if publish_at:
        status["privacyStatus"] = "private"
        status["publishAt"] = publish_at
    body["status"] = status

    chunk_mb = int(os.environ.get("YOUTUBE_CHUNK_SIZE_MB", "8"))
    chunk_size = -1 if chunk_mb <= 0 else chunk_mb * 1024 * 1024
    media = MediaFileUpload(
        str(video_path), mimetype="video/mp4", resumable=True, chunksize=chunk_size
    )
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )
    response = None
    retry = 0
    max_retries_env = int(os.environ.get("YOUTUBE_UPLOAD_RETRIES", "0"))
    max_retries = None if max_retries_env <= 0 else max_retries_env
    last_percent = -1
    label_text = f"{label}: " if label else ""

    while response is None:
        try:
            status, response = request.next_chunk(num_retries=1)
            if status:
                progress = status.progress()
                percent = int(progress * 100)
                if percent != last_percent:
                    width = 30
                    filled = int(round(width * max(0.0, min(1.0, progress))))
                    bar = "#" * filled + "-" * (width - filled)
                    sys.stdout.write(f"\rSubiendo: {label_text}[{bar}] {percent:3d}%")
                    sys.stdout.flush()
                    last_percent = percent
        except HttpError as exc:
            if last_percent >= 0:
                sys.stdout.write("\n")
                sys.stdout.flush()
                last_percent = -1
            # Detectar error de cuota agotada (API)
            if is_quota_error(exc):
                raise QuotaExceededError(f"Cuota diaria agotada: {exc}")
            # Detectar limite de subidas del canal
            if is_upload_limit_error(exc):
                raise UploadLimitExceededError(
                    f"Limite de subidas del canal alcanzado: {exc}"
                )
            if exc.resp is not None and exc.resp.status in RETRIABLE_STATUS_CODES:
                retry += 1
                if max_retries is not None and retry > max_retries:
                    raise
                wait_time = min(60, 2**retry)
                print(
                    f"Error temporal al subir ({exc.resp.status}). Reintentando en {wait_time}s..."
                )
                time.sleep(wait_time)
                continue
            raise
        except RETRIABLE_EXCEPTIONS as exc:
            if last_percent >= 0:
                sys.stdout.write("\n")
                sys.stdout.flush()
                last_percent = -1
            retry += 1
            if max_retries is not None and retry > max_retries:
                raise
            wait_time = min(60, 2**retry)
            print(
                f"Error de conexion al subir ({exc}). Reintentando en {wait_time}s..."
            )
            time.sleep(wait_time)
            continue
        except OSError as exc:
            if exc.errno not in RETRIABLE_ERRNOS:
                raise
            if last_percent >= 0:
                sys.stdout.write("\n")
                sys.stdout.flush()
                last_percent = -1
            retry += 1
            if max_retries is not None and retry > max_retries:
                raise
            wait_time = min(60, 2**retry)
            print(
                f"Error de conexion al subir ({exc}). Reintentando en {wait_time}s..."
            )
            time.sleep(wait_time)
            continue
    if last_percent >= 0:
        width = 30
        bar = "#" * width
        sys.stdout.write(f"\rSubiendo: {label_text}[{bar}] 100%\n")
        sys.stdout.flush()
    return response


def procesar_carpeta(
    folder_path,
    session,
    generos,
    generos_lookup,
    repertorio,
    allow_genre_fallback=False,
):
    tracks, context = collect_audio_tracks(folder_path)
    if not tracks:
        print(f"No hay audios en {folder_path}, se omite.")
        return None

    band, album = parse_band_album_from_folder(folder_path.name)
    if context["band"]:
        band = context["band"]
    if context["album"]:
        album = context["album"]

    csv_path = get_repertorio_csv_path()

    release = None
    if csv_path:
        release = buscar_release_en_csv(csv_path, band, album)
        if release:
            print("Release encontrado en CSV local.")

    if release is None and repertorio:
        release = buscar_release_en_cache(repertorio, band, album)
    post = None
    if release is None and session:
        post = esperar_release_en_api(
            session, band, album, generos, allow_genre_fallback
        )
        if post:
            print(f"Release encontrado en API: {band} - {album}")
        else:
            print(f"Release NO encontrado en API: {band} - {album}")
    elif release is not None:
        post = release

    release_tipo = None
    release_year = None
    genre_text = None
    post_id = None
    band_id = None
    post_detail = None
    csv_country = None

    if post:
        release_tipo = elegir_tipo(post.get("type") or post.get("type_id"))
        release_year = extraer_anio(post)
        genre_text = extraer_genero(post, generos_lookup)
        post_id = post.get("postId") or post.get("post_id") or post.get("id")
        band_id = (
            post.get("bandId") or post.get("band_id") or obtener_band_id(post, band)
        )
        csv_country = extraer_pais_desde_data(post, band_id=band_id, band_name=band)

    if (
        session
        and post_id
        and (not band_id or not release_tipo or not release_year or not genre_text)
    ):
        post_detail = api_get(session, f"/posts/{post_id}")
        post_detail = unwrap_deathgrind_payload(post_detail)
        if post_detail:
            if not band_id:
                band_id = (
                    post_detail.get("bandId")
                    or post_detail.get("band_id")
                    or obtener_band_id(post_detail, band)
                )
            if not release_tipo:
                release_tipo = elegir_tipo(
                    post_detail.get("type") or post_detail.get("type_id")
                )
            if not release_year:
                release_year = extraer_anio(post_detail)
            if not genre_text:
                genre_text = extraer_genero(post_detail, generos_lookup)
            post = post_detail

    if not session:
        if not release_year:
            release_year = extraer_anio_de_texto(folder_path.name) or context["year"]
        if not genre_text:
            genre_text = context["genre"]

    tipo_full = formatear_tipo_full(release_tipo)

    band_data = None
    if session and band_id:
        band_data = api_get(session, f"/bands/{band_id}")
        band_data = unwrap_deathgrind_payload(band_data)

    pais = None
    if band_data:
        pais = extraer_pais_desde_data(band_data, band_id=band_id, band_name=band)
    if not pais:
        pais = extraer_pais_desde_data(
            post_detail or post, band_id=band_id, band_name=band
        )
    if not pais and csv_country:
        pais = csv_country
    pais = normalizar_pais(pais)

    # Si DeathGrind esta activo, no inventar valores. Mejor omitir y reintentar luego.
    if session and (not release_year or not genre_text or not pais):
        print(
            f"Metadata incompleta desde API para {band} - {album}. "
            "Se omite para reintentar en la siguiente pasada."
        )
        return None

    country_meta = resolve_country_meta(pais)
    country_name = country_meta.get("name") or pais or ""
    country_flag = country_meta.get("flag") or ""

    year_text = str(release_year) if release_year else ""
    title = compress_title_to_limit(
        country_flag=country_flag,
        band=band,
        album=album,
        year_text=year_text,
        genre_text=genre_text,
        tipo_full=tipo_full,
    )

    links = {"stream": {}, "follow": {}}
    if session and post_id:
        links = extraer_links_deathgrind(session, post_id)
        tiene_links = any(links["stream"].values()) or any(links["follow"].values())
        if not tiene_links:
            html_links = extraer_links_post_html(session, post_id)
            links = merge_links(links, html_links)

    tiene_links = any(links["stream"].values()) or any(links["follow"].values())
    if session and band_id:
        band_links = extraer_links_band_api(session, band_id, band_data=band_data)
        links = merge_links(links, band_links)
        tiene_links = any(links["stream"].values()) or any(links["follow"].values())
        if not tiene_links:
            html_links = extraer_links_band_html(session, band_id)
            links = merge_links(links, html_links)

    track_titles = []
    if post:
        track_titles = obtener_tracklist_api(session, post, post_id)
        if track_titles:
            print(
                f"Tracklist desde API: {len(track_titles)} pistas en {folder_path.name}"
            )
        else:
            print(
                f"No se encontro tracklist en la API para {folder_path.name}, usando audio."
            )

    tracklist = build_tracklist(tracks, track_titles if track_titles else None)
    safe_tracklist = [censor_profanity(line) for line in tracklist]
    playlist_cache = load_playlist_links_cache()
    description = construir_descripcion(
        band=band,
        genre=genre_text,
        year=release_year,
        links=links,
        tracklist=safe_tracklist,
        country_name=country_name,
        country_flag=country_flag,
        tipo_full=tipo_full,
        playlist_cache=playlist_cache,
    )
    description = ensure_description_limit(description)
    title = censor_profanity(title)
    title = truncar_titulo(title)
    description = censor_profanity(description)
    description = truncar_descripcion(description)
    genres_list = split_genres(genre_text)

    return {
        "title": title,
        "description": description,
        "band": band,
        "country": pais,
        "genres": genres_list,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Subir videos con API de YouTube y descripcion automatizada."
    )
    parser.add_argument("--carpeta", help="Carpeta especifica dentro de upload_video")
    parser.add_argument(
        "--limite",
        type=int,
        default=1,
        help="Cantidad maxima de videos a subir (default: 1)",
    )
    parser.add_argument(
        "--todo", action="store_true", help="Procesar todas las carpetas disponibles"
    )
    parser.add_argument(
        "--con-generos",
        action="store_true",
        help="Hacer busqueda completa por generos (mas lenta)",
    )
    parser.add_argument(
        "--sin-deathgrind", action="store_true", help="No usar la API de DeathGrind"
    )
    parser.add_argument(
        "--sin-links-web", action="store_true", help="(Deprecated) Ya no se usa"
    )
    parser.add_argument(
        "--buscar-links", action="store_true", help="(Deprecated) Ya no se usa"
    )
    parser.add_argument(
        "--sin-verificacion",
        action="store_true",
        help="No abrir verificacion manual en YouTube",
    )
    parser.add_argument(
        "--privacidad", choices=["public", "private", "unlisted"], default="public"
    )
    parser.add_argument(
        "--publicar-ahora",
        action="store_true",
        help="(Deprecated) Ya se publica al momento",
    )
    parser.add_argument(
        "--modo-inmediato",
        action="store_true",
        help="Subir videos sin programar en lote",
    )
    parser.add_argument(
        "--cantidad-lote",
        type=int,
        help="Cantidad de videos para programar en lote (default: 96)",
    )
    parser.add_argument(
        "--gap-horas",
        type=float,
        help="Horas de diferencia entre publicaciones (default: 0.25)",
    )
    args = parser.parse_args()

    cargar_env()
    session = None
    if not args.sin_deathgrind:
        try:
            session = crear_sesion_autenticada()
        except Exception as exc:
            print(f"Error al iniciar sesion DeathGrind: {exc}")

    generos, generos_lookup = cargar_generos()
    repertorio = cargar_repertorio()

    youtube = authenticate(prefix="upload")
    if session and not args.sin_deathgrind:
        ok = deathgrind_smoke_test(session, generos)
        if not ok:
            global DEATHGRIND_DISABLED
            DEATHGRIND_DISABLED = True
            print("DeathGrind desactivado para esta ejecucion por fallas en la prueba.")

    upload_dir = DIR_UPLOAD
    if args.carpeta:
        upload_dir = Path(args.carpeta)
        if not upload_dir.is_absolute():
            upload_dir = DIR_UPLOAD / args.carpeta

    if not upload_dir.exists():
        print(f"No existe la carpeta: {upload_dir}")
        return

    allow_genre_fallback = args.con_generos
    max_uploads = None if args.todo else args.limite
    scan_interval = int(os.environ.get("UPLOAD_SCAN_INTERVAL", "30"))
    batch_mode = not args.modo_inmediato
    batch_size = (
        args.cantidad_lote
        if args.cantidad_lote
        else int(os.environ.get("YOUTUBE_BATCH_SIZE", "96"))
    )
    gap_hours = (
        args.gap_horas
        if args.gap_horas is not None
        else float(os.environ.get("YOUTUBE_BATCH_GAP_HOURS", "0.25"))
    )
    if batch_size <= 0:
        batch_size = 1
    if gap_hours <= 0:
        gap_hours = 0.25
    pending_items = []
    queued_names = set()
    revisadas = set()  # Todas las carpetas ya revisadas (agregadas o no)
    total_uploaded = 0
    total_skipped_missing_video = 0
    total_failed_api = 0

    def imprimir_resumen_subida():
        print(f"\n{'=' * 60}")
        print("RESUMEN DE SUBIDA")
        print(f"Subidos: {total_uploaded}")
        print(f"Omitidos sin video: {total_skipped_missing_video}")
        print(f"Fallidos API: {total_failed_api}")
        print(f"{'=' * 60}\n")

    while True:
        if args.carpeta:
            folders = [upload_dir] if upload_dir.exists() else []
        else:
            folders = [path for path in upload_dir.iterdir() if path.is_dir()]
            folders = ordenar_carpetas_para_subida(folders)

        if not folders:
            if args.carpeta:
                print("No se encontro la carpeta especificada.")
                return
            print(f"No hay carpetas para subir. Esperando {scan_interval}s...")
            time.sleep(scan_interval)
            continue

        if batch_mode:
            # Fase 1: Seleccion rapida (solo verificacion manual, sin buscar metadata)
            for folder_path in folders:
                if len(pending_items) >= batch_size:
                    break
                if folder_path.name in revisadas:
                    continue
                # Marcar como revisada inmediatamente (antes de cualquier decision)
                revisadas.add(folder_path.name)
                video_path = encontrar_video(folder_path)
                if not video_path:
                    print(f"No hay video .mp4 en {folder_path}, se omite.")
                    total_skipped_missing_video += 1
                    continue
                if not args.sin_verificacion:
                    if verificacion_manual(folder_path, repertorio):
                        continue
                # Solo guardamos folder y video, metadata se busca despues
                pending_items.append(
                    {
                        "folder": folder_path,
                        "video": video_path,
                        "metadata": None,
                    }
                )
                queued_names.add(folder_path.name)
                print(
                    f"En cola para programar: {folder_path.name} ({len(pending_items)}/{batch_size})"
                )

            if len(pending_items) < batch_size:
                # Contar carpetas no revisadas que aun existen
                carpetas_pendientes = sum(1 for f in folders if f.name not in revisadas)
                if carpetas_pendientes == 0:
                    # Ya no hay mas carpetas por revisar
                    if pending_items:
                        print(
                            f"\nNo hay mas carpetas disponibles. Continuando con {len(pending_items)} videos."
                        )
                    else:
                        print("No hay carpetas con videos para subir.")
                        if args.carpeta:
                            break
                        print(f"Esperando nuevas carpetas en {scan_interval}s...")
                        time.sleep(scan_interval)
                        revisadas.clear()  # Reset para detectar nuevas carpetas
                        continue
                else:
                    if args.carpeta:
                        print(f"No se juntaron {batch_size} videos para programar.")
                        break
                    faltan = batch_size - len(pending_items)
                    print(
                        f"Aun faltan {faltan} videos para programar. Esperando {scan_interval}s..."
                    )
                    time.sleep(scan_interval)
                    continue

            # Fase 2: Busqueda de metadata (diferida, despues de seleccionar todo el lote)
            print(
                f"\nIniciando busqueda de metadata para {len(pending_items)} videos..."
            )
            items_validos = []
            total_items = len(pending_items)
            mostrar_progreso_busqueda(0, total_items)
            for idx, item in enumerate(pending_items, start=1):
                folder_path = item["folder"]
                metadata = procesar_carpeta(
                    folder_path,
                    session,
                    generos,
                    generos_lookup,
                    repertorio,
                    allow_genre_fallback,
                )
                mostrar_progreso_busqueda(idx, total_items)
                if metadata:
                    item["metadata"] = metadata
                    items_validos.append(item)
                else:
                    sys.stdout.write(f"\n[!] Sin metadata: {folder_path.name}\n")
                    sys.stdout.flush()
                    mostrar_progreso_busqueda(idx, total_items)
            pending_items = items_validos
            print(f"Metadata obtenida para {len(pending_items)} videos.")

            if not pending_items:
                print("No hay videos con metadata valida para programar.")
                break

            # Usar el tamano real del lote (puede ser menor si algunos fallaron)
            actual_batch_size = len(pending_items)

            tz_local = datetime.now().astimezone().tzinfo
            schedule_date = (datetime.now(tz_local) + timedelta(days=1)).date()
            max_per_day = max_videos_por_dia(gap_hours)
            max_days = int(os.environ.get("YOUTUBE_BATCH_MAX_DAYS", "365"))
            if max_days <= 0:
                max_days = 365
            scan_limit = resolve_schedule_scan_limit(max_per_day, max_days)
            taken_slots, counts_by_date = fetch_scheduled_publish_times_safe(
                youtube,
                max_items=scan_limit,
            )
            if actual_batch_size > max_per_day:
                print(
                    f"No se puede programar {actual_batch_size} videos con separacion de {gap_hours}h."
                )
                break
            schedule_date, slots = find_next_batch_schedule(
                schedule_date,
                actual_batch_size,
                gap_hours,
                taken_slots,
                counts_by_date,
                tz_local,
                max_days=max_days,
            )
            if not slots or len(slots) < actual_batch_size:
                print("No se pudo generar el horario de programacion para el lote.")
                break

            latest_slot = max(taken_slots) if taken_slots else None
            requested_date, override = prompt_confirm_schedule_date(
                schedule_date, latest_slot
            )
            if override:
                schedule_date, slots = find_next_batch_schedule(
                    requested_date,
                    actual_batch_size,
                    gap_hours,
                    taken_slots,
                    counts_by_date,
                    tz_local,
                    max_days=max_days,
                )
                if not slots or len(slots) < actual_batch_size:
                    print(
                        "No se pudo generar el horario de programacion para la fecha solicitada."
                    )
                    break
                if schedule_date != requested_date:
                    print(
                        "No se pudo usar la fecha solicitada "
                        f"({requested_date.isoformat()}); se programa para {schedule_date.isoformat()}."
                    )

            existentes = counts_by_date.get(schedule_date, 0)
            if existentes:
                print(
                    f"Ya hay {existentes} videos programados para {schedule_date.isoformat()}, se programa el lote ahi."
                )
            if RANDOMIZE_VIDEO_SELECTION:
                random.shuffle(pending_items)
            random.shuffle(slots)
            print(
                f"Programando {actual_batch_size} videos para {schedule_date.isoformat()} con separacion de {gap_hours}h."
            )
            total_items = len(pending_items)
            upload_limit_hit = False
            for idx, (item, slot) in enumerate(zip(pending_items, slots), start=1):
                if upload_limit_hit:
                    break
                folder_path = item["folder"]
                metadata = item["metadata"]
                video_path = item.get("video")
                if not video_path or not Path(video_path).exists():
                    video_path = encontrar_video(folder_path)
                    if video_path:
                        item["video"] = video_path
                if not video_path:
                    print(f"[OMITIDO] No hay video .mp4 en {folder_path}, se omite.")
                    total_skipped_missing_video += 1
                    continue
                publish_at = format_rfc3339(slot)
                restantes = total_items - idx
                print(f"Subiendo {idx}/{total_items}. Faltan {restantes}.")
                video_missing = False
                while True:
                    try:
                        # Revalidar justo antes de subir para evitar caidas si la carpeta se movio.
                        if not Path(video_path).exists():
                            refreshed_video = encontrar_video(folder_path)
                            if not refreshed_video:
                                print(
                                    f"[OMITIDO] Video no encontrado al subir: {folder_path.name}"
                                )
                                total_skipped_missing_video += 1
                                video_missing = True
                                break
                            video_path = refreshed_video
                            item["video"] = video_path
                        response = subir_video(
                            youtube,
                            video_path,
                            metadata["title"],
                            metadata["description"],
                            privacy=args.privacidad,
                            publish_at=publish_at,
                            label=metadata.get("title"),
                        )
                        print(
                            f"Video subido: {response.get('id')} (programado: {publish_at})"
                        )
                        total_uploaded += 1
                        destino = mover_carpeta_subida(folder_path, DIR_YA_SUBIDOS)
                        if destino:
                            print(f"Carpeta movida a {destino}")
                        break  # Subida exitosa, salir del while
                    except QuotaExceededError as exc:
                        print(f"Cuota agotada: {exc}")
                        print("Intentando rotar a otra credencial de upload...")
                        youtube = authenticate_next(prefix="upload")
                        if youtube is None:
                            print("No hay mas credenciales disponibles. Abortando.")
                            return
                        print("Credencial rotada exitosamente. Reintentando subida...")
                    except UploadLimitExceededError as exc:
                        print(f"\n{'=' * 60}")
                        print(f"LIMITE DE SUBIDAS DEL CANAL ALCANZADO")
                        print(f"{'=' * 60}")
                        print(
                            f"YouTube limita la cantidad de videos que puedes subir por dia."
                        )
                        print(f"Este limite es por CANAL, no por proyecto de API.")
                        print(f"Rotar credenciales NO ayuda con este error.")
                        print(f"Videos subidos en este lote: {idx - 1}/{total_items}")
                        print(f"Debes esperar ~24 horas para continuar subiendo.")
                        print(f"{'=' * 60}\n")
                        upload_limit_hit = True
                        break
                    except HttpError as exc:
                        print(f"Error YouTube API en {folder_path.name}: {exc}")
                        total_failed_api += 1
                        break  # Otro error HTTP, continuar con siguiente video
                    except FileNotFoundError:
                        refreshed_video = encontrar_video(folder_path)
                        if refreshed_video:
                            video_path = refreshed_video
                            item["video"] = video_path
                            print(
                                f"[REINTENTO] Video reaparecio, reintentando: {folder_path.name}"
                            )
                            continue
                        print(f"[OMITIDO] Video no existe para {folder_path.name}")
                        total_skipped_missing_video += 1
                        video_missing = True
                        break
                if video_missing:
                    continue
            if upload_limit_hit:
                print("Abortando lote por limite de subidas del canal.")
                imprimir_resumen_subida()
                return
            break

        uploaded = 0
        for folder_path in folders:
            if max_uploads is not None and uploaded >= max_uploads:
                break
            video_path = encontrar_video(folder_path)
            if not video_path:
                print(f"No hay video .mp4 en {folder_path}, se omite.")
                total_skipped_missing_video += 1
                continue
            if not args.sin_verificacion:
                if verificacion_manual(folder_path, repertorio):
                    continue
            print(f"Procesando {folder_path.name}...")
            metadata = procesar_carpeta(
                folder_path,
                session,
                generos,
                generos_lookup,
                repertorio,
                allow_genre_fallback,
            )
            if not metadata:
                continue
            upload_limit_hit = False
            video_missing = False
            while True:
                try:
                    video_path = encontrar_video(folder_path)
                    if not video_path:
                        print(
                            f"[OMITIDO] No hay video .mp4 en {folder_path}, se omite."
                        )
                        video_missing = True
                        break
                    response = subir_video(
                        youtube,
                        video_path,
                        metadata["title"],
                        metadata["description"],
                        privacy=args.privacidad,
                        publish_at=None,
                        label=metadata.get("title"),
                    )
                    print(f"Video subido: {response.get('id')}")
                    uploaded += 1
                    total_uploaded += 1
                    destino = mover_carpeta_subida(folder_path, DIR_YA_SUBIDOS)
                    if destino:
                        print(f"Carpeta movida a {destino}")
                    break  # Subida exitosa, salir del while
                except QuotaExceededError as exc:
                    print(f"Cuota agotada: {exc}")
                    print("Intentando rotar a otra credencial de upload...")
                    youtube = authenticate_next(prefix="upload")
                    if youtube is None:
                        print("No hay mas credenciales disponibles. Abortando.")
                        return
                    print("Credencial rotada exitosamente. Reintentando subida...")
                except UploadLimitExceededError as exc:
                    print(f"\n{'=' * 60}")
                    print(f"LIMITE DE SUBIDAS DEL CANAL ALCANZADO")
                    print(f"{'=' * 60}")
                    print(
                        f"YouTube limita la cantidad de videos que puedes subir por dia."
                    )
                    print(f"Este limite es por CANAL, no por proyecto de API.")
                    print(f"Rotar credenciales NO ayuda con este error.")
                    print(f"Videos subidos en esta sesion: {uploaded}")
                    print(f"Debes esperar ~24 horas para continuar subiendo.")
                    print(f"{'=' * 60}\n")
                    upload_limit_hit = True
                    break
                except HttpError as exc:
                    print(f"Error YouTube API en {folder_path.name}: {exc}")
                    total_failed_api += 1
                    break  # Otro error HTTP, continuar con siguiente video
                except FileNotFoundError:
                    print(
                        f"[OMITIDO] Video no encontrado durante la subida: {folder_path.name}"
                    )
                    total_skipped_missing_video += 1
                    video_missing = True
                    break
            if video_missing:
                continue
            if upload_limit_hit:
                print("Abortando por limite de subidas del canal.")
                imprimir_resumen_subida()
                return

        if args.carpeta:
            break

    imprimir_resumen_subida()


if __name__ == "__main__":
    main()
