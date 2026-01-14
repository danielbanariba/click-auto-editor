#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import os
import random
import re
import shutil
import ssl
import sys
import time
import webbrowser
from urllib.parse import quote, unquote
import uuid
from datetime import datetime, timedelta, time as dtime, timezone
from pathlib import Path

import requests
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from mutagen import File as MutagenFile

from config import AUDIO_FORMATS, DIR_UPLOAD, DIR_YA_SUBIDOS
from subir_video.authenticate import authenticate

BASE_URL = "https://deathgrind.club"
API_URL = f"{BASE_URL}/api"
INTRO_SECONDS = 8

DELAY_ENTRE_PAGINAS = 1.0
DELAY_BASE_429 = 30
MAX_RETRIES_ERROR = 5
MAX_RETRIES_429 = int(os.environ.get("DEATHGRIND_MAX_429", "1"))
DEATHGRIND_DISABLED = False

DEFAULT_SCHEDULE_HOURS = [8, 12]
DEFAULT_PLAYLIST_PRIVACY = os.environ.get("YOUTUBE_PLAYLIST_PRIVACY", "public")

TIPOS_DISCO = {
    1: "Album", 2: "EP", 3: "Demo", 4: "Single",
    5: "Split", 6: "Compilation", 7: "Live", 8: "Boxset", 9: "EP",
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
    "facebook",
    "instagram",
    "youtube",
    "tiktok",
    "twitter",
    "metal_archives",
    "spirit_of_metal",
]

SERVICE_LABELS = {
    "bandcamp": "Bandcamp",
    "spotify": "Spotify",
    "youtube_music": "YouTube Music",
    "apple_music": "Apple Music",
    "deezer": "Deezer",
    "amazon": "Amazon Music",
    "facebook": "Facebook",
    "instagram": "Instagram",
    "youtube": "YouTube",
    "tiktok": "TikTok",
    "twitter": "X/Twitter",
    "metal_archives": "Metal Archives",
    "spirit_of_metal": "Spirit of Metal",
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
}


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
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json",
    })

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


def api_get(session, endpoint, params=None, max_retries=MAX_RETRIES_ERROR):
    global DEATHGRIND_DISABLED
    if DEATHGRIND_DISABLED or session is None:
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
                    DEATHGRIND_DISABLED = True
                    print("Rate limit DeathGrind, se desactiva la API en esta ejecucion.")
                    return None
                wait_time = DELAY_BASE_429 * retries_429
                print(f"Rate limit DeathGrind, esperando {wait_time}s...")
                time.sleep(wait_time)
                continue
            if response.status_code != 200:
                retries_error += 1
                if retries_error >= max_retries:
                    return None
                time.sleep(5)
                continue
            return response.json()
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
    try:
        resp = session.get(
            f"{API_URL}/posts/filter",
            params={"genres": genre_id},
            timeout=30,
        )
    except requests.RequestException as exc:
        print(f"Prueba DeathGrind fallo: {exc}")
        return False

    print(f"Prueba DeathGrind /posts/filter?genres={genre_id} ({genre_name}) status: {resp.status_code}")
    if resp.status_code != 200:
        return False

    data = resp.json()
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
        try:
            resp = session.get(f"{API_URL}/bands/{band_id}/discography", timeout=30)
            print(f"Prueba DeathGrind /bands/{band_id}/discography status: {resp.status_code}")
            if resp.status_code == 200:
                discography = resp.json()
                disc_posts = discography.get("posts") or []
                print(f"Prueba DeathGrind discography posts: {len(disc_posts)}")
        except requests.RequestException as exc:
            print(f"Prueba DeathGrind /bands/{band_id}/discography fallo: {exc}")
            return False
    else:
        print("Prueba DeathGrind: no se encontro bandId en el primer post.")

    if post_id:
        try:
            resp = session.get(f"{API_URL}/posts/{post_id}/links", timeout=30)
            print(f"Prueba DeathGrind /posts/{post_id}/links status: {resp.status_code}")
            if resp.status_code == 200:
                links = resp.json().get("links") or []
                print(f"Prueba DeathGrind links: {len(links)}")
        except requests.RequestException as exc:
            print(f"Prueba DeathGrind /posts/{post_id}/links fallo: {exc}")
            return False
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
    return "/".join(parts) if parts else "Unknown"


def strip_album_suffix(value):
    cleaned = str(value)
    patterns = [
        re.compile(r"\s*[\[(]\s*\d{4}\s*[\])]\s*$"),
        re.compile(r"\s*[\[(]\s*(ep|album|demo|single|split|compilation|live|boxset)\s*[\])]\s*$", re.IGNORECASE),
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
    return ", ".join(generos) if generos else None


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


def extraer_pais_valor(value):
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("name", "title", "label", "value", "country", "countryName", "country_name"):
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


def extraer_pais_desde_data(data, band_id=None, band_name=None):
    if not isinstance(data, dict):
        return None
    for key in ("country", "pais", "origin", "countryName", "country_name", "location"):
        parsed = extraer_pais_valor(data.get(key))
        if parsed:
            return parsed
    bands = data.get("bands") or data.get("band")
    if isinstance(bands, dict):
        return extraer_pais_desde_data(bands, band_id=band_id, band_name=band_name)
    if isinstance(bands, list):
        for band in bands:
            if not isinstance(band, dict):
                continue
            if band_id and band.get("bandId") and str(band.get("bandId")) != str(band_id):
                continue
            if band_name and band.get("name") and normalize_name(band.get("name")) != normalize_name(band_name):
                continue
            parsed = extraer_pais_desde_data(band)
            if parsed:
                return parsed
    return None


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
        return normalize_album_name(post_album) == normalize_album_name(album_name)
    return True


def buscar_release_en_cache(repertorio, band_name, album_name):
    if not repertorio:
        return None
    norm_band = normalize_name(band_name)
    norm_album = normalize_album_name(album_name) if album_name else ""
    for release in repertorio:
        if normalize_name(release.get("band", "")) != norm_band:
            continue
        if norm_album and normalize_album_name(release.get("album", "")) != norm_album:
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

                genre_ids = []
                genre_raw = row.get("genre_ids") or ""
                if genre_raw:
                    try:
                        genre_ids = ast.literal_eval(genre_raw)
                        if isinstance(genre_ids, int):
                            genre_ids = [genre_ids]
                    except Exception:
                        genre_ids = [int(val) for val in re.findall(r"\d+", str(genre_raw))]
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

                return {
                    "band": bands[0] if bands else band_name,
                    "album": album or album_name,
                    "post_id": int(row.get("post_id")) if row.get("post_id") else None,
                    "type_id": type_id,
                    "genres": genre_ids,
                    "year": year,
                    "post_url": row.get("post_url"),
                }
    except Exception:
        return None
    return None


def buscar_release_en_api(session, band_name, album_name, generos, allow_genre_fallback=False):
    if DEATHGRIND_DISABLED:
        return None
    album_search = strip_album_suffix(album_name)
    query = f"{band_name} {album_search}".strip()
    data = api_get(session, "/posts/filter", params={"search": query})
    if data:
        posts = data.get("posts", [])
        for post in posts:
            if match_post(post, band_name, album_name):
                return post

        offset = data.get("offset")
        while data.get("hasMore") and offset is not None:
            data = api_get(session, "/posts/filter", params={"search": query, "offset": offset})
            if not data:
                break
            posts = data.get("posts", [])
            for post in posts:
                if match_post(post, band_name, album_name):
                    return post
            offset = data.get("offset")

    if not allow_genre_fallback or not generos:
        return None

    for genre_id, _genre_name in generos:
        offset = None
        while True:
            params = {"genres": genre_id}
            if offset is not None:
                params["offset"] = offset
            data = api_get(session, "/posts/filter", params=params)
            if not data:
                break
            posts = data.get("posts", [])
            for post in posts:
                if match_post(post, band_name, album_name):
                    return post
            if not data.get("hasMore"):
                break
            offset = data.get("offset")
            if offset is None:
                break
            time.sleep(DELAY_ENTRE_PAGINAS)
    return None


def iter_link_items(data):
    if isinstance(data, list):
        for item in data:
            yield item
        return
    if not isinstance(data, dict):
        return
    for key in ("links", "stream", "download", "social", "socials", "follow"):
        value = data.get(key)
        if isinstance(value, list):
            for item in value:
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
    name = get_link_value(item, ["name", "type", "title", "label"])
    if not url:
        return None, None
    url_lower = url.lower()
    service_key = None
    for key, patterns in SERVICE_MATCHERS.items():
        if any(pattern in url_lower for pattern in patterns):
            service_key = key
            break
    if not service_key and name:
        name_lower = name.lower()
        for key in SERVICE_MATCHERS.keys():
            if key.replace("_", " ") in name_lower:
                service_key = key
                break
    return service_key, url


def extraer_links_desde_data(data):
    links = {"stream": {}, "follow": {}}
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
        meta["track_number"] = parse_track_number(first_value(tags.get("tracknumber") or tags.get("track")))
        date_value = first_value(tags.get("date") or tags.get("year") or tags.get("originaldate"))
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
        tracks.append({
            "title": title,
            "track_number": track_number,
            "duration": duration,
            "filename": audio_path.name,
        })
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
        entries.sort(key=lambda item: (item[0] if item[0] is not None else 9999, item[1]))
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


def max_videos_por_dia(gap_hours):
    if gap_hours <= 0:
        return 0
    return max(1, int(24 // gap_hours))


def build_batch_schedule_for_date(date_value, count, gap_hours, taken_slots, tz_local):
    if count <= 0:
        return []
    gap_minutes = max(1, int(gap_hours)) * 60
    max_span = gap_minutes * (count - 1)
    if max_span >= 24 * 60:
        return []

    max_offset = 24 * 60 - max_span - 1
    day_start = datetime.combine(date_value, dtime(0, 0), tzinfo=tz_local)
    attempts = min(60, max_offset + 1)

    for _ in range(attempts):
        offset = random.randint(0, max_offset)
        slots = [normalize_slot(day_start + timedelta(minutes=offset + gap_minutes * idx))
                 for idx in range(count)]
        if any(slot in taken_slots for slot in slots):
            continue
        return slots

    for offset in range(max_offset + 1):
        slots = [normalize_slot(day_start + timedelta(minutes=offset + gap_minutes * idx))
                 for idx in range(count)]
        if any(slot in taken_slots for slot in slots):
            continue
        return slots

    return slots


def find_next_batch_schedule(start_date, count, gap_hours, taken_slots, counts_by_date, tz_local, max_days=365):
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
        resp = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=page_token,
        ).execute()
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
            chunk = video_ids[i:i + 50]
            vids = youtube.videos().list(
                part="status",
                id=",".join(chunk),
            ).execute()
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
        if fetched >= max_items:
            break
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return scheduled, counts_by_date


def fetch_scheduled_publish_times_safe(youtube, max_items=200, retries=3):
    for attempt in range(1, retries + 1):
        try:
            return fetch_scheduled_publish_times(youtube, max_items=max_items)
        except (HttpError, OSError) as exc:
            if attempt >= retries:
                print(f"No se pudo consultar videos programados: {exc}. Se continua sin evitar choques.")
                return set(), {}
            wait_time = min(30, 2 ** attempt)
            print(f"Error consultando programados ({exc}). Reintentando en {wait_time}s...")
            time.sleep(wait_time)


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
            day_slots.append(build_daily_slot(current_date, hour, tz_local, taken_slots))
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


def list_playlists(youtube):
    playlists = {}
    page_token = None
    while True:
        resp = youtube.playlists().list(
            part="snippet",
            mine=True,
            maxResults=50,
            pageToken=page_token,
        ).execute()
        for item in resp.get("items", []):
            title = item.get("snippet", {}).get("title")
            if not title:
                continue
            playlists[title.lower()] = item.get("id")
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return playlists


def is_playlist_not_found(exc):
    if not isinstance(exc, HttpError):
        return False
    if getattr(exc, "resp", None) and exc.resp.status == 404:
        return True
    try:
        data = json.loads(exc.content.decode("utf-8"))
        for err in data.get("error", {}).get("errors", []):
            if err.get("reason") == "playlistNotFound":
                return True
    except Exception:
        pass
    return False


def refresh_playlist(youtube, title, cache, privacy=DEFAULT_PLAYLIST_PRIVACY):
    key = title.lower().strip()
    cache.pop(key, None)
    return get_or_create_playlist(youtube, title, cache, privacy=privacy)


def get_or_create_playlist(youtube, title, cache, privacy=DEFAULT_PLAYLIST_PRIVACY):
    if not title:
        return None
    key = title.lower().strip()
    if not key:
        return None
    playlist_id = cache.get(key)
    if playlist_id:
        return playlist_id

    body = {
        "snippet": {"title": title},
        "status": {"privacyStatus": privacy},
    }
    resp = youtube.playlists().insert(part="snippet,status", body=body).execute()
    playlist_id = resp.get("id")
    if playlist_id:
        cache[key] = playlist_id
    return playlist_id


def is_video_in_playlist(youtube, playlist_id, video_id, max_items=200):
    page_token = None
    fetched = 0
    while True:
        resp = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=page_token,
        ).execute()
        items = resp.get("items", [])
        for item in items:
            resource = item.get("snippet", {}).get("resourceId", {})
            if resource.get("videoId") == video_id:
                return True
        fetched += len(items)
        if fetched >= max_items:
            return False
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return False


def add_video_to_playlist(youtube, playlist_id, video_id):
    body = {
        "snippet": {
            "playlistId": playlist_id,
            "resourceId": {
                "kind": "youtube#video",
                "videoId": video_id,
            },
        }
    }
    youtube.playlistItems().insert(part="snippet", body=body).execute()


def add_video_to_playlists(youtube, video_id, band_name, genres, playlist_cache):
    if not video_id:
        return
    max_scan = int(os.environ.get("YOUTUBE_PLAYLIST_SCAN", "200"))
    playlist_titles = []
    if band_name and band_name.lower() != "unknown":
        playlist_titles.append(band_name)
    playlist_titles.extend(genres)

    for title in playlist_titles:
        if not title:
            continue
        playlist_id = get_or_create_playlist(youtube, title, playlist_cache)
        if not playlist_id:
            continue
        try:
            in_playlist = is_video_in_playlist(youtube, playlist_id, video_id, max_items=max_scan)
        except HttpError as exc:
            if is_playlist_not_found(exc):
                playlist_id = refresh_playlist(youtube, title, playlist_cache)
                in_playlist = False
            else:
                raise
        if not playlist_id:
            continue
        if not in_playlist:
            try:
                add_video_to_playlist(youtube, playlist_id, video_id)
            except HttpError as exc:
                if is_playlist_not_found(exc):
                    playlist_id = refresh_playlist(youtube, title, playlist_cache)
                    if playlist_id:
                        add_video_to_playlist(youtube, playlist_id, video_id)
                else:
                    raise


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
        links = merge_links(links, extraer_links_desde_data(band_data))

    endpoints = []
    if not band_data:
        endpoints.append(f"/bands/{band_id}")
    endpoints.extend((f"/bands/{band_id}/links", f"/bands/{band_id}/discography"))

    for endpoint in endpoints:
        data = api_get(session, endpoint)
        if not data:
            continue
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
    default_csv = Path("/home/banar/Desktop/scrapper-deathgrind/data/bandas_completo.csv")
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


def construir_descripcion(genre, year, links, tracklist, country=None):
    lines = []
    lines.append(f"Genre: {format_genre_text(genre)}")
    if country:
        lines.append(f"Country: {country}")
    lines.append(f"Year: {year or 'Unknown'}")
    lines.append("")
    stream_links = [f"{SERVICE_LABELS[key]}: {links.get('stream', {}).get(key)}"
                    for key in STREAM_SERVICES if links.get("stream", {}).get(key)]
    follow_links = [f"{SERVICE_LABELS[key]}: {links.get('follow', {}).get(key)}"
                    for key in FOLLOW_SERVICES if links.get("follow", {}).get(key)]

    if stream_links:
        lines.append("Stream/Download:")
        lines.extend(stream_links)
        lines.append("")

    if follow_links:
        lines.append("Follow:")
        lines.extend(follow_links)
        lines.append("")

    lines.append("Tracklist:")
    lines.append("")
    lines.extend(tracklist)
    return "\n".join(lines)


def encontrar_video(folder_path):
    candidates = [path for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"]
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
)


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
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True, chunksize=chunk_size)
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )
    response = None
    retry = 0
    max_retries = int(os.environ.get("YOUTUBE_UPLOAD_RETRIES", "8"))
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
            if exc.resp is not None and exc.resp.status in RETRIABLE_STATUS_CODES:
                retry += 1
                if retry > max_retries:
                    raise
                wait_time = min(60, 2 ** retry)
                print(f"Error temporal al subir ({exc.resp.status}). Reintentando en {wait_time}s...")
                time.sleep(wait_time)
                continue
            raise
        except RETRIABLE_EXCEPTIONS as exc:
            if last_percent >= 0:
                sys.stdout.write("\n")
                sys.stdout.flush()
                last_percent = -1
            retry += 1
            if retry > max_retries:
                raise
            wait_time = min(60, 2 ** retry)
            print(f"Error de conexion al subir ({exc}). Reintentando en {wait_time}s...")
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
        post = buscar_release_en_api(session, band, album, generos, allow_genre_fallback)
    elif release is not None:
        post = release

    release_tipo = None
    release_year = None
    genre_text = None
    post_id = None
    band_id = None
    post_detail = None

    if post:
        release_tipo = elegir_tipo(post.get("type") or post.get("type_id"))
        release_year = extraer_anio(post)
        genre_text = extraer_genero(post, generos_lookup)
        post_id = post.get("postId") or post.get("post_id") or post.get("id")
        band_id = post.get("bandId") or post.get("band_id") or obtener_band_id(post, band)

    if session and post_id and (not band_id or not release_tipo or not release_year or not genre_text):
        post_detail = api_get(session, f"/posts/{post_id}")
        if post_detail:
            if not band_id:
                band_id = post_detail.get("bandId") or post_detail.get("band_id") or obtener_band_id(post_detail, band)
            if not release_tipo:
                release_tipo = elegir_tipo(post_detail.get("type") or post_detail.get("type_id"))
            if not release_year:
                release_year = extraer_anio(post_detail)
            if not genre_text:
                genre_text = extraer_genero(post_detail, generos_lookup)
            post = post_detail

    if not release_year:
        release_year = extraer_anio_de_texto(folder_path.name) or context["year"]
    if not genre_text:
        genre_text = context["genre"]

    tipo_full = formatear_tipo_full(release_tipo)
    title = f"{band} - {album} ({tipo_full})"

    band_data = None
    if session and band_id:
        band_data = api_get(session, f"/bands/{band_id}")

    pais = None
    if band_data:
        pais = extraer_pais_desde_data(band_data, band_id=band_id, band_name=band)
    if not pais:
        pais = extraer_pais_desde_data(post_detail or post, band_id=band_id, band_name=band)

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
            print(f"Tracklist desde API: {len(track_titles)} pistas en {folder_path.name}")
        else:
            print(f"No se encontro tracklist en la API para {folder_path.name}, usando audio.")

    tracklist = build_tracklist(tracks, track_titles if track_titles else None)
    description = construir_descripcion(genre_text, release_year, links, tracklist, country=pais)
    genres_list = split_genres(genre_text)

    return {
        "title": title,
        "description": description,
        "band": band,
        "country": pais,
        "genres": genres_list,
    }


def main():
    parser = argparse.ArgumentParser(description="Subir videos con API de YouTube y descripcion automatizada.")
    parser.add_argument("--carpeta", help="Carpeta especifica dentro de upload_video")
    parser.add_argument("--limite", type=int, default=1, help="Cantidad maxima de videos a subir (default: 1)")
    parser.add_argument("--todo", action="store_true", help="Procesar todas las carpetas disponibles")
    parser.add_argument("--con-generos", action="store_true", help="Hacer busqueda completa por generos (mas lenta)")
    parser.add_argument("--sin-deathgrind", action="store_true", help="No usar la API de DeathGrind")
    parser.add_argument("--sin-links-web", action="store_true", help="(Deprecated) Ya no se usa")
    parser.add_argument("--buscar-links", action="store_true", help="(Deprecated) Ya no se usa")
    parser.add_argument("--sin-verificacion", action="store_true", help="No abrir verificacion manual en YouTube")
    parser.add_argument("--privacidad", choices=["public", "private", "unlisted"], default="public")
    parser.add_argument("--publicar-ahora", action="store_true", help="(Deprecated) Ya se publica al momento")
    parser.add_argument("--modo-inmediato", action="store_true", help="Subir videos sin programar en lote")
    parser.add_argument("--cantidad-lote", type=int, help="Cantidad de videos para programar en lote (default: 12)")
    parser.add_argument("--gap-horas", type=int, help="Horas de diferencia entre publicaciones (default: 2)")
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

    youtube = authenticate()
    playlist_cache = {}
    try:
        playlist_cache = list_playlists(youtube)
    except HttpError as exc:
        print(f"No se pudo leer playlists existentes: {exc}")
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
    batch_size = args.cantidad_lote if args.cantidad_lote else int(os.environ.get("YOUTUBE_BATCH_SIZE", "12"))
    gap_hours = args.gap_horas if args.gap_horas else int(os.environ.get("YOUTUBE_BATCH_GAP_HOURS", "2"))
    if batch_size <= 0:
        batch_size = 1
    if gap_hours <= 0:
        gap_hours = 2
    pending_items = []
    queued_names = set()

    while True:
        if args.carpeta:
            folders = [upload_dir] if upload_dir.exists() else []
        else:
            folders = [path for path in upload_dir.iterdir() if path.is_dir()]
            random.shuffle(folders)

        if not folders:
            if args.carpeta:
                print("No se encontro la carpeta especificada.")
                return
            print(f"No hay carpetas para subir. Esperando {scan_interval}s...")
            time.sleep(scan_interval)
            continue

        if batch_mode:
            for folder_path in folders:
                if len(pending_items) >= batch_size:
                    break
                if folder_path.name in queued_names:
                    continue
                video_path = encontrar_video(folder_path)
                if not video_path:
                    print(f"No hay video .mp4 en {folder_path}, se omite.")
                    continue
                if not args.sin_verificacion:
                    if verificacion_manual(folder_path, repertorio):
                        continue
                print(f"Preparando {folder_path.name} para programar...")
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
                pending_items.append({
                    "folder": folder_path,
                    "video": video_path,
                    "metadata": metadata,
                })
                queued_names.add(folder_path.name)
                print(f"En cola para programar: {folder_path.name} ({len(pending_items)}/{batch_size})")

            if len(pending_items) < batch_size:
                if args.carpeta:
                    print(f"No se juntaron {batch_size} videos para programar.")
                    break
                faltan = batch_size - len(pending_items)
                print(f"Aun faltan {faltan} videos para programar. Esperando {scan_interval}s...")
                time.sleep(scan_interval)
                continue

            tz_local = datetime.now().astimezone().tzinfo
            schedule_date = (datetime.now(tz_local) + timedelta(days=1)).date()
            taken_slots, counts_by_date = fetch_scheduled_publish_times_safe(youtube)
            max_per_day = max_videos_por_dia(gap_hours)
            if batch_size > max_per_day:
                print(f"No se puede programar {batch_size} videos con separacion de {gap_hours}h.")
                break
            schedule_date, slots = find_next_batch_schedule(
                schedule_date,
                batch_size,
                gap_hours,
                taken_slots,
                counts_by_date,
                tz_local,
            )
            if not slots or len(slots) < batch_size:
                print("No se pudo generar el horario de programacion para el lote.")
                break

            existentes = counts_by_date.get(schedule_date, 0)
            if existentes:
                print(f"Ya hay {existentes} videos programados para {schedule_date.isoformat()}, se programa el lote ahi.")
            random.shuffle(pending_items)
            random.shuffle(slots)
            print(f"Programando {batch_size} videos para {schedule_date.isoformat()} con separacion de {gap_hours}h.")
            for item, slot in zip(pending_items, slots):
                folder_path = item["folder"]
                metadata = item["metadata"]
                publish_at = format_rfc3339(slot)
                try:
                    response = subir_video(
                        youtube,
                        item["video"],
                        metadata["title"],
                        metadata["description"],
                        privacy=args.privacidad,
                        publish_at=publish_at,
                        label=metadata.get("title"),
                    )
                    print(f"Video subido: {response.get('id')} (programado: {publish_at})")
                    try:
                        add_video_to_playlists(
                            youtube,
                            response.get("id"),
                            metadata.get("band"),
                            metadata.get("genres", []),
                            playlist_cache,
                        )
                    except HttpError as exc:
                        print(f"No se pudo agregar a playlists: {exc}")
                    destino = mover_carpeta_subida(folder_path, DIR_YA_SUBIDOS)
                    if destino:
                        print(f"Carpeta movida a {destino}")
                except HttpError as exc:
                    print(f"Error YouTube API en {folder_path.name}: {exc}")
            break

        uploaded = 0
        for folder_path in folders:
            if max_uploads is not None and uploaded >= max_uploads:
                break
            video_path = encontrar_video(folder_path)
            if not video_path:
                print(f"No hay video .mp4 en {folder_path}, se omite.")
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
            try:
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
                try:
                    add_video_to_playlists(
                        youtube,
                        response.get("id"),
                        metadata.get("band"),
                        metadata.get("genres", []),
                        playlist_cache,
                    )
                except HttpError as exc:
                    print(f"No se pudo agregar a playlists: {exc}")
                destino = mover_carpeta_subida(folder_path, DIR_YA_SUBIDOS)
                if destino:
                    print(f"Carpeta movida a {destino}")
            except HttpError as exc:
                print(f"Error YouTube API en {folder_path.name}: {exc}")

        if args.carpeta:
            break


if __name__ == "__main__":
    main()
