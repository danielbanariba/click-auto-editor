#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import os
import re
import time
import uuid
from pathlib import Path

import requests
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from mutagen import File as MutagenFile

from config import AUDIO_FORMATS, DIR_UPLOAD
from subir_video.authenticate import authenticate

BASE_URL = "https://deathgrind.club"
API_URL = f"{BASE_URL}/api"
INTRO_SECONDS = 8

DELAY_ENTRE_PAGINAS = 1.0
DELAY_BASE_429 = 30
MAX_RETRIES_ERROR = 5
MAX_RETRIES_429 = int(os.environ.get("DEATHGRIND_MAX_429", "1"))
DEATHGRIND_DISABLED = False

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
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                genre_id = item.get("id") or item.get("genreId")
                name = item.get("name") or item.get("genre")
                if not name and genre_id in generos_lookup:
                    name = generos_lookup.get(genre_id)
                if genre_id and name:
                    generos.append(f"({genre_id}){name}")
                elif name:
                    generos.append(str(name))
                elif genre_id and genre_id in generos_lookup:
                    generos.append(f"({genre_id}){generos_lookup.get(genre_id)}")
            else:
                if isinstance(item, int) and item in generos_lookup:
                    generos.append(f"({item}){generos_lookup[item]}")
                else:
                    generos.append(str(item))
    elif isinstance(raw, str):
        generos.append(raw)
    return ", ".join(generos) if generos else None


def extraer_anio(post):
    release_date = post.get("releaseDate") or post.get("year")
    if isinstance(release_date, list) and release_date:
        return release_date[0]
    if isinstance(release_date, (str, int)):
        return release_date
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
                try:
                    type_id = int(type_id) if type_id else None
                except ValueError:
                    type_id = None
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


def merge_links(base, new_links):
    for category in ("stream", "follow"):
        base.setdefault(category, {})
        for key, value in new_links.get(category, {}).items():
            if key not in base[category]:
                base[category][key] = value
    return base


def extraer_links_deathgrind(session, post_id):
    links = {"stream": {}, "follow": {}}
    if DEATHGRIND_DISABLED:
        return links
    if not session or not post_id:
        return links
    data = api_get(session, f"/posts/{post_id}/links")
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


def parse_band_album_from_folder(folder_name):
    if " - " in folder_name:
        band, album = folder_name.split(" - ", 1)
        return band.strip(), album.strip()
    return folder_name, folder_name


def construir_descripcion(genre, year, links, tracklist):
    lines = []
    lines.append(f"Genre: {genre or 'Unknown'}")
    lines.append(f"Year: {year or 'Unknown'}")
    lines.append("")
    lines.append("Stream/Download:")
    for key in STREAM_SERVICES:
        url = links.get("stream", {}).get(key)
        if url:
            lines.append(f"{SERVICE_LABELS[key]}: {url}")
    lines.append("")
    lines.append("Follow:")
    for key in FOLLOW_SERVICES:
        url = links.get("follow", {}).get(key)
        if url:
            lines.append(f"{SERVICE_LABELS[key]}: {url}")
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


def subir_video(youtube, video_path, title, description, privacy="private", category_id="10"):
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
        },
    }

    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )
    response = request.execute()
    return response


def procesar_carpeta(folder_path, session, generos, generos_lookup, repertorio, allow_genre_fallback=False):
    tracks, context = collect_audio_tracks(folder_path)
    if not tracks:
        print(f"No hay audios en {folder_path}, se omite.")
        return None

    band, album = parse_band_album_from_folder(folder_path.name)
    if context["band"]:
        band = context["band"]
    if context["album"]:
        album = context["album"]

    csv_path = os.environ.get("DEATHGRIND_REPERTORIO_CSV")
    if not csv_path:
        default_csv = Path("/home/banar/Desktop/scrapper-deathgrind/data/bandas_completo.csv")
        if default_csv.exists():
            csv_path = str(default_csv)

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

    if post:
        release_tipo = elegir_tipo(post.get("type") or post.get("type_id"))
        release_year = extraer_anio(post)
        genre_text = extraer_genero(post, generos_lookup)
        post_id = post.get("postId") or post.get("post_id") or post.get("id")

    if not release_year:
        release_year = context["year"]
    if not genre_text:
        genre_text = context["genre"]

    tipo_full = formatear_tipo_full(release_tipo)
    title = f"{band} - {album} ({tipo_full})"

    links = {"stream": {}, "follow": {}}
    if session and post_id:
        links = extraer_links_deathgrind(session, post_id)

    track_titles = []
    if post:
        track_titles = obtener_tracklist_api(session, post, post_id)
        if track_titles:
            print(f"Tracklist desde API: {len(track_titles)} pistas en {folder_path.name}")
        else:
            print(f"No se encontro tracklist en la API para {folder_path.name}, usando audio.")

    tracklist = build_tracklist(tracks, track_titles if track_titles else None)
    description = construir_descripcion(genre_text, release_year, links, tracklist)

    return {
        "title": title,
        "description": description,
    }


def main():
    parser = argparse.ArgumentParser(description="Subir videos con API de YouTube y descripcion automatizada.")
    parser.add_argument("--carpeta", help="Carpeta especifica dentro de upload_video")
    parser.add_argument("--limite", type=int, default=1, help="Cantidad maxima de videos a subir (default: 1)")
    parser.add_argument("--todo", action="store_true", help="Procesar todas las carpetas disponibles")
    parser.add_argument("--con-generos", action="store_true", help="Hacer busqueda completa por generos (mas lenta)")
    parser.add_argument("--sin-deathgrind", action="store_true", help="No usar la API de DeathGrind")
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

    if args.carpeta:
        folders = [upload_dir]
    else:
        folders = [path for path in upload_dir.iterdir() if path.is_dir()]
        if not args.todo and args.limite:
            folders = folders[:args.limite]

    if not folders:
        print("No se encontraron carpetas para subir.")
        return

    allow_genre_fallback = args.con_generos

    for folder_path in folders:
        video_path = encontrar_video(folder_path)
        if not video_path:
            print(f"No hay video .mp4 en {folder_path}, se omite.")
            continue
        print(f"Procesando {folder_path.name}...")
        metadata = procesar_carpeta(folder_path, session, generos, generos_lookup, repertorio, allow_genre_fallback)
        if not metadata:
            continue
        try:
            response = subir_video(
                youtube,
                video_path,
                metadata["title"],
                metadata["description"],
            )
            print(f"Video subido: {response.get('id')}")
        except HttpError as exc:
            print(f"Error YouTube API en {folder_path.name}: {exc}")


if __name__ == "__main__":
    main()
