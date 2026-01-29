#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

from googleapiclient.errors import HttpError

from subir_video.authenticate import authenticate

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "mapear_playlists_checkpoint.txt"
DEFAULT_STATE = Path(__file__).resolve().parent / "mapear_playlists_state.json"


class RateLimitError(RuntimeError):
    pass


class QuotaExceededError(RuntimeError):
    pass


def get_uploads_playlist_id(youtube):
    response = run_with_backoff(
        lambda: youtube.channels().list(part="contentDetails", mine=True).execute(),
        descripcion="listar canal",
    )
    items = response.get("items", [])
    if not items:
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def clean_title(value):
    if value is None:
        return None
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def normalize_title(value):
    text = clean_title(value)
    return text.casefold() if text else None


def list_playlists(youtube):
    playlists = {}
    duplicadas = {}
    candidatos = {}
    todas = []
    page_token = None
    while True:
        try:
            resp = run_with_backoff(
                lambda: youtube.playlists().list(
                    part="snippet,contentDetails",
                    mine=True,
                    maxResults=50,
                    pageToken=page_token,
                ).execute(),
                descripcion="listar playlists",
            )
        except QuotaExceededError:
            print("Cuota diaria agotada al listar playlists. Se detiene la ejecucion.")
            raise
        except RateLimitError:
            print("Rate limit al listar playlists. Se detiene la ejecucion.")
            raise
        for item in resp.get("items", []):
            title = item.get("snippet", {}).get("title")
            title_clean = clean_title(title)
            if not title_clean:
                continue
            key = normalize_title(title_clean)
            playlist_id = item.get("id")
            if not key or not playlist_id:
                continue
            info = {
                "id": playlist_id,
                "title": title_clean,
                "count": item.get("contentDetails", {}).get("itemCount", 0) or 0,
                "published": item.get("snippet", {}).get("publishedAt") or "",
            }
            candidatos.setdefault(key, []).append(info)
            todas.append(info)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    for key, items in candidatos.items():
        if not items:
            continue
        items_sorted = sorted(
            items,
            key=lambda info: (
                -(info.get("count") or 0),
                info.get("published") or "",
                info.get("id") or "",
            ),
        )
        canonical = items_sorted[0]
        playlists[key] = canonical["id"]
        if len(items_sorted) > 1:
            duplicadas[key] = {"canonical": canonical, "extras": items_sorted[1:]}
    if duplicadas:
        for data in duplicadas.values():
            nombre = data["canonical"]["title"]
            print(
                f"Aviso: playlists duplicadas detectadas para '{nombre}'. "
                "Se usara la que tenga mas videos."
            )
    return playlists, duplicadas, todas


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


def is_video_not_found(exc):
    if not isinstance(exc, HttpError):
        return False
    if getattr(exc, "resp", None) and exc.resp.status == 404:
        try:
            data = json.loads(exc.content.decode("utf-8"))
            for err in data.get("error", {}).get("errors", []):
                if err.get("reason") == "videoNotFound":
                    return True
        except Exception:
            pass
    try:
        data = json.loads(exc.content.decode("utf-8"))
        for err in data.get("error", {}).get("errors", []):
            if err.get("reason") == "videoNotFound":
                return True
    except Exception:
        pass
    return False


def is_rate_limit_error(exc):
    if not isinstance(exc, HttpError):
        return False
    if getattr(exc, "resp", None) and exc.resp.status not in (403, 429):
        return False
    try:
        data = json.loads(exc.content.decode("utf-8"))
        reasons = [err.get("reason") for err in data.get("error", {}).get("errors", [])]
        for reason in reasons:
            if reason in {
                "rateLimitExceeded",
                "userRateLimitExceeded",
                "quotaExceeded",
                "dailyLimitExceeded",
                "RATE_LIMIT_EXCEEDED",
            }:
                return True
    except Exception:
        pass
    return False


def get_error_reasons(exc):
    if not isinstance(exc, HttpError):
        return []
    try:
        data = json.loads(exc.content.decode("utf-8"))
        return [
            err.get("reason")
            for err in data.get("error", {}).get("errors", [])
            if err.get("reason")
        ]
    except Exception:
        return []


def is_daily_quota_error(exc):
    reasons = get_error_reasons(exc)
    return any(reason in {"quotaExceeded", "dailyLimitExceeded"} for reason in reasons)


def run_with_backoff(action, descripcion, max_retries=5, base_delay=1.0, max_delay=60.0):
    attempt = 0
    while True:
        try:
            return action()
        except HttpError as exc:
            if not is_rate_limit_error(exc):
                raise
            if is_daily_quota_error(exc):
                raise QuotaExceededError(descripcion)
            attempt += 1
            if attempt >= max_retries:
                raise RateLimitError(descripcion)
            wait = min(max_delay, base_delay * (2 ** (attempt - 1)))
            print(
                "Rate limit al intentar "
                + descripcion
                + f". Esperando {wait:.1f}s y reintentando ({attempt}/{max_retries})."
            )
            time.sleep(wait)


def get_or_create_playlist(youtube, title, cache, privacy="public", pause=None):
    title_clean = clean_title(title)
    if not title_clean:
        return None
    key = normalize_title(title_clean)
    if not key:
        return None
    playlist_id = cache.get(key)
    if playlist_id:
        return playlist_id
    body = {
        "snippet": {"title": title_clean},
        "status": {"privacyStatus": privacy},
    }
    resp = run_with_backoff(
        lambda: youtube.playlists().insert(part="snippet,status", body=body).execute(),
        descripcion=f"crear playlist '{title}'",
    )
    playlist_id = resp.get("id")
    if playlist_id:
        cache[key] = playlist_id
        if pause:
            time.sleep(pause)
    return playlist_id


def refresh_playlist(youtube, title, cache, privacy="public"):
    key = normalize_title(title)
    cache.pop(key, None)
    return get_or_create_playlist(youtube, title, cache, privacy=privacy)


def get_playlist_video_ids(youtube, playlist_id, max_items=200, cache=None):
    if cache is not None:
        cached = cache.get(playlist_id)
        if cached is not None:
            return cached, False
    video_ids = set()
    page_token = None
    fetched = 0
    truncated = False
    while True:
        resp = run_with_backoff(
            lambda: youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=page_token,
            ).execute(),
            descripcion="listar items de playlist",
        )
        items = resp.get("items", [])
        for item in items:
            resource = item.get("snippet", {}).get("resourceId", {})
            item_id = resource.get("videoId")
            if item_id:
                video_ids.add(item_id)
        fetched += len(items)
        if max_items is not None and fetched >= max_items:
            truncated = True
            break
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    if cache is not None:
        cache[playlist_id] = video_ids
    return video_ids, truncated


def is_video_in_playlist(youtube, playlist_id, video_id, max_items=200, cache=None):
    if cache is not None:
        video_ids, _ = get_playlist_video_ids(
            youtube,
            playlist_id,
            max_items=max_items,
            cache=cache,
        )
        return video_id in video_ids

    page_token = None
    fetched = 0
    while True:
        resp = run_with_backoff(
            lambda: youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=page_token,
            ).execute(),
            descripcion="listar items de playlist",
        )
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
    run_with_backoff(
        lambda: youtube.playlistItems().insert(part="snippet", body=body).execute(),
        descripcion=f"agregar video {video_id} a la playlist",
    )


def set_playlist_privacy(youtube, playlist_id, privacy="private"):
    if not playlist_id:
        return
    body = {"id": playlist_id, "status": {"privacyStatus": privacy}}
    run_with_backoff(
        lambda: youtube.playlists().update(part="status", body=body).execute(),
        descripcion=f"actualizar privacidad de playlist {playlist_id}",
    )


def delete_playlist(youtube, playlist_id):
    if not playlist_id:
        return
    run_with_backoff(
        lambda: youtube.playlists().delete(id=playlist_id).execute(),
        descripcion=f"eliminar playlist {playlist_id}",
    )


def merge_duplicate_playlists(
    youtube,
    canonical_id,
    duplicate_id,
    max_items=200,
    cache=None,
    pause=None,
):
    if not canonical_id or not duplicate_id or canonical_id == duplicate_id:
        return 0, False
    canonical_ids, canonical_trunc = get_playlist_video_ids(
        youtube,
        canonical_id,
        max_items=max_items,
        cache=cache,
    )
    duplicate_ids, duplicate_trunc = get_playlist_video_ids(
        youtube,
        duplicate_id,
        max_items=max_items,
        cache=cache,
    )
    nuevos = duplicate_ids - canonical_ids
    added = 0
    skipped_missing = 0
    for video_id in nuevos:
        try:
            add_video_to_playlist(youtube, canonical_id, video_id)
            canonical_ids.add(video_id)
            added += 1
            if pause:
                time.sleep(pause)
        except HttpError as exc:
            if is_video_not_found(exc):
                skipped_missing += 1
                continue
            raise
    if skipped_missing:
        print(
            f"Aviso: {skipped_missing} videos no encontrados al fusionar playlists."
        )
    return added, canonical_trunc or duplicate_trunc


def auto_cleanup_playlists(
    youtube,
    duplicadas,
    todas_playlists,
    playlist_cache,
    playlist_items_cache,
    pause=None,
    merge_scan=None,
):
    if not duplicadas and not todas_playlists:
        return
    info_por_id = {item["id"]: item for item in todas_playlists}
    duplicadas_ids = set()
    canon_ids = set()
    grupos_con_videos = set()

    for data in duplicadas.values():
        canonical = data["canonical"]
        canon_ids.add(canonical["id"])
        group_has_videos = (canonical.get("count") or 0) > 0
        for extra in data["extras"]:
            duplicadas_ids.add(extra["id"])
            if (extra.get("count") or 0) > 0:
                group_has_videos = True
        if group_has_videos:
            grupos_con_videos.add(canonical["id"])

    if duplicadas:
        print(f"Limpieza: fusionando {len(duplicadas)} titulos duplicados.")
        for data in duplicadas.values():
            canonical = data["canonical"]
            truncado = False
            for extra in data["extras"]:
                added_count, was_truncated = merge_duplicate_playlists(
                    youtube,
                    canonical["id"],
                    extra["id"],
                    max_items=merge_scan,
                    cache=playlist_items_cache,
                    pause=pause,
                )
                truncado = truncado or was_truncated
                msg = (
                    f"Fusion: '{extra['title']}' -> '{canonical['title']}'. "
                    f"Agregados: {added_count}."
                )
                if was_truncated:
                    msg += " Aviso: fusion truncada."
                print(msg)
            if truncado:
                print(
                    "Aviso: no se eliminaran duplicadas de '"
                    + canonical["title"]
                    + "' por fusion incompleta."
                )
                continue
            for extra in data["extras"]:
                titulo = extra.get("title") or extra["id"]
                count = extra.get("count", 0) or 0
                if count:
                    print(f"Eliminando duplicada: '{titulo}' ({count} videos).")
                else:
                    print(f"Eliminando duplicada: '{titulo}'.")
                delete_playlist(youtube, extra["id"])
                if playlist_items_cache is not None:
                    playlist_items_cache.pop(extra["id"], None)
                key = normalize_title(titulo)
                if key and playlist_cache.get(key) == extra["id"]:
                    playlist_cache.pop(key, None)

    vacias_ids = []
    for item in todas_playlists:
        if (item.get("count") or 0) != 0:
            continue
        playlist_id = item["id"]
        if playlist_id in duplicadas_ids:
            continue
        if playlist_id in grupos_con_videos:
            continue
        vacias_ids.append(playlist_id)

    if vacias_ids:
        print(f"Limpieza: eliminando playlists vacias: {len(vacias_ids)}.")
        for playlist_id in vacias_ids:
            info = info_por_id.get(playlist_id, {})
            titulo = info.get("title") or playlist_id
            print(f"Eliminando vacia: '{titulo}'.")
            delete_playlist(youtube, playlist_id)
            if playlist_items_cache is not None:
                playlist_items_cache.pop(playlist_id, None)
            key = normalize_title(titulo)
            if key and playlist_cache.get(key) == playlist_id:
                playlist_cache.pop(key, None)


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
        item = clean_title(item)
        if not item:
            continue
        key = normalize_title(item)
        if key == "unknown":
            continue
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


def parse_band_from_title(title):
    if not title:
        return None
    if " - " in title:
        return title.split(" - ", 1)[0].strip()
    return None


def parse_genres_from_description(description):
    if not description:
        return []
    for line in description.splitlines():
        text = line.strip()
        if not text:
            continue
        lower = text.lower()
        if lower.startswith("genre:") or lower.startswith("genero:"):
            value = text.split(":", 1)[1].strip()
            return split_genres(value)
    return []


def split_countries(value):
    if not value:
        return []
    if isinstance(value, list):
        parts = value
    else:
        parts = re.split(r"[,/;]+", str(value))
    cleaned = []
    seen = set()
    for part in parts:
        item = clean_title(part)
        if not item:
            continue
        key = normalize_title(item)
        if key in {"unknown", "desconocido", "n/a", "na"}:
            continue
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


def parse_countries_from_description(description):
    if not description:
        return []
    for line in description.splitlines():
        text = line.strip()
        if not text:
            continue
        lower = text.lower()
        if (
            lower.startswith("country:")
            or lower.startswith("contry:")
            or lower.startswith("pais:")
            or lower.startswith("pa\u00eds:")
        ):
            value = text.split(":", 1)[1].strip()
            return split_countries(value)
    return []


def split_years(value):
    if not value:
        return []
    text = str(value)
    years = re.findall(r"\b(?:18|19|20)\d{2}\b", text)
    cleaned = []
    seen = set()
    if years:
        for year in years:
            if year in seen:
                continue
            seen.add(year)
            cleaned.append(year)
        return cleaned
    parts = re.split(r"[,/;]+", text)
    for part in parts:
        item = clean_title(part)
        if not item:
            continue
        key = normalize_title(item)
        if key in {"unknown", "desconocido", "n/a", "na"}:
            continue
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


def parse_years_from_description(description):
    if not description:
        return []
    for line in description.splitlines():
        text = line.strip()
        if not text:
            continue
        lower = text.lower()
        if (
            lower.startswith("year:")
            or lower.startswith("anio:")
            or lower.startswith("a\u00f1o:")
        ):
            value = text.split(":", 1)[1].strip()
            return split_years(value)
    return []


def chunked(items, size):
    for index in range(0, len(items), size):
        yield items[index:index + size]


def iter_videos_by_ids(youtube, video_ids, include_unlisted=False):
    for chunk in chunked(video_ids, 50):
        if not chunk:
            continue
        try:
            vids = run_with_backoff(
                lambda: youtube.videos().list(
                    part="snippet,status",
                    id=",".join(chunk),
                ).execute(),
                descripcion="listar detalles de videos",
            )
        except QuotaExceededError:
            print("Cuota diaria agotada al listar videos. Se detiene la ejecucion.")
            raise
        except RateLimitError:
            print("Rate limit al listar videos. Se detiene la ejecucion.")
            raise
        for video in vids.get("items", []):
            status = video.get("status", {})
            privacy = status.get("privacyStatus") or "public"
            if privacy != "public" and not (include_unlisted and privacy == "unlisted"):
                continue
            yield video


def iter_public_videos(youtube, limit=None, include_unlisted=False, stop_at_video_id=None):
    playlist_id = get_uploads_playlist_id(youtube)
    state = {"first_video_id": None, "completed": False, "stopped_at": False, "limited": False}
    if not playlist_id:
        return iter(()), state

    def generator():
        nonlocal state
        page_token = None
        processed = 0
        while True:
            try:
                resp = run_with_backoff(
                    lambda: youtube.playlistItems().list(
                        part="snippet,status",
                        playlistId=playlist_id,
                        maxResults=50,
                        pageToken=page_token,
                    ).execute(),
                    descripcion="listar uploads",
                )
            except QuotaExceededError:
                print("Cuota diaria agotada al listar uploads. Se detiene la ejecucion.")
                raise
            except RateLimitError:
                print("Rate limit al listar uploads. Se detiene la ejecucion.")
                raise
            items = resp.get("items", [])
            for item in items:
                snippet = item.get("snippet", {})
                resource = snippet.get("resourceId", {})
                video_id = resource.get("videoId")
                if not video_id:
                    continue
                if state["first_video_id"] is None:
                    state["first_video_id"] = video_id
                if stop_at_video_id and video_id == stop_at_video_id:
                    state["stopped_at"] = True
                    return
                status = item.get("status", {})
                privacy = status.get("privacyStatus") or "public"
                if privacy != "public" and not (include_unlisted and privacy == "unlisted"):
                    continue
                yield {
                    "id": video_id,
                    "snippet": {
                        "title": snippet.get("title") or "",
                        "description": snippet.get("description") or "",
                    },
                    "status": {"privacyStatus": privacy},
                }
                processed += 1
                if limit is not None and processed >= limit:
                    state["limited"] = True
                    return
            page_token = resp.get("nextPageToken")
            if not page_token:
                state["completed"] = True
                return

    return generator(), state


def load_checkpoint(path):
    processed = {}
    if not path or not path.exists():
        return processed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            parts = text.split("\t", 2)
            video_id = parts[0].strip()
            if not video_id:
                continue
            status = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "ok"
            processed[video_id] = status
    return processed


def load_state(path):
    if not path or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(path, state):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=True, indent=2)


def append_checkpoint(path, video_id, status="ok"):
    if not path or not video_id:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{video_id}\t{status}\n")


def update_checkpoint(path, processed, video_id, status):
    if not path or not video_id:
        return False
    if processed.get(video_id) == status:
        return False
    append_checkpoint(path, video_id, status=status)
    processed[video_id] = status
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Mapea videos publicos a playlists por banda, genero, ano y pais."
    )
    parser.add_argument("--limite", type=int, help="Cantidad maxima de videos a procesar")
    parser.add_argument("--sin-check", action="store_true", help="No validar si ya existe en playlist")
    parser.add_argument("--incluir-no-listado", action="store_true", help="Incluye videos no listados")
    parser.add_argument("--pausa", type=float, help="Segundos de espera entre operaciones")
    parser.add_argument("--max-scan", type=int, help="Maximo de items a escanear por playlist")
    parser.add_argument("--solo-bandas", action="store_true", help="Solo playlists por banda")
    parser.add_argument(
        "--solo-generos",
        action="store_true",
        help="Solo playlists por genero/ano/pais",
    )
    parser.add_argument(
        "--sin-limpieza",
        action="store_true",
        help="No limpiar playlists duplicadas o vacias.",
    )
    args = parser.parse_args()

    pause = args.pausa if args.pausa is not None else float(os.environ.get("YOUTUBE_PLAYLIST_DELAY", "0.2"))
    create_pause = max(pause or 0.0, 1.0)
    max_scan = args.max_scan if args.max_scan is not None else int(os.environ.get("YOUTUBE_PLAYLIST_SCAN", "200"))
    if args.solo_bandas and args.solo_generos:
        print("No se puede usar --solo-bandas y --solo-generos al mismo tiempo.")
        return

    checkpoint_path = DEFAULT_CHECKPOINT
    processed = {}
    try:
        processed = load_checkpoint(checkpoint_path)
        if processed:
            print(f"Checkpoint cargado: {len(processed)} videos.")
    except OSError as exc:
        print(f"No se pudo leer el checkpoint: {exc}. Se continua sin checkpoint.")
        checkpoint_path = None
        processed = {}

    state_path = DEFAULT_STATE
    state = load_state(state_path)
    last_full_scan = bool(state.get("last_full_scan_completed"))
    stop_at_video_id = state.get("last_newest_video_id") if last_full_scan else None

    youtube = authenticate()
    playlist_cache, duplicadas, todas_playlists = list_playlists(youtube)
    playlist_items_cache = {}
    auto_limpieza = not args.sin_limpieza and os.environ.get("YOUTUBE_AUTO_CLEAN", "1") == "1"
    if auto_limpieza:
        print("Limpieza automatica: fusionar duplicadas y eliminar vacias/duplicadas.")
        auto_cleanup_playlists(
            youtube,
            duplicadas,
            todas_playlists,
            playlist_cache,
            playlist_items_cache,
            pause=pause,
            merge_scan=None,
        )
    crear_playlists_bloqueado = False

    total = 0
    added = 0
    skipped_no_data = 0
    skipped_checkpoint = 0
    checkpointed = 0

    pending_ids = [
        video_id for video_id, status in processed.items() if status == "pendiente"
    ]
    seen_video_ids = set()

    def procesar_video(video):
        nonlocal total, added, skipped_no_data, skipped_checkpoint, checkpointed, crear_playlists_bloqueado
        video_id = video.get("id")
        if not video_id:
            return
        if video_id in seen_video_ids:
            return
        seen_video_ids.add(video_id)

        snippet = video.get("snippet", {})
        title = clean_title(snippet.get("title") or "") or ""
        description = snippet.get("description") or ""

        if checkpoint_path and video_id in processed and processed.get(video_id) in {"ok", "sin_datos"}:
            skipped_checkpoint += 1
            print(f"Omitido por checkpoint: {title} ({video_id})")
            return

        video_pendiente = False

        band = parse_band_from_title(title)
        genres = parse_genres_from_description(description)
        countries = parse_countries_from_description(description)
        years = parse_years_from_description(description)
        playlist_titles = []
        if band and band.lower() != "unknown" and not args.solo_generos:
            playlist_titles.append(band)
        if genres and not args.solo_bandas:
            playlist_titles.extend(genres)
        if countries and not args.solo_bandas:
            playlist_titles.extend(countries)
        if years and not args.solo_bandas:
            playlist_titles.extend(years)

        if playlist_titles:
            seen = set()
            unique_titles = []
            for item in playlist_titles:
                cleaned = clean_title(item)
                key = normalize_title(cleaned)
                if not key or key in seen:
                    continue
                seen.add(key)
                unique_titles.append(cleaned)
            playlist_titles = unique_titles

        if not playlist_titles:
            skipped_no_data += 1
            print(f"Sin banda/genero/ano/pais: {title} ({video_id})")
            if checkpoint_path and update_checkpoint(
                checkpoint_path, processed, video_id, status="sin_datos"
            ):
                checkpointed += 1
            return

        for playlist_title in playlist_titles:
            key = normalize_title(playlist_title)
            playlist_id = playlist_cache.get(key)
            if not playlist_id:
                if crear_playlists_bloqueado:
                    print(f"Creacion de playlists en pausa. Se salta '{playlist_title}'.")
                    video_pendiente = True
                    continue
                try:
                    playlist_id = get_or_create_playlist(
                        youtube,
                        playlist_title,
                        playlist_cache,
                        pause=create_pause,
                    )
                except QuotaExceededError:
                    print("Cuota diaria agotada al crear playlists. Se detiene la ejecucion.")
                    raise
                except RateLimitError:
                    print("Rate limit al crear playlists. Se pausa la creacion por este run.")
                    crear_playlists_bloqueado = True
                    video_pendiente = True
                    continue
                except HttpError:
                    raise
            if not playlist_id:
                video_pendiente = True
                continue
            try:
                if not args.sin_check:
                    in_playlist = is_video_in_playlist(
                        youtube,
                        playlist_id,
                        video_id,
                        max_items=max_scan,
                        cache=playlist_items_cache,
                    )
                    if in_playlist:
                        continue
                add_video_to_playlist(youtube, playlist_id, video_id)
                added += 1
                if playlist_items_cache is not None:
                    cached_items = playlist_items_cache.get(playlist_id)
                    if cached_items is not None:
                        cached_items.add(video_id)
                if pause:
                    time.sleep(pause)
            except QuotaExceededError:
                print("Cuota diaria agotada al agregar videos. Se detiene la ejecucion.")
                raise
            except RateLimitError:
                print("Rate limit en playlists. Se detiene la ejecucion.")
                raise
            except HttpError as exc:
                if is_playlist_not_found(exc):
                    playlist_id = refresh_playlist(youtube, playlist_title, playlist_cache)
                    if playlist_id:
                        add_video_to_playlist(youtube, playlist_id, video_id)
                        added += 1
                        if pause:
                            time.sleep(pause)
                    continue
                print(f"No se pudo agregar {video_id} a {playlist_title}: {exc}")

        total += 1
        print(f"Procesado: {title} ({video_id})")
        if checkpoint_path:
            status = "pendiente" if video_pendiente else "ok"
            if update_checkpoint(checkpoint_path, processed, video_id, status=status):
                checkpointed += 1

    if pending_ids:
        print(f"Reintentando pendientes: {len(pending_ids)}.")
        for video in iter_videos_by_ids(
            youtube, pending_ids, include_unlisted=args.incluir_no_listado
        ):
            procesar_video(video)

    videos_iter, scan_state = iter_public_videos(
        youtube,
        limit=args.limite,
        include_unlisted=args.incluir_no_listado,
        stop_at_video_id=stop_at_video_id,
    )
    for video in videos_iter:
        procesar_video(video)

    if scan_state.get("first_video_id") and not scan_state.get("limited"):
        if scan_state.get("completed") or scan_state.get("stopped_at"):
            state["last_newest_video_id"] = scan_state.get("first_video_id")
            if scan_state.get("completed"):
                state["last_full_scan_completed"] = True
            save_state(state_path, state)

    omitted = skipped_no_data + skipped_checkpoint
    if checkpoint_path:
        print(f"Checkpoint guardado: {checkpointed} nuevos.")
    print(
        "Listo. Videos procesados: "
        f"{total}, agregados: {added}, omitidos: {omitted} "
        f"(sin datos: {skipped_no_data}, checkpoint: {skipped_checkpoint})."
    )


if __name__ == "__main__":
    auto_wait = os.environ.get("YOUTUBE_AUTO_WAIT", "0") == "1"
    quota_wait = int(os.environ.get("YOUTUBE_QUOTA_WAIT", "3600"))
    rate_wait = int(os.environ.get("YOUTUBE_RATE_WAIT", "300"))
    if not auto_wait:
        try:
            main()
        except QuotaExceededError:
            print("Cuota diaria agotada. Intenta mas tarde o activa YOUTUBE_AUTO_WAIT=1.")
        except RateLimitError:
            print("Rate limit. Intenta mas tarde o activa YOUTUBE_AUTO_WAIT=1.")
    else:
        while True:
            try:
                main()
                break
            except QuotaExceededError:
                print(f"Cuota diaria agotada. Esperando {quota_wait}s y reintentando.")
                time.sleep(quota_wait)
            except RateLimitError:
                print(f"Rate limit. Esperando {rate_wait}s y reintentando.")
                time.sleep(rate_wait)
