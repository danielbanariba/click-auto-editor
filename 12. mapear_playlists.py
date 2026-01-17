#!/usr/bin/env python3
import argparse
import json
import os
import re
import time

from googleapiclient.errors import HttpError

from subir_video.authenticate import authenticate


def get_uploads_playlist_id(youtube):
    response = youtube.channels().list(part="contentDetails", mine=True).execute()
    items = response.get("items", [])
    if not items:
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


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


def get_or_create_playlist(youtube, title, cache, privacy="public"):
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


def refresh_playlist(youtube, title, cache, privacy="public"):
    key = title.lower().strip()
    cache.pop(key, None)
    return get_or_create_playlist(youtube, title, cache, privacy=privacy)


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


def chunked(items, size):
    for index in range(0, len(items), size):
        yield items[index:index + size]


def iter_public_videos(youtube, limit=None, include_unlisted=False):
    playlist_id = get_uploads_playlist_id(youtube)
    if not playlist_id:
        return
    page_token = None
    processed = 0
    while True:
        resp = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=page_token,
        ).execute()
        items = resp.get("items", [])
        video_ids = []
        for item in items:
            resource = item.get("snippet", {}).get("resourceId", {})
            video_id = resource.get("videoId")
            if video_id:
                video_ids.append(video_id)

        for chunk in chunked(video_ids, 50):
            vids = youtube.videos().list(
                part="snippet,status",
                id=",".join(chunk),
            ).execute()
            for video in vids.get("items", []):
                status = video.get("status", {})
                privacy = status.get("privacyStatus")
                if privacy != "public" and not (include_unlisted and privacy == "unlisted"):
                    continue
                yield video
                processed += 1
                if limit is not None and processed >= limit:
                    return

        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def main():
    parser = argparse.ArgumentParser(
        description="Mapea videos publicos a playlists por banda y genero."
    )
    parser.add_argument("--limite", type=int, help="Cantidad maxima de videos a procesar")
    parser.add_argument("--sin-check", action="store_true", help="No validar si ya existe en playlist")
    parser.add_argument("--incluir-no-listado", action="store_true", help="Incluye videos no listados")
    parser.add_argument("--pausa", type=float, help="Segundos de espera entre operaciones")
    parser.add_argument("--max-scan", type=int, help="Maximo de items a escanear por playlist")
    parser.add_argument("--solo-bandas", action="store_true", help="Solo playlists por banda")
    parser.add_argument("--solo-generos", action="store_true", help="Solo playlists por genero")
    args = parser.parse_args()

    pause = args.pausa if args.pausa is not None else float(os.environ.get("YOUTUBE_PLAYLIST_DELAY", "0.2"))
    max_scan = args.max_scan if args.max_scan is not None else int(os.environ.get("YOUTUBE_PLAYLIST_SCAN", "200"))
    if args.solo_bandas and args.solo_generos:
        print("No se puede usar --solo-bandas y --solo-generos al mismo tiempo.")
        return

    youtube = authenticate()
    playlist_cache = list_playlists(youtube)

    total = 0
    added = 0
    skipped = 0

    for video in iter_public_videos(youtube, limit=args.limite, include_unlisted=args.incluir_no_listado):
        video_id = video.get("id")
        snippet = video.get("snippet", {})
        title = snippet.get("title") or ""
        description = snippet.get("description") or ""

        band = parse_band_from_title(title)
        genres = parse_genres_from_description(description)
        playlist_titles = []
        if band and band.lower() != "unknown" and not args.solo_generos:
            playlist_titles.append(band)
        if genres and not args.solo_bandas:
            playlist_titles.extend(genres)

        if playlist_titles:
            seen = set()
            unique_titles = []
            for item in playlist_titles:
                key = item.lower().strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                unique_titles.append(item)
            playlist_titles = unique_titles

        if not playlist_titles:
            skipped += 1
            print(f"Sin banda/genero: {title} ({video_id})")
            continue

        for playlist_title in playlist_titles:
            try:
                playlist_id = get_or_create_playlist(youtube, playlist_title, playlist_cache)
            except HttpError as exc:
                if is_rate_limit_error(exc):
                    print("Rate limit al crear playlists. Se detiene la ejecucion.")
                    return
                raise
            if not playlist_id:
                continue
            try:
                if not args.sin_check:
                    in_playlist = is_video_in_playlist(
                        youtube,
                        playlist_id,
                        video_id,
                        max_items=max_scan,
                    )
                    if in_playlist:
                        continue
                add_video_to_playlist(youtube, playlist_id, video_id)
                added += 1
                if pause:
                    time.sleep(pause)
            except HttpError as exc:
                if is_rate_limit_error(exc):
                    print("Rate limit en playlists. Se detiene la ejecucion.")
                    return
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

    print(f"Listo. Videos procesados: {total}, agregados: {added}, omitidos: {skipped}.")


if __name__ == "__main__":
    main()
