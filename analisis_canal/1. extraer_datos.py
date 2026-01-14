import json
from typing import Iterable, List

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from config import (
    CHANNEL_ID,
    CLIENT_SECRETS_FILE,
    MAX_VIDEOS,
    RAW_DIR,
    SCOPES,
    TOKEN_FILE,
    ensure_directories,
)


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def get_credentials():
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CLIENT_SECRETS_FILE.exists():
                raise FileNotFoundError(
                    f"No se encontro client_secrets.json en {CLIENT_SECRETS_FILE}"
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CLIENT_SECRETS_FILE), SCOPES
            )
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return creds


def get_youtube_client():
    creds = get_credentials()
    return build("youtube", "v3", credentials=creds)


def fetch_channel(youtube):
    if CHANNEL_ID:
        request = youtube.channels().list(
            part="snippet,contentDetails,statistics", id=CHANNEL_ID
        )
    else:
        request = youtube.channels().list(
            part="snippet,contentDetails,statistics", mine=True
        )

    response = request.execute()
    items = response.get("items", [])
    if not items:
        raise RuntimeError("No se encontro el canal. Revisa CHANNEL_ID o el OAuth.")

    return items[0]


def fetch_video_ids(youtube, uploads_playlist_id: str, max_videos: int) -> List[str]:
    video_ids: List[str] = []
    page_token = None

    while len(video_ids) < max_videos:
        response = (
            youtube.playlistItems()
            .list(
                part="contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=50,
                pageToken=page_token,
            )
            .execute()
        )

        for item in response.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])
            if len(video_ids) >= max_videos:
                break

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return video_ids


def fetch_videos(youtube, video_ids: List[str]):
    videos = []
    for chunk in chunked(video_ids, 50):
        response = (
            youtube.videos()
            .list(
                part="snippet,contentDetails,statistics,status",
                id=",".join(chunk),
            )
            .execute()
        )
        videos.extend(response.get("items", []))

    return videos


def main():
    print("=== Extraer datos de YouTube ===")
    ensure_directories()

    youtube = get_youtube_client()
    channel = fetch_channel(youtube)

    save_json(RAW_DIR / "canal.json", channel)
    print("[OK] Canal guardado")

    uploads_id = channel["contentDetails"]["relatedPlaylists"]["uploads"]
    video_ids = fetch_video_ids(youtube, uploads_id, MAX_VIDEOS)
    save_json(RAW_DIR / "video_ids.json", video_ids)
    print(f"[OK] Video IDs guardados: {len(video_ids)}")

    videos = fetch_videos(youtube, video_ids)
    save_json(RAW_DIR / "videos.json", videos)
    print(f"[OK] Videos guardados: {len(videos)}")


if __name__ == "__main__":
    main()
