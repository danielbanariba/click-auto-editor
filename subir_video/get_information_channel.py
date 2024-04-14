from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import json
from subir_video import authenticate

def get_my_videos(youtube):
    request = youtube.channels().list(
        part="contentDetails",
        mine=True
    )
    response = request.execute()

    playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    videos = []
    next_page_token = None

    while True:
        request = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()

        videos += response['items']

        next_page_token = response.get('nextPageToken')

        if next_page_token is None:
            break

    video_titles = [video['snippet']['title'] for video in videos]
    return video_titles

def update_video_list(video_titles):
    with open('lista_de_bandas_ya_subidas_al_canal.txt', 'r') as f:
        existing_titles = f.read().splitlines()

    new_titles = [title for title in video_titles if title not in existing_titles]

    with open('lista_de_bandas_ya_subidas_al_canal.txt', 'a') as f:
        for title in new_titles:
            f.write(title + '\n')

if __name__ == '__main__':
    youtube = authenticate()
    video_titles = get_my_videos(youtube)
    update_video_list(video_titles)