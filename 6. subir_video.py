from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
import os
import json

# Configura las credenciales de la API de YouTube
CLIENT_SECRETS_FILE = "client_secrets.json"
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

def get_authenticated_service():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    credentials = flow.run_local_server(port=0)
    return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

def initialize_upload(youtube, file):
    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": "My video title",
                "description": "This is a description of my video",
                "tags": ["my", "video", "tags"],
                "categoryId": "22"
            },
            "status": {
                "privacyStatus": "private"
            }
        },
        media_body=MediaFileUpload(file)
    )
    response = request.execute()
    print(response)

if __name__ == '__main__':
    youtube = get_authenticated_service()
    initialize_upload(youtube, 'prueba.mp4')