from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
import os
import json
from subir_video import authenticate
from subir_video import consulta_disponibilidad_API

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
    youtube = authenticate()
    initialize_upload(youtube, 'prueba.mp4')
    
# consultar si ya puedo hacer petición a la API de youtube
consulta_disponibilidad_API()