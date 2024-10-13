from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
import os
import json
from subir_video import authenticate
from subir_video import consulta_disponibilidad_API

# TODO se me ocurre mejor hacer que el programa consiga el titulo de todos los videos que tengo subidos, y se cree una lista de reproducción con los titulos de las bandas,
# Osea solo los que terminen el nombre con - Banda, para poder indentificar mas facilmente el nombre y si existen dos bandas con el mismo nombre, por ejemplo Totenmond, que hay dos videos diferentes, pero
# Son de la misma banda, entonces solo se le agregue a la lista de reproducción una vez, y se le agregue el video a esa lista de reproducción, y si no existe la lista de reproducción, se cree una nueva



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