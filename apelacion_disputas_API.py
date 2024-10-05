import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']

def get_authenticated_service():
    credentials = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            credentials = pickle.load(token)
    
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secrets.json', SCOPES)
            credentials = flow.run_local_server(port=8080, host='127.0.0.1')
        
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)

    return build('youtube', 'v3', credentials=credentials)

def main():
    youtube = get_authenticated_service()

    try:
        # Obtener todos los videos del canal
        videos = []
        next_page_token = None
        while True:
            request = youtube.search().list(
                part="id",
                type="video",
                forMine=True,
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response['items']:
                videos.append(item['id']['videoId'])
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

        # Revisar cada video para reclamaciones de derechos de autor
        for video_id in videos:
            request = youtube.videos().list(
                part="contentDetails",
                id=video_id
            )
            response = request.execute()

            if 'contentDetails' in response['items'][0] and 'contentClaims' in response['items'][0]['contentDetails']:
                print(f"El video {video_id} tiene reclamaciones de derechos de autor:")
                claims = response['items'][0]['contentDetails']['contentClaims']
                for claim in claims:
                    print(f"  - Tipo de reclamación: {claim['origin']}")
                    print(f"    Contenido reclamado: {claim.get('content', 'No especificado')}")
                print("--------------------")

    except HttpError as e:
        print(f"Ocurrió un error en la API de YouTube: {e}")

if __name__ == "__main__":
    main()