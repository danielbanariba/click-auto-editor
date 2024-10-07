import os
import pickle
import time
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timezone

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
            credentials = flow.run_local_server(port=0)
        
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)
    
    return build('youtube', 'v3', credentials=credentials)

def main():
    youtube = get_authenticated_service()
    
    try:
        scheduled_videos = 0
        next_page_token = None
        
        while True:
            # Primero, obtenemos los IDs de los videos
            search_request = youtube.search().list(
                part="id",
                type="video",
                forMine=True,
                maxResults=50,
                pageToken=next_page_token
            )
            search_response = search_request.execute()
            
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            
            # Luego, obtenemos los detalles de los videos, incluyendo el estado
            videos_request = youtube.videos().list(
                part="status",
                id=','.join(video_ids)
            )
            videos_response = videos_request.execute()
            
            for item in videos_response['items']:
                status = item['status']
                if status['privacyStatus'] == 'private' and 'publishAt' in status:
                    publish_time = datetime.fromisoformat(status['publishAt'].replace('Z', '+00:00'))
                    if publish_time.year == 2026:
                        scheduled_videos += 1
            
            next_page_token = search_response.get('nextPageToken')
            if not next_page_token:
                break
            
            time.sleep(1)  # Añade un pequeño retraso entre solicitudes
        
        print(f"Total de videos programados para 2026: {scheduled_videos}")
    
    except HttpError as e:
        print(f"Ocurrió un error en la API de YouTube: {e}")

if __name__ == "__main__":
    main()