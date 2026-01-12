import os
import pickle
import time
import re
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

def extract_copyright_info(snippet, content_details):
    copyright_info = {}
    copyright_info['title'] = snippet['title']
    copyright_info['description'] = snippet['description']
    copyright_info['licensedContent'] = content_details.get('licensedContent', False)
    
    # Intentar extraer información del reclamante de la descripción
    claimant_patterns = [
        r"provided to YouTube by\s*(.+)",
        r"©\s*(.+)",
        r"copyright\s*(.+)",
        r"licensed to\s*(.+)",
        r"courtesy of\s*(.+)",
    ]
    
    for pattern in claimant_patterns:
        match = re.search(pattern, copyright_info['description'], re.IGNORECASE)
        if match:
            copyright_info['possible_claimant'] = match.group(1).strip()
            break
    
    return copyright_info

def main():
    youtube = get_authenticated_service()
    
    try:
        scheduled_videos = 0
        copyright_videos = 0
        videos_with_claimants = 0
        next_page_token = None
        
        with open('copyright_info.txt', 'w', encoding='utf-8') as f:
            f.write("Videos programados para 2026 con información de derechos de autor:\n\n")
        
        while True:
            search_request = youtube.search().list(
                part="id,snippet",
                type="video",
                forMine=True,
                maxResults=50,
                pageToken=next_page_token
            )
            search_response = search_request.execute()
            
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            
            videos_request = youtube.videos().list(
                part="status,snippet,contentDetails",
                id=','.join(video_ids)
            )
            videos_response = videos_request.execute()
            
            for item in videos_response['items']:
                status = item['status']
                snippet = item['snippet']
                content_details = item['contentDetails']
                
                if status['privacyStatus'] == 'private' and 'publishAt' in status:
                    publish_time = datetime.fromisoformat(status['publishAt'].replace('Z', '+00:00'))
                    if publish_time.year == 2026:
                        scheduled_videos += 1
                        
                        copyright_info = extract_copyright_info(snippet, content_details)
                        
                        if copyright_info['licensedContent']:
                            copyright_videos += 1
                            
                            with open('copyright_info.txt', 'a', encoding='utf-8') as f:
                                f.write(f"Título: {copyright_info['title']}\n")
                                f.write(f"ID del video: {item['id']}\n")
                                f.write(f"Fecha de publicación programada: {status['publishAt']}\n")
                                f.write(f"Contenido con licencia: {'Sí' if copyright_info['licensedContent'] else 'No'}\n")
                                if 'possible_claimant' in copyright_info:
                                    f.write(f"Posible reclamante: {copyright_info['possible_claimant']}\n")
                                    videos_with_claimants += 1
                                f.write("Descripción:\n")
                                f.write(f"{copyright_info['description']}\n")
                                f.write("\n" + "="*50 + "\n\n")
            
            next_page_token = search_response.get('nextPageToken')
            if not next_page_token:
                break
            
            time.sleep(1)  # Pequeño retraso entre solicitudes
        
        print(f"Total de videos programados para 2026: {scheduled_videos}")
        print(f"Videos con información de derechos de autor: {copyright_videos}")
        print(f"Videos con posibles reclamantes identificados: {videos_with_claimants}")
        print("Se ha guardado la información detallada en 'copyright_info.txt'")
    
    except HttpError as e:
        print(f"Ocurrió un error en la API de YouTube: {e}")

if __name__ == "__main__":
    main()