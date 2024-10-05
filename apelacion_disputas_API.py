import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime

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
    
    with open('youtube_copyright_check_2026.txt', 'w', encoding='utf-8') as log_file:
        try:
            # Obtener videos
            videos = []
            next_page_token = None
            while True:
                request = youtube.search().list(
                    part="id,snippet",
                    type="video",
                    forMine=True,
                    maxResults=50,
                    pageToken=next_page_token
                )
                response = request.execute()
                
                for item in response['items']:
                    video_id = item['id']['videoId']
                    video_title = item['snippet']['title']
                    scheduled_time = item['snippet'].get('publishTime')
                    
                    # Verificar si está programado para 2026
                    if scheduled_time:
                        scheduled_year = datetime.strptime(scheduled_time, "%Y-%m-%dT%H:%M:%SZ").year
                        if scheduled_year == 2026:
                            videos.append((video_id, video_title))
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
            
            log_file.write(f"Total de videos potencialmente programados para 2026: {len(videos)}\n\n")

            # Revisar cada video para estado y reclamaciones de derechos de autor
            videos_with_claims = []
            for video_id, video_title in videos:
                request = youtube.videos().list(
                    part="contentDetails,status",
                    id=video_id
                )
                response = request.execute()
                
                # Registrar la respuesta completa en el archivo de texto
                log_file.write(f"Respuesta para el video '{video_title}' (ID: {video_id}):\n{response}\n\n")
                
                if 'items' in response and len(response['items']) > 0:
                    status = response['items'][0].get('status', {})
                    content_details = response['items'][0].get('contentDetails', {})
                    
                    # Verificar si el video no es público y está programado para 2026
                    privacy_status = status.get('privacyStatus')
                    publish_at = status.get('publishAt')
                    
                    if privacy_status != 'public' and publish_at:
                        publish_year = datetime.strptime(publish_at, "%Y-%m-%dT%H:%M:%SZ").year
                        if publish_year == 2026:
                            claims = content_details.get('contentClaims', [])
                            if claims:
                                claim_info = {
                                    'title': video_title,
                                    'id': video_id,
                                    'claims': [claim['origin'] for claim in claims]
                                }
                                videos_with_claims.append(claim_info)
                            elif 'rejectionReason' in status:
                                claim_info = {
                                    'title': video_title,
                                    'id': video_id,
                                    'claims': [status['rejectionReason']]
                                }
                                videos_with_claims.append(claim_info)
            
            # Registrar los videos con reclamaciones de derechos de autor
            log_file.write("\nVideos programados para 2026 con reclamaciones de derechos de autor:\n")
            if videos_with_claims:
                for video in videos_with_claims:
                    log_file.write(f"El video '{video['title']}' (ID: {video['id']}) tiene reclamaciones de derechos de autor:\n")
                    for claim in video['claims']:
                        log_file.write(f"  - Tipo de reclamación: {claim}\n")
                    log_file.write("--------------------\n")
            else:
                log_file.write("No se encontraron videos programados para 2026 con reclamaciones de derechos de autor.\n")
            
            log_file.write(f"\nTotal de videos programados para 2026 con reclamaciones: {len(videos_with_claims)}\n")
        
        except HttpError as e:
            log_file.write(f"Ocurrió un error en la API de YouTube: {e}\n")

if __name__ == "__main__":
    main()