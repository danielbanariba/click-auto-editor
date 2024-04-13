
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Define los scopes de la API de YouTube
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

def authenticate():
    # Autentica usando las credenciales de OAuth 2.0
    flow = InstalledAppFlow.from_client_secrets_file('client_secrets.json', SCOPES)
    credentials = flow.run_local_server(port=0)
    return build('youtube', 'v3', credentials=credentials)