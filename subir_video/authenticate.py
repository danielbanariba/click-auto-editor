from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from pathlib import Path
import os

# Define los scopes de la API de YouTube
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

def authenticate():
    # Autentica usando las credenciales de OAuth 2.0
    secrets_path = os.environ.get("YOUTUBE_CLIENT_SECRETS", "client_secrets.json")
    secrets_file = Path(secrets_path)
    if not secrets_file.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo de credenciales: {secrets_file}. "
            "Configura YOUTUBE_CLIENT_SECRETS en tu .env o coloca client_secrets.json en la raiz."
        )

    token_path = Path(os.environ.get("YOUTUBE_TOKEN_FILE", "token.json"))
    credentials = None
    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(secrets_file), SCOPES)
            credentials = flow.run_local_server(port=0)
        token_path.write_text(credentials.to_json(), encoding="utf-8")

    return build('youtube', 'v3', credentials=credentials)
