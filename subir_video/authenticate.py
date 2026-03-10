from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from pathlib import Path
import json
import os

# Define los scopes de la API de YouTube
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube",
]

# Carpeta de credenciales
CREDENTIALS_DIR = Path(__file__).resolve().parent.parent / "credentials"

# Estado de credenciales agotadas (por sesion)
_exhausted_credentials = set()


def get_credential_sets(prefix=None):
    """
    Obtiene pares de (client_secrets, token) disponibles.
    Si prefix es None, usa YOUTUBE_CREDENTIAL_PREFIX del entorno o busca todos.

    Retorna lista de tuplas: [(secrets_path, token_path), ...]
    """
    prefix = prefix or os.environ.get("YOUTUBE_CREDENTIAL_PREFIX")

    # Si no hay carpeta credentials, usar el modo legacy
    if not CREDENTIALS_DIR.exists():
        secrets = Path(os.environ.get("YOUTUBE_CLIENT_SECRETS", "client_secrets.json"))
        token = Path(os.environ.get("YOUTUBE_TOKEN_FILE", "token.json"))
        return [(secrets, token)]

    pairs = []

    if prefix:
        # Buscar credenciales con el prefijo especifico (ej: "playlists", "upload")
        for secrets_file in sorted(
            CREDENTIALS_DIR.glob(f"client_secrets_{prefix}_*.json")
        ):
            # client_secrets_playlists_1.json -> token_playlists_1.json
            name = secrets_file.stem.replace("client_secrets_", "token_")
            token_file = CREDENTIALS_DIR / f"{name}.json"
            pairs.append((secrets_file, token_file))

    if not pairs:
        # Buscar todas las credenciales disponibles
        for secrets_file in sorted(CREDENTIALS_DIR.glob("client_secrets_*.json")):
            name = secrets_file.stem.replace("client_secrets_", "token_")
            token_file = CREDENTIALS_DIR / f"{name}.json"
            pairs.append((secrets_file, token_file))

    # Fallback al modo legacy si no hay credenciales en la carpeta
    if not pairs:
        secrets = Path(os.environ.get("YOUTUBE_CLIENT_SECRETS", "client_secrets.json"))
        token = Path(os.environ.get("YOUTUBE_TOKEN_FILE", "token.json"))
        pairs = [(secrets, token)]

    return pairs


def authenticate_with_credentials(secrets_path, token_path):
    """
    Autentica usando un par especifico de credenciales.
    """
    secrets_file = Path(secrets_path)
    token_file = Path(token_path)

    if not secrets_file.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo de credenciales: {secrets_file}. "
        )

    credentials = None
    if token_file.exists():
        credentials = Credentials.from_authorized_user_file(str(token_file), SCOPES)
        if not credentials.has_scopes(SCOPES):
            credentials = None

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except RefreshError:
                credentials = None
        if not credentials or not credentials.valid:
            flow = InstalledAppFlow.from_client_secrets_file(str(secrets_file), SCOPES)
            credentials = flow.run_local_server(port=0)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(credentials.to_json(), encoding="utf-8")

    return build("youtube", "v3", credentials=credentials)


def authenticate(prefix=None):
    """
    Autentica usando las credenciales disponibles.
    Si hay multiples credenciales con el prefijo, usa la primera disponible
    que no este agotada.

    Args:
        prefix: Prefijo de credenciales (ej: "playlists", "upload")
                Si es None, usa YOUTUBE_CREDENTIAL_PREFIX del entorno.
    """
    pairs = get_credential_sets(prefix)

    # Filtrar credenciales agotadas
    available = [(s, t) for s, t in pairs if str(s) not in _exhausted_credentials]

    if not available:
        # Si todas estan agotadas, reiniciar y usar la primera
        _exhausted_credentials.clear()
        available = pairs

    if not available:
        raise FileNotFoundError(
            "No se encontraron credenciales. "
            "Coloca archivos client_secrets_*.json en la carpeta credentials/"
        )

    secrets_path, token_path = available[0]
    print(f"Usando credenciales: {secrets_path.name}")
    return authenticate_with_credentials(secrets_path, token_path)


def mark_credential_exhausted(secrets_path):
    """
    Marca una credencial como agotada (cuota excedida).
    """
    _exhausted_credentials.add(str(secrets_path))
    print(f"Credencial agotada: {Path(secrets_path).name}")


def get_current_credential_path():
    """
    Retorna la ruta del client_secrets actualmente en uso.
    """
    pairs = get_credential_sets()
    available = [(s, t) for s, t in pairs if str(s) not in _exhausted_credentials]
    if available:
        return available[0][0]
    return pairs[0][0] if pairs else None


def reset_exhausted_credentials():
    """
    Reinicia el estado de credenciales agotadas.
    """
    _exhausted_credentials.clear()


def authenticate_next(prefix=None):
    """
    Marca la credencial actual como agotada y autentica con la siguiente disponible.
    Re-escanea la carpeta para detectar nuevas credenciales agregadas.
    Si no hay mas con el prefijo, prueba con TODAS las credenciales disponibles.
    Retorna None si no hay mas credenciales disponibles.
    """
    current = get_current_credential_path()
    if current:
        mark_credential_exhausted(current)

    # Re-escanear carpeta para detectar nuevas credenciales con el prefijo
    pairs = get_credential_sets(prefix)
    available = [(s, t) for s, t in pairs if str(s) not in _exhausted_credentials]

    # Si no hay con el prefijo, probar con TODAS las credenciales
    if not available and prefix and CREDENTIALS_DIR.exists():
        all_pairs = []
        for secrets_file in sorted(CREDENTIALS_DIR.glob("client_secrets_*.json")):
            name = secrets_file.stem.replace("client_secrets_", "token_")
            token_file = CREDENTIALS_DIR / f"{name}.json"
            all_pairs.append((secrets_file, token_file))
        available = [(s, t) for s, t in all_pairs if str(s) not in _exhausted_credentials]
        if available:
            print(f"Credenciales '{prefix}' agotadas. Probando credenciales alternativas.")

    if not available:
        print("Todas las credenciales agotadas. Agrega mas en credentials/")
        return None

    secrets_path, token_path = available[0]
    print(f"Cambiando a credenciales: {secrets_path.name}")
    return authenticate_with_credentials(secrets_path, token_path)
