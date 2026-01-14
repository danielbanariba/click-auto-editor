from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

CLIENT_SECRETS_FILE = PROJECT_ROOT.parent / "client_secrets.json"
TOKEN_FILE = PROJECT_ROOT / "token.json"

SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
]

# Si esta vacio, se usa mine=True y se consulta el canal autenticado.
CHANNEL_ID = ""

# Limites
MAX_VIDEOS = 200
TOP_N = 10
MIN_DAYS_FOR_RATE = 3

STOPWORDS = {
    "a", "al", "algo", "alguna", "algunas", "alguno", "algunos",
    "ante", "antes", "asi", "aun", "aunque", "bajo", "bien", "cada",
    "casi", "como", "con", "contra", "cual", "cuando", "de", "del",
    "desde", "donde", "dos", "el", "ella", "ellas", "ellos", "en",
    "entre", "era", "erais", "eran", "eras", "eres", "es", "esa",
    "esas", "ese", "eso", "esos", "esta", "estaba", "estaban", "estado",
    "estais", "estamos", "estan", "estar", "estas", "este", "esto",
    "estos", "estoy", "fue", "fueron", "ha", "hace", "hacia", "han",
    "hasta", "hay", "la", "las", "le", "les", "lo", "los", "mas",
    "me", "mi", "mis", "mucho", "muy", "ni", "no", "nos", "nuestra",
    "nuestro", "o", "os", "otra", "otro", "para", "pero", "poco",
    "por", "porque", "que", "quien", "se", "sea", "ser", "si", "sin",
    "sobre", "son", "su", "sus", "tambien", "te", "ti", "tiene",
    "toda", "todo", "tu", "tus", "un", "una", "uno", "unos", "y",
    "ya", "yo",
}


def ensure_directories():
    for directory in (DATA_DIR, RAW_DIR, PROCESSED_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
