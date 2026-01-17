"""
Configuración centralizada para el proyecto de edición automatizada
"""

import os
from pathlib import Path

# ============================================================================
# RUTAS PRINCIPALES DEL PROYECTO
# ============================================================================

# Directorio raíz donde está montado el disco externo
DEFAULT_BASE_DIR = Path("/run/media/banar/Entretenimiento/01_edicion_automatizada")
# Opcional: usar un base_dir rápido (NVMe) con variables de entorno
FAST_BASE_DIR = os.environ.get("FAST_BASE_DIR")
USE_FAST_BASE = os.environ.get("USE_FAST_BASE", "0") == "1"

BASE_DIR = Path(FAST_BASE_DIR) if USE_FAST_BASE and FAST_BASE_DIR else DEFAULT_BASE_DIR

# Pipeline de procesamiento
DIR_LIMPIEZA = BASE_DIR / "01_limpieza_de_impurezas"
DIR_JUNTAR_AUDIOS = BASE_DIR / "02_juntar_audios"
DIR_AUDIO_SCRIPTS = BASE_DIR / "01_limpieza_de_impurezas"  # Carpeta fuente para renderizado
DIR_UPLOAD = BASE_DIR / "upload_video"
DIR_VERIFICACION = BASE_DIR / "verificacion"

# Directorio temporal en SSD para renderizado rápido
# Los archivos se copian aquí antes de renderizar para evitar I/O bottleneck
SSD_TEMP_DIR = Path("/home/banar/temp_render")
USE_SSD_TEMP = True  # Copiar a SSD antes de renderizar

# Directorios auxiliares
DIR_VOLVER_A_BUSCAR = BASE_DIR / "03_volver_a_buscar"
DIR_REGRESAR_AUDIO = BASE_DIR / "04_regresar_audio_scripts"
DIR_NO_TIENEN_CARPETAS = BASE_DIR / "no_tienen_carpetas"
DIR_NO_TIENEN_PORTADA = BASE_DIR / "no_tienen_portada"
DIR_NO_TIENEN_DESCRIPCION = BASE_DIR / "no_tienen_descripcion"
DIR_YA = BASE_DIR / "Ya"
DIR_YA_SUBIDOS = BASE_DIR / "Ya subidos"
DIR_RARO = BASE_DIR / "raro"
DIR_NEED_CENSURED = BASE_DIR / "need_censured"
DIR_CORREGIR_EDICION = BASE_DIR / "corregir_edicion"
DIR_BANDAS_SUBIDAS = BASE_DIR / "bandas_que_supuestamente_ya_se_subieron"

# ============================================================================
# ARCHIVOS DEL PROYECTO
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
INTRO_VIDEO = PROJECT_ROOT / "content" / "0000000000000000.mp4"
BANDAS_SUBIDAS_TXT = PROJECT_ROOT / "bandas-subidas-al-canal.txt"
PLAYWRIGHT_PROFILE_DIR = PROJECT_ROOT / ".playwright_profile"
PLAYWRIGHT_SELECTORS_DIR = PROJECT_ROOT / "selectores"

# ============================================================================
# CONFIGURACIÓN DE RENDERIZADO
# ============================================================================

# FFmpeg render settings
VIDEO_WIDTH = 3840
VIDEO_HEIGHT = 2160
FPS = 24
INTRO_DURATION = 7.0  # Segundos del video de intro

# Multiprocessing
# i9-9900K tiene 16 threads - los filtros VHS son CPU-bound
# RTX 3090 Ti puede manejar 3-5 streams NVENC simultáneos
# Configuración óptima: 4 renders paralelos (4 threads por render ≈ 16 total)
MAX_PARALLEL_RENDERS = 4  # 4K es pesado; ajusta si tu VRAM lo permite
MAX_FOLDERS_TO_PROCESS = 150  # Límite de carpetas por ejecución

# Video quality settings
# GPU (NVENC) settings - Requiere GPU Nvidia con soporte NVENC
USE_GPU = True  # RTX 3090 Ti detectada - NVENC activado
VIDEO_PRESET_NVENC = "p1"  # p1 = máxima velocidad (ideal para RTX 3090 Ti)
VIDEO_CQ = 20  # Constant Quality (20 es excelente para YouTube, más rápido que 18)
VIDEO_BITRATE = "45M"
VIDEO_MAXRATE = "45M"
VIDEO_BUFSIZE = "90M"

# Opciones NVENC recomendadas para YouTube 4K SDR
NVENC_EXTRA_OPTS = [
    "-rc", "vbr",              # Variable bitrate
    "-b:v", VIDEO_BITRATE,
    "-maxrate", VIDEO_MAXRATE,
    "-bufsize", VIDEO_BUFSIZE,
    "-bf", "2",
    "-g", str(FPS // 2),
    "-profile:v", "high",
    "-movflags", "+faststart",
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
    "-colorspace", "bt709",
    "-spatial_aq", "1",        # Adaptive quantization espacial
    "-temporal_aq", "1",       # Adaptive quantization temporal
]

# CPU (libx264) settings - Fallback si no hay GPU
VIDEO_PRESET_CPU = "fast"  # Opciones: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
VIDEO_CRF = 20  # Calidad para libx264 (0-51, menor = mejor calidad, 18-23 es visualmente sin pérdidas)

# Audio
AUDIO_BITRATE = "384k"
AUDIO_SAMPLE_RATE = 48000

# ============================================================================
# CONFIGURACIÓN VHS GPU (C++/CUDA)
# ============================================================================

# Usar el pipeline GPU en C++ para los efectos VHS (recomendado con 3090)
USE_CPP_VHS = True
VHS_CPP_BIN = PROJECT_ROOT / "cpp" / "build" / "vhs_render"
VHS_CPP_INTENSITY = 1.0
VHS_CPP_OVERLAY = PROJECT_ROOT / "content" / "vhs_noise.mp4"
# Si el render CUDA falla, usar FFmpeg como fallback para no perder el video
ALLOW_FFMPEG_FALLBACK = True
# Si ocurre un error CUDA, desactivar C++ para el resto del run
DISABLE_CPP_ON_CUDA = False
# Atajo para evitar múltiples reintentos CUDA (más rápido, menos ruido)
CUDA_FAIL_FAST = False

# LEGACY: Mantener compatibilidad con scripts antiguos
VIDEO_PRESET = VIDEO_PRESET_CPU

# ============================================================================
# CONFIGURACIÓN DE AUDIO
# ============================================================================

# Formatos de audio soportados
AUDIO_FORMATS = (".mp3", ".flac", ".wav", ".wma", ".m4a", ".MP3", ".Mp3")

# Formatos de imagen soportados
IMAGE_FORMATS = (".png", ".jpg", ".jpeg", ".jfif", ".gif", ".tiff", ".raw")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def create_directories():
    """
    Crea todos los directorios necesarios si no existen
    """
    dirs = [
        DIR_LIMPIEZA,
        DIR_JUNTAR_AUDIOS,
        DIR_AUDIO_SCRIPTS,
        DIR_UPLOAD,
        DIR_VERIFICACION,
        DIR_VOLVER_A_BUSCAR,
        DIR_REGRESAR_AUDIO,
        DIR_NO_TIENEN_CARPETAS,
        DIR_NO_TIENEN_PORTADA,
        DIR_NO_TIENEN_DESCRIPCION,
        DIR_YA,
        DIR_RARO,
        DIR_NEED_CENSURED,
        DIR_CORREGIR_EDICION,
        DIR_BANDAS_SUBIDAS,
    ]

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ {directory}")

def check_environment():
    """
    Verifica que el entorno esté correctamente configurado
    """
    errors = []

    # Verificar que el disco externo esté montado
    if not BASE_DIR.exists():
        errors.append(f"El disco externo no está montado en {BASE_DIR}")

    # Verificar que el video de intro exista
    if not INTRO_VIDEO.exists():
        errors.append(f"No se encuentra el video de intro en {INTRO_VIDEO}")

    return errors

if __name__ == "__main__":
    print("\n" + "="*60)
    print("VERIFICACIÓN DE ENTORNO")
    print("="*60 + "\n")

    # Verificar entorno
    errors = check_environment()

    if errors:
        print("❌ ERRORES ENCONTRADOS:\n")
        for error in errors:
            print(f"  • {error}")
        print("\nPor favor, corrige estos errores antes de continuar.")
    else:
        print("✓ Entorno verificado correctamente\n")

        # Crear directorios
        print("Creando directorios necesarios...\n")
        create_directories()

        print("\n" + "="*60)
        print("CONFIGURACIÓN COMPLETA")
        print("="*60 + "\n")
