import os
import random
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
import time
import sys

# Importar configuración
sys.path.append(str(Path(__file__).parent))
from config import (
    DIR_AUDIO_SCRIPTS,
    INTRO_VIDEO,
    MAX_PARALLEL_RENDERS,
    MAX_FOLDERS_TO_PROCESS,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    FPS,
    INTRO_DURATION,
    USE_GPU,
    VIDEO_PRESET_NVENC,
    VIDEO_CQ,
    VIDEO_PRESET_CPU,
    VIDEO_CRF,
    AUDIO_BITRATE
)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Rutas principales
MAIN_DIR = DIR_AUDIO_SCRIPTS
OUTPUT_DIR = DIR_AUDIO_SCRIPTS


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def get_complementary_color(r, g, b):
    """
    Calcula el color complementario de un RGB dado
    (mismo algoritmo que en auto_effects.py)
    """
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn

    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/diff) + 120) % 360
    else:
        h = (60 * ((r-g)/diff) + 240) % 360

    s = 0 if mx == 0 else (diff/mx)
    v = mx

    # Calcular el color complementario (rotar 180°)
    h = (h + 180) % 360

    # Convertir HSV de vuelta a RGB
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r = int((r + m) * 255)
    g = int((g + m) * 255)
    b = int((b + m) * 255)

    return r, g, b


def extract_average_color(image_path):
    """
    Extrae el color promedio de una imagen
    """
    try:
        img = Image.open(image_path)
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Calcular color promedio
        data = np.array(img)
        avg_color = data.mean(axis=(0, 1))

        return tuple(avg_color.astype(int))
    except Exception as e:
        print(f"Error extrayendo color de {image_path}: {e}")
        return (255, 255, 255)  # Blanco por defecto


def analyze_audio_amplitude(audio_path, fps=30):
    """
    Analiza la amplitud del audio usando FFmpeg (más rápido que librosa)
    Retorna valores normalizados para escala reactiva al audio
    """
    try:
        # Usar FFmpeg para extraer volumen RMS
        # Esto es MÁS RÁPIDO que librosa y ya tienes FFmpeg instalado
        cmd = [
            'ffmpeg',
            '-i', str(audio_path),
            '-af', f'astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-',
            '-f', 'null',
            '-'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.DEVNULL)

        # Parsear salida (simplificado - usaremos valores fijos para velocidad)
        # Para 3 videos diarios, pre-calcular audio es overhead innecesario
        # Usamos expresión dinámica de FFmpeg directamente
        return None

    except Exception as e:
        print(f"Error analizando audio {audio_path}: {e}")
        return None


def generate_wiggle_expression(seed):
    """
    Genera valores aleatorios para simular wiggle(3,3) de After Effects
    Usa un seed para reproducibilidad
    """
    np.random.seed(seed)
    # Generar ruido aleatorio que cambia cada ~0.33 segundos (3 veces por segundo)
    return np.random.uniform(-3, 3)


def render_video(folder_path, folder_name):
    """
    Renderiza un video completo usando FFmpeg
    """
    try:
        print(f"[INICIO] Procesando: {folder_name}")
        start_time = time.time()

        # Buscar archivos necesarios
        audio_file = None
        cover_shadow = None
        cover_hq = None

        for file in os.listdir(folder_path):
            if file.endswith('.mp3'):
                audio_file = folder_path / file
            elif file.endswith('_shadow.png'):
                cover_shadow = folder_path / file
            elif file.endswith('.png') and not file.endswith('_shadow.png'):
                cover_hq = folder_path / file

        if not audio_file or not cover_shadow:
            print(f"[ERROR] Archivos faltantes en {folder_name}")
            return False

        # Usar cover con sombra como principal, HQ como fallback para fondo
        cover_main = cover_shadow
        cover_bg = cover_hq if cover_hq else cover_shadow

        # Extraer color promedio y calcular complementario
        avg_color = extract_average_color(cover_bg)
        comp_r, comp_g, comp_b = get_complementary_color(*avg_color)
        spectrum_color = f"0x{comp_r:02x}{comp_g:02x}{comp_b:02x}"

        # Ruta de salida
        output_file = folder_path / f"{folder_name}.mp4"

        # ====================================================================
        # FILTRO COMPLEJO DE FFMPEG
        # ====================================================================

        # Este es el filtro que replica el look de After Effects:
        # 1. Fondo: Portada difuminada a pantalla completa
        # 2. Espectro: Vertical en el lado izquierdo
        # 3. Portada: Centro-derecha con zoom + wiggle rotation + audio reactive

        # Filtro FFmpeg MEJORADO con efecto VHS
        # INTRO: Video de intro escalado con alta calidad
        # FONDO: Portada difuminada + rotación suave + efecto VHS (ruido, scanlines, color degradado)
        # PORTADA: Cover centrado grande (1100px) con nitidez
        # TRANSICIÓN: Fade de 1 segundo
        # Audio: fusiona intro + música original
        filter_complex = f"""
[0:v]scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:flags=lanczos,fps={FPS}[intro];
[1:v]scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:flags=lanczos:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},gblur=sigma=25,fps={FPS},rotate=a='sin(t*0.5)*0.1':fillcolor=black:ow={VIDEO_WIDTH}:oh={VIDEO_HEIGHT},noise=alls=12:allf=t,eq=saturation=0.7:contrast=1.2:brightness=-0.03,drawbox=y='mod(t*200,4)':color=black@0.25:width={VIDEO_WIDTH}:height=1:t=fill[bg];
[1:v]scale=-1:1100:flags=lanczos,format=yuva420p,fps={FPS},unsharp=5:5:1.0:5:5:0.0[cover];
[bg][cover]overlay=(W-w)/2:(H-h)/2,fps={FPS}[content];
[intro][content]xfade=transition=fade:duration=1:offset={INTRO_DURATION-1}[outv];
[0:a]atrim=0:{INTRO_DURATION}[intro_audio];
[2:a]adelay=delays={int(INTRO_DURATION*1000)}:all=1[music_delayed];
[intro_audio][music_delayed]amix=inputs=2:duration=longest[outa]
""".strip()

        # Comando FFmpeg completo con soporte GPU/CPU
        if USE_GPU:
            # Usar NVENC (GPU Nvidia)
            cmd = [
                'ffmpeg',
                '-y',  # Sobrescribir sin preguntar
                '-hwaccel', 'cuda',  # Aceleración por hardware
                '-i', str(INTRO_VIDEO),  # [0] Intro
                '-loop', '1', '-t', '9999', '-i', str(cover_main),  # [1] Portada
                '-i', str(audio_file),  # [2] Audio
                '-filter_complex', filter_complex,
                '-map', '[outv]',  # Video de salida
                '-map', '[outa]',  # Audio fusionado (intro + música)
                '-c:v', 'h264_nvenc',  # Encoder GPU
                '-preset', VIDEO_PRESET_NVENC,  # p1-p7
                '-cq', str(VIDEO_CQ),  # Constant Quality
                '-rc', 'vbr',  # Variable bitrate
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', AUDIO_BITRATE,
                '-shortest',  # Terminar cuando el audio termine
                str(output_file)
            ]
        else:
            # Usar libx264 (CPU)
            cmd = [
                'ffmpeg',
                '-y',
                '-i', str(INTRO_VIDEO),
                '-loop', '1', '-t', '9999', '-i', str(cover_main),
                '-i', str(audio_file),
                '-filter_complex', filter_complex,
                '-map', '[outv]',
                '-map', '[outa]',
                '-c:v', 'libx264',
                '-preset', VIDEO_PRESET_CPU,
                '-crf', str(VIDEO_CRF),
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', AUDIO_BITRATE,
                '-shortest',
                str(output_file)
            ]

        # Ejecutar FFmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:
            elapsed = time.time() - start_time
            print(f"[ÉXITO] {folder_name} renderizado en {elapsed:.1f}s")
            return True
        else:
            print(f"[ERROR] FFmpeg falló en {folder_name}")
            print(f"STDERR: {result.stderr[-500:]}")  # Últimas 500 chars del error
            return False

    except Exception as e:
        print(f"[EXCEPCIÓN] Error procesando {folder_name}: {e}")
        return False


def process_folders_parallel():
    """
    Procesa carpetas en paralelo (hasta MAX_PARALLEL_RENDERS simultáneos)
    """
    # Recoger todas las carpetas
    folders = [
        (MAIN_DIR / folder_name, folder_name)
        for folder_name in os.listdir(MAIN_DIR)
        if (MAIN_DIR / folder_name).is_dir()
    ]

    # Mezclar aleatoriamente
    random.shuffle(folders)

    # Limitar cantidad
    folders = folders[:MAX_FOLDERS_TO_PROCESS]

    print(f"\n{'='*60}")
    print(f"INICIANDO RENDERIZADO PARALELO")
    print(f"Carpetas a procesar: {len(folders)}")
    print(f"Renders paralelos: {MAX_PARALLEL_RENDERS}")
    print(f"{'='*60}\n")

    # Procesar en paralelo
    successful = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_RENDERS) as executor:
        # Enviar todos los trabajos
        future_to_folder = {
            executor.submit(render_video, folder_path, folder_name): folder_name
            for folder_path, folder_name in folders
        }

        # Procesar conforme terminan
        for future in as_completed(future_to_folder):
            folder_name = future_to_folder[future]
            try:
                success = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"[EXCEPCIÓN] Error en {folder_name}: {e}")
                failed += 1

    print(f"\n{'='*60}")
    print(f"RENDERIZADO COMPLETADO")
    print(f"Exitosos: {successful}")
    print(f"Fallidos: {failed}")
    print(f"{'='*60}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Verificar que el intro existe
    if not INTRO_VIDEO.exists():
        print(f"ERROR: No se encuentra el video de intro en {INTRO_VIDEO}")
        exit(1)

    # Verificar que el directorio principal existe
    if not MAIN_DIR.exists():
        print(f"ERROR: No se encuentra el directorio {MAIN_DIR}")
        exit(1)

    # Procesar
    process_folders_parallel()
