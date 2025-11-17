import os
import random
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
import time
import sys
import re
import select

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


def get_audio_duration(audio_path):
    """
    Obtiene la duración del audio en segundos usando ffprobe
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception:
        return 0


def format_time(seconds):
    """
    Formatea segundos en formato HH:MM:SS
    """
    if seconds < 0:
        return "00:00:00"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_progress_bar(current_time, total_duration, elapsed_time, folder_name, bar_width=40):
    """
    Imprime una barra de progreso con porcentaje y tiempo estimado
    """
    if total_duration <= 0:
        return

    percentage = min(100, (current_time / total_duration) * 100)
    filled = int(bar_width * percentage / 100)
    bar = '█' * filled + '░' * (bar_width - filled)

    # Calcular tiempo restante estimado
    if current_time > 0 and elapsed_time > 0:
        speed = current_time / elapsed_time  # segundos de video por segundo real
        remaining_video_time = total_duration - current_time
        if speed > 0:
            eta_seconds = remaining_video_time / speed
        else:
            eta_seconds = 0
    else:
        eta_seconds = 0

    eta_str = format_time(eta_seconds)
    elapsed_str = format_time(elapsed_time)

    # Limpiar línea y escribir progreso
    sys.stdout.write(f'\r[{folder_name[:30]:30s}] |{bar}| {percentage:5.1f}% | Tiempo: {elapsed_str} | ETA: {eta_str}')
    sys.stdout.flush()


def render_video(folder_path, folder_name, show_progress=False):
    """
    Renderiza un video completo usando FFmpeg
    show_progress: Si True, muestra barra de progreso en tiempo real (solo para modo single)
    """
    try:
        if not show_progress:
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

        # Obtener duración total del audio (para la barra de progreso)
        audio_duration = get_audio_duration(audio_file)
        total_duration = INTRO_DURATION + audio_duration  # Duración total del video

        if show_progress:
            print(f"\n[RENDERIZANDO] {folder_name}")
            print(f"Duración estimada: {format_time(total_duration)}")
            print("")

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
        if show_progress:
            # Modo con barra de progreso (capturar stderr en tiempo real)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Leer stderr en tiempo real para obtener progreso
            stderr_output = ""
            time_pattern = re.compile(r'time=(\d+):(\d+):(\d+\.?\d*)')

            while True:
                # Leer una línea del stderr
                line = process.stderr.readline()
                if not line and process.poll() is not None:
                    break

                stderr_output += line

                # Buscar el tiempo actual en la salida
                match = time_pattern.search(line)
                if match:
                    hours = int(match.group(1))
                    minutes = int(match.group(2))
                    seconds = float(match.group(3))
                    current_time = hours * 3600 + minutes * 60 + seconds

                    elapsed = time.time() - start_time
                    print_progress_bar(current_time, total_duration, elapsed, folder_name)

            process.wait()
            print()  # Nueva línea después de la barra de progreso

            if process.returncode == 0:
                elapsed = time.time() - start_time
                print(f"\n[ÉXITO] {folder_name} renderizado en {format_time(elapsed)}")
                return True
            else:
                print(f"\n[ERROR] FFmpeg falló en {folder_name}")
                print(f"STDERR: {stderr_output[-500:]}")
                return False
        else:
            # Modo sin progreso (para renderizado paralelo)
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
                print(f"STDERR: {result.stderr[-500:]}")
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


def process_folders_sequential():
    """
    Procesa carpetas secuencialmente (1 a la vez) con barra de progreso
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
    print(f"INICIANDO RENDERIZADO SECUENCIAL")
    print(f"Carpetas a procesar: {len(folders)}")
    print(f"{'='*60}")

    # Procesar secuencialmente
    successful = 0
    failed = 0
    total_start_time = time.time()

    for i, (folder_path, folder_name) in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] ", end="")
        success = render_video(folder_path, folder_name, show_progress=True)

        if success:
            successful += 1
        else:
            failed += 1

        # Mostrar estadísticas parciales
        total_elapsed = time.time() - total_start_time
        avg_time = total_elapsed / i
        remaining = len(folders) - i
        eta_total = remaining * avg_time

        print(f"\nProgreso total: {i}/{len(folders)} videos")
        print(f"Exitosos: {successful} | Fallidos: {failed}")
        print(f"Tiempo promedio por video: {format_time(avg_time)}")
        print(f"ETA para completar todos: {format_time(eta_total)}")
        print(f"{'='*60}")

    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"RENDERIZADO COMPLETADO")
    print(f"Tiempo total: {format_time(total_elapsed)}")
    print(f"Exitosos: {successful}")
    print(f"Fallidos: {failed}")
    print(f"{'='*60}\n")


def render_single_video():
    """
    Renderiza un solo video aleatorio con barra de progreso (modo prueba)
    """
    # Recoger todas las carpetas
    folders = [
        (MAIN_DIR / folder_name, folder_name)
        for folder_name in os.listdir(MAIN_DIR)
        if (MAIN_DIR / folder_name).is_dir()
    ]

    if not folders:
        print("ERROR: No hay carpetas para procesar")
        return

    # Seleccionar una carpeta aleatoria
    folder_path, folder_name = random.choice(folders)

    print(f"\n{'='*60}")
    print(f"MODO PRUEBA - RENDERIZADO ÚNICO")
    print(f"{'='*60}")

    # Renderizar con barra de progreso
    success = render_video(folder_path, folder_name, show_progress=True)

    print(f"\n{'='*60}")
    if success:
        print(f"PRUEBA COMPLETADA EXITOSAMENTE")
        print(f"Video: {folder_path / f'{folder_name}.mp4'}")
    else:
        print(f"PRUEBA FALLIDA")
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

    # Verificar argumentos
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test" or sys.argv[1] == "-t":
            # Modo prueba: renderizar un solo video con barra de progreso
            render_single_video()
        elif sys.argv[1] == "--parallel" or sys.argv[1] == "-p":
            # Modo paralelo: renderizar múltiples videos simultáneamente (sin barra de progreso)
            process_folders_parallel()
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("\nUso:")
            print(f"  python '{Path(__file__).name}'              # Renderizado secuencial (1 video a la vez con barra de progreso)")
            print(f"  python '{Path(__file__).name}' --test       # Prueba con 1 video + barra de progreso")
            print(f"  python '{Path(__file__).name}' -t           # Igual que --test")
            print(f"  python '{Path(__file__).name}' --parallel   # Renderizado paralelo (3 videos simultáneos, sin barra)")
            print(f"  python '{Path(__file__).name}' -p           # Igual que --parallel")
            print("")
        else:
            print(f"Argumento desconocido: {sys.argv[1]}")
            print("Usa --help para ver opciones")
    else:
        # Modo normal: renderizado secuencial con barra de progreso
        process_folders_sequential()
