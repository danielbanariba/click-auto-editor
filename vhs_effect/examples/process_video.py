#!/usr/bin/env python3
"""
Ejemplo: Procesar un video con efecto VHS

Este script demuestra cómo aplicar el efecto VHS a un video completo,
incluyendo overlays opcionales de VCR.
"""

import sys
import os
import time

# Añadir directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
from vhs_effect import VHSEffect, VHSTextOverlay

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

INPUT_VIDEO = "input.mp4"  # Cambiar por tu video
OUTPUT_VIDEO = "output_vhs.mp4"
INTENSITY = 0.6  # 0.0-1.0

# Opciones de overlay
ADD_TIMECODE = True
ADD_REC_INDICATOR = True
ADD_DATE = True

# =============================================================================
# PROCESAMIENTO
# =============================================================================

def process_video(input_path: str, output_path: str,
                  intensity: float = 0.5,
                  add_timecode: bool = True,
                  add_rec: bool = True,
                  add_date: bool = True) -> None:
    """
    Procesa un video con efecto VHS completo.

    Args:
        input_path: Ruta al video de entrada
        output_path: Ruta al video de salida
        intensity: Intensidad del efecto (0.0-1.0)
        add_timecode: Añadir timecode VCR
        add_rec: Añadir indicador REC parpadeante
        add_date: Añadir fecha/hora
    """
    # Abrir video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video: {input_path}")
        return

    # Obtener propiedades
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video de entrada: {input_path}")
    print(f"Resolución: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Frames totales: {total_frames}")
    print(f"Intensidad VHS: {intensity}")
    print()

    # Crear writer (codec mp4v para compatibilidad)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Inicializar VHS
    vhs = VHSEffect(intensity=intensity)
    overlay = VHSTextOverlay()

    # Procesar frames
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calcular tiempo actual del video
        current_time = frame_count / fps

        # Aplicar efectos VHS
        processed = vhs.process_frame(frame, frame_time=current_time, fps=fps)

        # Aplicar overlays
        if add_timecode:
            processed = overlay.add_timecode(processed, current_time)

        if add_rec:
            # Parpadeo cada 15 frames
            show_circle = (frame_count % 30) < 15
            processed = overlay.add_rec_indicator(processed, show_circle=show_circle)

        if add_date:
            processed = overlay.add_date_stamp(processed)

        # Escribir frame
        writer.write(processed)

        # Mostrar progreso cada 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0

            print(f"\rProcesando: {progress:5.1f}% | "
                  f"Frame {frame_count}/{total_frames} | "
                  f"FPS: {fps_actual:.1f} | "
                  f"ETA: {eta:.0f}s", end="")

        frame_count += 1

    # Liberar recursos
    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    print(f"\n\nCompletado en {elapsed:.1f} segundos")
    print(f"✓ Guardado: {output_path}")


def main():
    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: No se encontró el video: {INPUT_VIDEO}")
        print("Por favor, coloca un video con ese nombre o modifica INPUT_VIDEO")
        return

    process_video(
        INPUT_VIDEO,
        OUTPUT_VIDEO,
        intensity=INTENSITY,
        add_timecode=ADD_TIMECODE,
        add_rec=ADD_REC_INDICATOR,
        add_date=ADD_DATE
    )


if __name__ == "__main__":
    main()
