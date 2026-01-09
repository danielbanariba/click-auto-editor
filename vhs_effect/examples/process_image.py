#!/usr/bin/env python3
"""
Ejemplo: Procesar una imagen con efecto VHS

Este script demuestra cómo aplicar el efecto VHS a una imagen estática.
"""

import sys
import os

# Añadir directorio padre al path para importar vhs_effect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
from vhs_effect import VHSEffect

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

INPUT_IMAGE = "foto.jpg"  # Cambiar por tu imagen
OUTPUT_IMAGE = "foto_vhs.jpg"
INTENSITY = 0.7  # 0.0 (leve) a 1.0 (muy dañado)

# =============================================================================
# PROCESAMIENTO
# =============================================================================

def main():
    # Verificar que existe el archivo
    if not os.path.exists(INPUT_IMAGE):
        # Crear imagen de prueba si no existe
        print(f"Creando imagen de prueba...")
        test_frame = create_test_image()
        cv2.imwrite(INPUT_IMAGE, test_frame)
        print(f"Imagen de prueba creada: {INPUT_IMAGE}")

    # Leer imagen
    frame = cv2.imread(INPUT_IMAGE)

    if frame is None:
        print(f"Error: No se pudo leer la imagen: {INPUT_IMAGE}")
        return

    print(f"Procesando: {INPUT_IMAGE}")
    print(f"Resolución: {frame.shape[1]}x{frame.shape[0]}")
    print(f"Intensidad VHS: {INTENSITY}")

    # Crear procesador VHS
    vhs = VHSEffect(intensity=INTENSITY)

    # Procesar (frame_time=0 para imagen estática)
    processed = vhs.process_frame(frame, frame_time=0.0)

    # Guardar resultado
    cv2.imwrite(OUTPUT_IMAGE, processed)
    print(f"✓ Guardado: {OUTPUT_IMAGE}")


def create_test_image(width=640, height=480):
    """Crea una imagen de prueba con formas coloridas"""
    import numpy as np

    # Fondo degradado
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        frame[y, :] = [100 + y//4, 150 - y//6, 200 - y//3]

    # Rectángulo azul
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 100, 50), -1)

    # Círculo verde
    cv2.circle(frame, (450, 240), 80, (50, 255, 100), -1)

    # Texto
    cv2.putText(frame, "VHS TEST", (150, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    return frame


if __name__ == "__main__":
    main()
