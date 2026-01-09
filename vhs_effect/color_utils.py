"""
Color Utilities - Conversiones RGB↔YIQ y procesamiento de color VHS

VHS usa el sistema "color-under" donde la crominancia tiene ~8x menos
resolución horizontal que la luminancia debido a anchos de banda limitados:
- Luminancia: ~3 MHz
- Crominancia: ~500 kHz

Este módulo implementa las conversiones de color necesarias para simular
este proceso analógico correctamente.
"""

import numpy as np
import cv2
from typing import Tuple


class ColorProcessor:
    """
    Implementa conversión RGB ↔ YIQ y color bleeding para efecto VHS.

    El espacio de color YIQ es el estándar NTSC usado en video analógico:
    - Y: Luminancia (brillo)
    - I: In-phase (crominancia naranja-cian)
    - Q: Quadrature (crominancia verde-magenta)
    """

    # Matriz de transformación RGB a YIQ (estándar ITU-R BT.470)
    RGB_TO_YIQ = np.array([
        [0.299,  0.587,  0.114],   # Y
        [0.596, -0.274, -0.322],   # I
        [0.211, -0.523,  0.312]    # Q
    ], dtype=np.float32)

    # Matriz inversa YIQ a RGB
    YIQ_TO_RGB = np.array([
        [1.0,  0.956,  0.621],   # R
        [1.0, -0.272, -0.647],   # G
        [1.0, -1.106,  1.703]    # B
    ], dtype=np.float32)

    @staticmethod
    def rgb_to_yiq(rgb: np.ndarray) -> np.ndarray:
        """
        Convierte RGB a YIQ usando matriz de transformación NTSC.

        Matriz de transformación:
        Y = 0.299*R + 0.587*G + 0.114*B
        I = 0.596*R - 0.274*G - 0.322*B
        Q = 0.211*R - 0.523*G + 0.312*B

        Args:
            rgb: Array (H, W, 3) con valores 0.0-1.0 (float) en orden RGB

        Returns:
            yiq: Array (H, W, 3) en espacio YIQ
                 Y: 0.0-1.0
                 I: -0.596 a 0.596
                 Q: -0.523 a 0.523
        """
        # Reshape para multiplicación matricial: (H*W, 3)
        original_shape = rgb.shape
        rgb_flat = rgb.reshape(-1, 3)

        # Aplicar transformación: YIQ = RGB @ M^T
        yiq_flat = rgb_flat @ ColorProcessor.RGB_TO_YIQ.T

        # Restaurar shape original
        return yiq_flat.reshape(original_shape)

    @staticmethod
    def yiq_to_rgb(yiq: np.ndarray) -> np.ndarray:
        """
        Convierte YIQ de vuelta a RGB.

        Matriz inversa:
        R = Y + 0.956*I + 0.621*Q
        G = Y - 0.272*I - 0.647*Q
        B = Y - 1.106*I + 1.703*Q

        Args:
            yiq: Array (H, W, 3) en espacio YIQ

        Returns:
            rgb: Array (H, W, 3) con valores 0.0-1.0 en orden RGB
        """
        original_shape = yiq.shape
        yiq_flat = yiq.reshape(-1, 3)

        # Aplicar transformación inversa
        rgb_flat = yiq_flat @ ColorProcessor.YIQ_TO_RGB.T

        # Restaurar shape y clip a rango válido
        rgb = rgb_flat.reshape(original_shape)
        return np.clip(rgb, 0.0, 1.0)

    def apply_color_bleeding(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """
        Aplica color bleeding simulando baja resolución de crominancia VHS.

        En VHS real, la crominancia tiene ~8x menos resolución que la luminancia.
        Esto se simula aplicando blur HORIZONTAL solo a los canales de color (I, Q),
        manteniendo la luminancia (Y) en alta resolución.

        Proceso:
        1. Convertir BGR → RGB → YIQ
        2. Extraer Y (luminancia), I y Q (crominancia)
        3. Mantener Y en alta resolución
        4. Aplicar blur horizontal SOLO a canales I y Q
        5. Recombinar Y + I_blurred + Q_blurred
        6. Convertir YIQ → RGB → BGR

        Args:
            frame: Array BGR (H, W, 3) uint8
            intensity: 0.0-1.0 (controla cantidad de blur)

        Returns:
            Frame con color bleeding aplicado (BGR uint8)
        """
        # Convertir BGR uint8 → RGB float 0-1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Convertir a YIQ
        yiq = self.rgb_to_yiq(rgb)

        # Separar canales
        y = yiq[:, :, 0]  # Luminancia - mantener intacta
        i = yiq[:, :, 1]  # Crominancia I
        q = yiq[:, :, 2]  # Crominancia Q

        # Calcular cantidad de blur (5-25 píxeles según intensity)
        blur_amount = int(5 + intensity * 20)

        # Kernel horizontal para blur de crominancia
        # (blur_amount*2+1, 1) = solo horizontal
        kernel_size = blur_amount * 2 + 1

        # Aplicar blur HORIZONTAL solo a I y Q (simulando baja resolución de croma)
        i_blurred = cv2.GaussianBlur(i, (kernel_size, 1), 0)
        q_blurred = cv2.GaussianBlur(q, (kernel_size, 1), 0)

        # Recombinar: Y original + I/Q con blur
        yiq_processed = np.stack([y, i_blurred, q_blurred], axis=2)

        # Convertir de vuelta a RGB
        rgb_result = self.yiq_to_rgb(yiq_processed)

        # Convertir RGB float → BGR uint8
        bgr_result = cv2.cvtColor(
            (rgb_result * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR
        )

        return bgr_result


def apply_chromatic_aberration(frame: np.ndarray, intensity: float) -> np.ndarray:
    """
    Desplaza canales de color horizontalmente simulando aberración cromática.

    La aberración cromática ocurre en lentes vintage y equipos analógicos
    donde los diferentes colores de luz se enfocan en puntos ligeramente
    diferentes, causando franjas de color en los bordes.

    Algoritmo:
    1. Calcular offset: 2-6 píxeles según intensity
    2. Separar canales BGR
    3. Desplazar azul (B) a la izquierda
    4. Desplazar rojo (R) a la derecha
    5. Verde (G) permanece sin cambios
    6. Recombinar

    Args:
        frame: Array BGR (H, W, 3) uint8
        intensity: 0.0-1.0

    Returns:
        Frame con aberración cromática (BGR uint8)
    """
    # Calcular offset en píxeles (2-6)
    offset = int(2 + intensity * 4)

    if offset == 0:
        return frame.copy()

    # Separar canales BGR
    b, g, r = cv2.split(frame)

    # Desplazar canal azul a la izquierda
    b_shifted = np.roll(b, -offset, axis=1)
    # Limpiar borde derecho (donde hizo wrap)
    b_shifted[:, -offset:] = b[:, -offset:]

    # Desplazar canal rojo a la derecha
    r_shifted = np.roll(r, offset, axis=1)
    # Limpiar borde izquierdo
    r_shifted[:, :offset] = r[:, :offset]

    # Canal verde permanece sin cambios
    # Recombinar
    return cv2.merge([b_shifted, g, r_shifted])


def apply_color_grading(frame: np.ndarray, intensity: float) -> np.ndarray:
    """
    Aplica color grading típico de VHS.

    Características del color VHS:
    - Negros elevados (no hay negros puros en VHS)
    - Contraste reducido
    - Ligera desaturación
    - Tinte cálido/amarillento

    Algoritmo:
    1. Elevar negros: añadir offset 10-20
    2. Reducir contraste: factor 0.7-0.9
    3. Desaturar: mezclar con grayscale 15-30%
    4. Tinte cálido: menos azul, más rojo

    Args:
        frame: Array BGR (H, W, 3) uint8
        intensity: 0.0-1.0

    Returns:
        Frame con color grading VHS (BGR uint8)
    """
    # Convertir a float para procesamiento
    frame_float = frame.astype(np.float32)

    # 1. Elevar negros (crush blacks)
    # VHS nunca tiene negro puro (0,0,0)
    black_lift = 10 + intensity * 10  # 10-20
    frame_float += black_lift

    # 2. Reducir contraste
    mid_gray = 127.5
    contrast_factor = 0.9 - intensity * 0.2  # 0.7-0.9
    frame_float = mid_gray + (frame_float - mid_gray) * contrast_factor

    # 3. Desaturar
    # Convertir a grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)

    # Mezclar con original (15-30% grayscale)
    desaturation = 0.15 + intensity * 0.15
    frame_float = frame_float * (1 - desaturation) + gray_3ch * desaturation

    # 4. Aplicar tinte cálido
    # Reducir azul, aumentar rojo
    frame_float[:, :, 0] *= 0.95  # Menos azul
    frame_float[:, :, 2] *= 1.05  # Más rojo

    # Clip y convertir a uint8
    return np.clip(frame_float, 0, 255).astype(np.uint8)
