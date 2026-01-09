"""
Color Utilities GPU - Procesamiento de color VHS acelerado por CUDA

Usa CuPy para operaciones en GPU con fallback automático a NumPy.
CuPy alcanza 100-204x speedup sobre NumPy en operaciones de arrays.

Requisitos:
    pip install cupy-cuda12x  # o cupy-cuda11x según tu versión CUDA
"""

import numpy as np
import cv2
from typing import Tuple, Optional

# Intentar importar CuPy, con fallback a NumPy
try:
    import cupy as cp
    HAS_CUPY = True
    print("[GPU] CuPy disponible - usando aceleración CUDA")
except ImportError:
    cp = np  # Fallback a NumPy
    HAS_CUPY = False
    print("[CPU] CuPy no disponible - usando NumPy")


class ColorProcessorGPU:
    """
    Procesador de color VHS acelerado por GPU.

    Operaciones en VRAM:
    - Conversión RGB ↔ YIQ usando matrices NTSC
    - Color bleeding con blur horizontal en canales I/Q
    - Chromatic aberration con desplazamiento de canales
    """

    def __init__(self):
        # Matrices de transformación en GPU
        self.RGB_TO_YIQ = cp.array([
            [0.299,  0.587,  0.114],
            [0.596, -0.274, -0.322],
            [0.211, -0.523,  0.312]
        ], dtype=cp.float32)

        self.YIQ_TO_RGB = cp.array([
            [1.0,  0.956,  0.621],
            [1.0, -0.272, -0.647],
            [1.0, -1.106,  1.703]
        ], dtype=cp.float32)

    def rgb_to_yiq_gpu(self, rgb_gpu: cp.ndarray) -> cp.ndarray:
        """
        Convierte RGB a YIQ en GPU.

        Args:
            rgb_gpu: Array CuPy (H, W, 3) float32 0-1 en RGB

        Returns:
            yiq_gpu: Array CuPy (H, W, 3) en espacio YIQ
        """
        original_shape = rgb_gpu.shape
        rgb_flat = rgb_gpu.reshape(-1, 3)
        yiq_flat = cp.dot(rgb_flat, self.RGB_TO_YIQ.T)
        return yiq_flat.reshape(original_shape)

    def yiq_to_rgb_gpu(self, yiq_gpu: cp.ndarray) -> cp.ndarray:
        """
        Convierte YIQ a RGB en GPU.

        Args:
            yiq_gpu: Array CuPy (H, W, 3) en espacio YIQ

        Returns:
            rgb_gpu: Array CuPy (H, W, 3) float32 0-1 en RGB
        """
        original_shape = yiq_gpu.shape
        yiq_flat = yiq_gpu.reshape(-1, 3)
        rgb_flat = cp.dot(yiq_flat, self.YIQ_TO_RGB.T)
        rgb = rgb_flat.reshape(original_shape)
        return cp.clip(rgb, 0.0, 1.0)

    def apply_color_bleeding_gpu(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """
        Color bleeding acelerado por GPU.

        Pipeline GPU:
        1. Upload a VRAM: BGR → RGB float
        2. RGB → YIQ en GPU
        3. Blur horizontal en I/Q (separable Gaussian)
        4. YIQ → RGB en GPU
        5. Download a RAM: RGB → BGR uint8

        Args:
            frame: Array BGR (H, W, 3) uint8
            intensity: 0.0-1.0

        Returns:
            Frame con color bleeding (BGR uint8)
        """
        if not HAS_CUPY:
            # Fallback a CPU
            from .color_utils import ColorProcessor
            return ColorProcessor().apply_color_bleeding(frame, intensity)

        # Upload a GPU
        frame_gpu = cp.asarray(frame)

        # BGR → RGB float32 (en GPU)
        rgb_gpu = cp.flip(frame_gpu, axis=2).astype(cp.float32) / 255.0

        # RGB → YIQ
        yiq_gpu = self.rgb_to_yiq_gpu(rgb_gpu)

        # Separar canales
        y = yiq_gpu[:, :, 0]
        i = yiq_gpu[:, :, 1]
        q = yiq_gpu[:, :, 2]

        # Blur horizontal en I/Q
        # CuPy no tiene GaussianBlur, usamos convolución con kernel
        blur_amount = int(5 + intensity * 20)
        kernel_size = blur_amount * 2 + 1

        # Crear kernel Gaussiano 1D horizontal
        sigma = blur_amount / 3.0
        x = cp.arange(kernel_size) - blur_amount
        kernel = cp.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()  # Normalizar
        kernel = kernel.reshape(1, -1)  # Shape (1, kernel_size)

        # Convolución separable (solo horizontal)
        from cupyx.scipy.ndimage import convolve1d
        i_blurred = convolve1d(i, kernel.flatten(), axis=1, mode='reflect')
        q_blurred = convolve1d(q, kernel.flatten(), axis=1, mode='reflect')

        # Recombinar
        yiq_processed = cp.stack([y, i_blurred, q_blurred], axis=2)

        # YIQ → RGB
        rgb_result = self.yiq_to_rgb_gpu(yiq_processed)

        # RGB → BGR uint8 y download
        bgr_result = cp.flip(rgb_result, axis=2)
        bgr_uint8 = (bgr_result * 255).astype(cp.uint8)

        return cp.asnumpy(bgr_uint8)


def apply_chromatic_aberration_gpu(frame: np.ndarray, intensity: float) -> np.ndarray:
    """
    Aberración cromática acelerada por GPU.

    Usa cp.roll para desplazamiento eficiente en VRAM.

    Args:
        frame: Array BGR (H, W, 3) uint8
        intensity: 0.0-1.0

    Returns:
        Frame con aberración cromática (BGR uint8)
    """
    if not HAS_CUPY:
        from .color_utils import apply_chromatic_aberration
        return apply_chromatic_aberration(frame, intensity)

    offset = int(2 + intensity * 4)
    if offset == 0:
        return frame.copy()

    # Upload a GPU
    frame_gpu = cp.asarray(frame)

    # Separar canales en GPU
    b = frame_gpu[:, :, 0]
    g = frame_gpu[:, :, 1]
    r = frame_gpu[:, :, 2]

    # Desplazar canales en GPU
    b_shifted = cp.roll(b, -offset, axis=1)
    b_shifted[:, -offset:] = b[:, -offset:]  # Limpiar wrap

    r_shifted = cp.roll(r, offset, axis=1)
    r_shifted[:, :offset] = r[:, :offset]  # Limpiar wrap

    # Recombinar en GPU
    result_gpu = cp.stack([b_shifted, g, r_shifted], axis=2)

    return cp.asnumpy(result_gpu)


def apply_color_grading_gpu(frame: np.ndarray, intensity: float) -> np.ndarray:
    """
    Color grading VHS acelerado por GPU.

    Operaciones vectorizadas en VRAM:
    - Elevar negros
    - Reducir contraste
    - Desaturar
    - Tinte cálido

    Args:
        frame: Array BGR (H, W, 3) uint8
        intensity: 0.0-1.0

    Returns:
        Frame con color grading (BGR uint8)
    """
    if not HAS_CUPY:
        from .color_utils import apply_color_grading
        return apply_color_grading(frame, intensity)

    # Upload a GPU
    frame_gpu = cp.asarray(frame).astype(cp.float32)

    # 1. Elevar negros
    black_lift = 10 + intensity * 10
    frame_gpu += black_lift

    # 2. Reducir contraste
    mid_gray = 127.5
    contrast_factor = 0.9 - intensity * 0.2
    frame_gpu = mid_gray + (frame_gpu - mid_gray) * contrast_factor

    # 3. Desaturar (calcular grayscale en GPU)
    # Pesos estándar: 0.114*B + 0.587*G + 0.299*R
    gray_gpu = (
        frame_gpu[:, :, 0] * 0.114 +
        frame_gpu[:, :, 1] * 0.587 +
        frame_gpu[:, :, 2] * 0.299
    )
    gray_3ch = cp.stack([gray_gpu, gray_gpu, gray_gpu], axis=2)

    desaturation = 0.15 + intensity * 0.15
    frame_gpu = frame_gpu * (1 - desaturation) + gray_3ch * desaturation

    # 4. Tinte cálido
    frame_gpu[:, :, 0] *= 0.95  # Menos azul
    frame_gpu[:, :, 2] *= 1.05  # Más rojo

    # Clip y download
    result = cp.clip(frame_gpu, 0, 255).astype(cp.uint8)
    return cp.asnumpy(result)


# Instancia global del procesador GPU
_gpu_processor: Optional[ColorProcessorGPU] = None

def get_gpu_processor() -> ColorProcessorGPU:
    """Obtiene instancia singleton del procesador GPU."""
    global _gpu_processor
    if _gpu_processor is None:
        _gpu_processor = ColorProcessorGPU()
    return _gpu_processor
