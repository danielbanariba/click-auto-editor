"""
VHS Displacement Map - Genera mapas de desplazamiento para distorsión temporal avanzada

Los displacement maps permiten crear efectos de distorsión más complejos
que simulan problemas severos de cinta como:
- Temporal smearing (diferentes partes del frame en diferentes tiempos)
- Warping severo
- Errores de sincronización graves
"""

import numpy as np
from scipy import ndimage
from typing import Tuple


class VHSDisplacementMap:
    """
    Genera mapas de desplazamiento para distorsión temporal avanzada.

    Un displacement map es una imagen en escala de grises donde:
    - Valores claros (>128): desplazan píxeles hacia la derecha
    - Valores oscuros (<128): desplazan píxeles hacia la izquierda
    - Valor 128: sin desplazamiento

    Esto permite crear efectos de "temporal smearing" donde diferentes
    partes del frame parecen estar en diferentes momentos temporales.

    Attributes:
        intensity: Intensidad del desplazamiento (0.0-1.0)
    """

    def __init__(self, intensity: float = 0.5):
        """
        Inicializa el generador de displacement maps.

        Args:
            intensity: Intensidad del efecto (0.0-1.0)
        """
        self.intensity = np.clip(intensity, 0.0, 1.0)

    def create_displacement_map(self, H: int, W: int, time_t: float) -> np.ndarray:
        """
        Genera mapa de desplazamiento animado.

        El mapa combina:
        1. Patrones sinusoidales superpuestos (para movimiento suave)
        2. Ruido aleatorio (para irregularidad)

        El patrón se anima en el tiempo para crear movimiento continuo.

        Args:
            H: Altura del frame
            W: Ancho del frame
            time_t: Tiempo actual en segundos (para animación)

        Returns:
            Array (H, W) con valores 0.0-1.0
        """
        # Crear grids de coordenadas normalizadas
        y = np.linspace(0, 1, H)
        x = np.linspace(0, 1, W)
        Y, X = np.meshgrid(y, x, indexing='ij')

        # Crear patrones sinusoidales superpuestos
        # Patrón 1: ondas verticales (simulan tracking)
        pattern1 = np.sin(Y * 15 * np.pi + time_t * 2)

        # Patrón 2: ondas horizontales más lentas
        pattern2 = np.sin(X * 8 * np.pi - time_t * 1.5)

        # Combinar patrones con diferentes pesos
        pattern = (pattern1 + pattern2 * 0.5) / 1.5

        # Añadir ruido para irregularidad
        noise = np.random.random((H, W)) * 0.3

        # Combinar patrón y ruido
        # 70% patrón, 30% ruido
        displacement = (pattern * 0.5 + 0.5) * 0.7 + noise * 0.3

        return displacement.astype(np.float32)

    def create_band_displacement(self, H: int, W: int, num_bands: int = 5) -> np.ndarray:
        """
        Genera displacement map con bandas horizontales aleatorias.

        Simula errores de tracking severos donde diferentes bandas
        horizontales están desplazadas de manera independiente.

        Args:
            H: Altura del frame
            W: Ancho del frame
            num_bands: Número de bandas horizontales

        Returns:
            Array (H, W) con valores 0.0-1.0
        """
        displacement = np.ones((H, W), dtype=np.float32) * 0.5  # Neutral

        band_height = H // num_bands

        for band_idx in range(num_bands):
            y_start = band_idx * band_height
            y_end = min((band_idx + 1) * band_height, H)

            # Valor aleatorio para esta banda
            band_value = np.random.uniform(0.3, 0.7)
            displacement[y_start:y_end, :] = band_value

        return displacement

    def apply_displacement(self, frame: np.ndarray,
                          displacement_map: np.ndarray) -> np.ndarray:
        """
        Aplica displacement map al frame.

        Proceso:
        1. Normalizar displacement a rango de píxeles
        2. Calcular nuevas coordenadas para cada píxel
        3. Interpolar valores de píxeles en nuevas posiciones

        Args:
            frame: Array (H, W, 3) BGR
            displacement_map: Array (H, W) valores 0.0-1.0

        Returns:
            Frame con displacement aplicado
        """
        H, W = frame.shape[:2]

        # Calcular intensidad en píxeles (10-30 según intensity)
        intensity_px = 10 + self.intensity * 20

        # Normalizar displacement a rango -intensity a +intensity
        disp_normalized = (displacement_map - 0.5) * 2.0 * intensity_px

        # Crear coordenadas base
        y, x = np.mgrid[0:H, 0:W]

        # Aplicar desplazamiento HORIZONTAL solamente
        x_displaced = x + disp_normalized

        # Clamp a rango válido
        x_displaced = np.clip(x_displaced, 0, W - 1)

        # Mapear píxeles usando interpolación
        result = np.zeros_like(frame)

        for c in range(3):
            result[:, :, c] = ndimage.map_coordinates(
                frame[:, :, c].astype(np.float32),
                [y, x_displaced],
                order=1,  # Linear interpolation
                mode='nearest'
            )

        return result.astype(np.uint8)

    def apply_animated_displacement(self, frame: np.ndarray,
                                    time_t: float) -> np.ndarray:
        """
        Aplica displacement animado al frame.

        Método conveniente que genera el displacement map y lo aplica
        en un solo paso.

        Args:
            frame: Array (H, W, 3) BGR
            time_t: Tiempo actual en segundos

        Returns:
            Frame con displacement animado aplicado
        """
        H, W = frame.shape[:2]
        displacement_map = self.create_displacement_map(H, W, time_t)
        return self.apply_displacement(frame, displacement_map)


class VHSTemporalDisplacement:
    """
    Implementa time displacement donde diferentes partes del frame
    muestran diferentes momentos temporales.

    Requiere buffer de frames anteriores para funcionar correctamente.
    """

    def __init__(self, buffer_size: int = 10, intensity: float = 0.5):
        """
        Inicializa el temporal displacement.

        Args:
            buffer_size: Número de frames anteriores a mantener
            intensity: Intensidad del efecto (0.0-1.0)
        """
        self.buffer_size = buffer_size
        self.intensity = np.clip(intensity, 0.0, 1.0)
        self.frame_buffer = []

    def add_frame(self, frame: np.ndarray) -> None:
        """
        Añade un frame al buffer.

        Args:
            frame: Frame a añadir
        """
        self.frame_buffer.append(frame.copy())

        # Mantener solo buffer_size frames
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

    def apply_temporal_displacement(self, displacement_map: np.ndarray) -> np.ndarray:
        """
        Aplica temporal displacement usando el buffer de frames.

        Diferentes regiones del frame mostrarán diferentes momentos
        temporales basándose en el displacement map:
        - Valores altos: frames más recientes
        - Valores bajos: frames más antiguos

        Args:
            displacement_map: Array (H, W) valores 0.0-1.0

        Returns:
            Frame con temporal displacement aplicado
        """
        if len(self.frame_buffer) < 2:
            return self.frame_buffer[-1] if self.frame_buffer else None

        H, W = self.frame_buffer[-1].shape[:2]
        result = np.zeros((H, W, 3), dtype=np.float32)

        num_frames = len(self.frame_buffer)

        for y in range(H):
            for x in range(W):
                # Calcular índice de frame basado en displacement
                disp_value = displacement_map[y, x]
                frame_idx = int(disp_value * (num_frames - 1))
                frame_idx = np.clip(frame_idx, 0, num_frames - 1)

                result[y, x] = self.frame_buffer[frame_idx][y, x]

        return result.astype(np.uint8)
