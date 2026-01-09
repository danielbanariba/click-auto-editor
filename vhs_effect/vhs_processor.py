"""
VHS Processor - Clase principal que aplica todos los efectos VHS

Este módulo implementa el procesador principal de efectos VHS con 10 efectos
aplicados en orden específico para simular correctamente el proceso analógico.

ORDEN DE APLICACIÓN (CRÍTICO para realismo):
1. Color bleeding (simula proceso analógico de grabación)
2. Chromatic aberration
3. Horizontal wobble (antes de tracking errors)
4. Tracking errors (bandas ocasionales)
5. Position jitter
6. Vertical jitter (ocasional)
7. Scanlines
8. Noise/grain
9. Color grading
10. Sharpness reduction (último)
"""

import numpy as np
import cv2
import random
from typing import Tuple, Optional

from .color_utils import ColorProcessor, apply_chromatic_aberration, apply_color_grading


class PositionJitter:
    """
    Maneja el shake/jitter de posición con frecuencia controlada.

    El jitter en VHS real proviene de inestabilidades mecánicas:
    - Variaciones en velocidad del motor
    - Desgaste de rodillos
    - Tensión inconsistente de la cinta

    Características:
    - Amplitud típica: 2-8 píxeles
    - Frecuencia: 5-10 veces por segundo para jitter rápido
    - Más pronunciado en vertical que en horizontal (típico de VHS)
    """

    def __init__(self):
        self.last_jitter_time = 0.0
        self.current_offset_x = 0.0
        self.current_offset_y = 0.0

    def get_offset(self, frame_time: float, fps: float, intensity: float) -> Tuple[int, int]:
        """
        Calcula offset de jitter para el frame actual.

        El jitter NO es continuo - se actualiza a una frecuencia específica
        (típicamente 7-12 Hz), manteniendo el mismo offset entre updates.

        Args:
            frame_time: Tiempo actual en segundos
            fps: Frames por segundo del video
            intensity: 0.0-1.0

        Returns:
            (offset_x, offset_y) en píxeles
        """
        # Calcular frecuencia de jitter (7-12 Hz según intensity)
        freq = 7 + intensity * 5
        interval = 1.0 / freq

        # Verificar si es momento de nuevo jitter
        time_since_last = frame_time - self.last_jitter_time

        if time_since_last >= interval:
            # Generar nuevo offset aleatorio
            # Amplitud X: 2-8 píxeles
            amplitude_x = 2 + intensity * 6
            # Amplitud Y: 3-12 píxeles (MÁS que X, típico de VHS)
            amplitude_y = 3 + intensity * 9

            # Distribución uniforme para movimiento natural
            self.current_offset_x = random.uniform(-amplitude_x, amplitude_x)
            self.current_offset_y = random.uniform(-amplitude_y, amplitude_y)

            self.last_jitter_time = frame_time

        return int(self.current_offset_x), int(self.current_offset_y)

    def apply_jitter(self, frame: np.ndarray, offset: Tuple[int, int]) -> np.ndarray:
        """
        Aplica offset de jitter al frame.

        Args:
            frame: Array (H, W, 3)
            offset: (offset_x, offset_y) en píxeles

        Returns:
            Frame desplazado
        """
        offset_x, offset_y = offset

        if offset_x == 0 and offset_y == 0:
            return frame.copy()

        result = np.roll(frame, offset_x, axis=1)  # Horizontal
        result = np.roll(result, offset_y, axis=0)  # Vertical

        return result


class VHSEffect:
    """
    Clase principal que aplica todos los efectos VHS.

    Implementa 10 efectos en orden específico basándose en la física
    real del formato VHS para lograr máximo realismo.

    Attributes:
        intensity: Intensidad global 0.0 (leve) a 1.0 (muy dañado)
        frame_count: Contador interno de frames procesados
        jitter: Instancia de PositionJitter para manejo de shake
        color_processor: Instancia de ColorProcessor para color bleeding

    Example:
        >>> vhs = VHSEffect(intensity=0.7)
        >>> processed = vhs.process_frame(frame, frame_time=1.5)
    """

    def __init__(self, intensity: float = 0.5):
        """
        Inicializa el procesador VHS.

        Args:
            intensity: Intensidad global 0.0 (leve) a 1.0 (muy dañado)
        """
        self.intensity = np.clip(intensity, 0.0, 1.0)
        self.frame_count = 0
        self.jitter = PositionJitter()
        self.color_processor = ColorProcessor()

    def apply_horizontal_wobble(self, frame: np.ndarray, time_t: float) -> np.ndarray:
        """
        Aplica ondulación horizontal línea por línea.

        Simula el efecto "gelatina" causado por variaciones en la velocidad
        de la cinta y errores de tracking. Cada línea de escaneo se desplaza
        independientemente según una función sinusoidal.

        Características:
        - Ondulación basada en posición Y
        - Animado en el tiempo
        - Amplitud: 0-15 píxeles según intensity

        Args:
            frame: Array (H, W, 3)
            time_t: Tiempo actual en segundos (para animación)

        Returns:
            Frame con wobble aplicado
        """
        if self.intensity < 0.1:
            return frame.copy()

        H, W = frame.shape[:2]
        result = np.zeros_like(frame)

        # Calcular parámetros
        # Frecuencia: 30-60 ondulaciones verticales
        frequency = 30 + self.intensity * 30
        # Amplitud: 0-15 píxeles
        amplitude = self.intensity * 15
        # Velocidad de animación
        speed = 2.0

        for y in range(H):
            # Normalizar posición Y
            normalized_y = y / H

            # Calcular offset sinusoidal para esta línea
            offset = int(np.sin(normalized_y * frequency + time_t * speed) * amplitude)

            # Desplazar línea horizontalmente
            if offset > 0:
                # Desplazar a la derecha
                result[y, offset:] = frame[y, :-offset]
                result[y, :offset] = frame[y, 0:1]  # Fill edge con primer píxel
            elif offset < 0:
                # Desplazar a la izquierda
                result[y, :offset] = frame[y, -offset:]
                result[y, offset:] = frame[y, -1:]  # Fill edge con último píxel
            else:
                result[y] = frame[y]

        return result

    def apply_tracking_errors(self, frame: np.ndarray) -> np.ndarray:
        """
        Simula errores de tracking con bandas horizontales aleatorias.

        Los errores de tracking ocurren cuando hay desalineación entre
        los cabezales de video y las pistas grabadas en la cinta.
        Produce desplazamiento horizontal en bandas aleatorias.

        Características:
        - 2-5 bandas horizontales
        - Cada banda tiene 30% probabilidad de desplazarse
        - Offset con distribución gaussiana
        - Efecto OCASIONAL, no constante

        Args:
            frame: Array (H, W, 3)

        Returns:
            Frame con tracking errors
        """
        # Solo aplicar ocasionalmente (30% de probabilidad base)
        if random.random() > 0.3 * self.intensity:
            return frame.copy()

        H, W = frame.shape[:2]
        result = frame.copy()

        # Número aleatorio de bandas (2-5)
        num_bands = random.randint(2, 5)
        band_height = H // num_bands

        for band_idx in range(num_bands):
            # 30% probabilidad por banda
            if random.random() > 0.3:
                continue

            y_start = band_idx * band_height
            y_end = min((band_idx + 1) * band_height, H)

            # Offset con distribución gaussiana
            # Mayor intensity = más variación
            offset = int(random.gauss(0, 5 * self.intensity))

            if offset != 0:
                # Extraer y desplazar banda
                band = result[y_start:y_end, :]
                band_shifted = np.roll(band, offset, axis=1)
                result[y_start:y_end] = band_shifted

        return result

    def apply_vertical_jitter(self, frame: np.ndarray) -> np.ndarray:
        """
        Aplica salto vertical OCASIONAL.

        Simula problemas de V-sync o incomplete field starts que causan
        saltos verticales abruptos. Esto NO es continuo - solo ocurre
        ocasionalmente (típicamente 5-10% de frames).

        Características:
        - Saltos de 1-5 líneas
        - Muy breve (1 frame de duración)
        - NO suave, sino "jumpy"
        - Solo 5-10% de frames afectados

        Args:
            frame: Array (H, W, 3)

        Returns:
            Frame con salto vertical (o sin cambio)
        """
        # Solo aplicar OCASIONALMENTE (5-10% según intensity)
        probability = self.intensity * 0.1

        if random.random() > probability:
            return frame.copy()

        # Calcular salto (1-5 líneas)
        jump_lines = int(random.uniform(1, 3 + self.intensity * 2))

        # Dirección aleatoria
        direction = random.choice([-1, 1])
        offset = jump_lines * direction

        # Desplazar verticalmente
        return np.roll(frame, offset, axis=0)

    def apply_scanlines(self, frame: np.ndarray) -> np.ndarray:
        """
        Añade líneas de escaneo simulando video entrelazado.

        VHS es video entrelazado (480i/576i) donde las líneas pares
        e impares se muestran alternadamente. Esto crea líneas de
        escaneo visibles, especialmente en pausas.

        Args:
            frame: Array (H, W, 3)

        Returns:
            Frame con scanlines
        """
        H, W = frame.shape[:2]

        # Crear máscara de scanlines
        scanlines = np.ones((H, W), dtype=np.float32)

        # Líneas alternas más oscuras (factor 0.7-0.85)
        darkness = 0.7 + self.intensity * 0.15
        scanlines[::2] = darkness  # Líneas pares (0, 2, 4, ...)

        # Convertir frame a float para aplicar máscara
        result = frame.astype(np.float32)

        # Aplicar máscara a cada canal
        for c in range(3):
            result[:, :, c] *= scanlines

        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_noise(self, frame: np.ndarray) -> np.ndarray:
        """
        Añade ruido de luminancia y color.

        VHS tiene múltiples fuentes de ruido:
        - Ruido de luminancia: afecta todos los canales por igual (grain)
        - Ruido de color: independiente por canal (artefactos de croma)

        Características:
        - Distribución gaussiana (normal)
        - Ruido de luminancia siempre presente
        - Ruido de color solo con intensity > 0.3

        Args:
            frame: Array (H, W, 3)

        Returns:
            Frame con noise
        """
        H, W = frame.shape[:2]
        result = frame.astype(np.float32)

        # 1. Ruido de luminancia (afecta todo por igual)
        noise_strength_luma = 10 + self.intensity * 20  # 10-30
        noise_luma = np.random.normal(0, noise_strength_luma, (H, W))

        # Aplicar a todos los canales
        for c in range(3):
            result[:, :, c] += noise_luma

        # 2. Ruido de color (solo si intensity > 0.3)
        if self.intensity > 0.3:
            noise_strength_color = 5 * self.intensity

            for c in range(3):
                noise_color = np.random.normal(0, noise_strength_color, (H, W))
                result[:, :, c] += noise_color

        return np.clip(result, 0, 255).astype(np.uint8)

    def reduce_sharpness(self, frame: np.ndarray) -> np.ndarray:
        """
        Reduce sharpness con Gaussian blur.

        VHS tiene resolución limitada (~333x480 píxeles efectivos).
        Esto se simula con un blur sutil.

        Args:
            frame: Array (H, W, 3)

        Returns:
            Frame con sharpness reducida
        """
        # Calcular blur (1-3)
        blur_amount = 1 + int(self.intensity * 2)

        # Kernel size debe ser impar (3, 5, 7)
        kernel_size = blur_amount * 2 + 1

        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    def process_frame(self, frame: np.ndarray, frame_time: float = 0.0,
                      fps: float = 30.0) -> np.ndarray:
        """
        Procesa un frame con todos los efectos VHS EN ORDEN ESPECÍFICO.

        ORDEN DE APLICACIÓN (IMPORTANTE para realismo):
        1. Color bleeding (debe ir primero - simula proceso analógico)
        2. Chromatic aberration
        3. Horizontal wobble (antes de tracking errors)
        4. Tracking errors (bandas ocasionales)
        5. Position jitter
        6. Vertical jitter (ocasional)
        7. Scanlines
        8. Noise/grain
        9. Color grading
        10. Sharpness reduction (último)

        Args:
            frame: Array BGR (H, W, 3) uint8
            frame_time: Tiempo en segundos (para efectos animados)
            fps: Frames por segundo (para cálculo de jitter)

        Returns:
            Frame procesado con efecto VHS completo
        """
        self.frame_count += 1
        result = frame.copy()

        # 1. Color bleeding (primero - simula grabación analógica)
        result = self.color_processor.apply_color_bleeding(result, self.intensity)

        # 2. Chromatic aberration
        result = apply_chromatic_aberration(result, self.intensity)

        # 3. Horizontal wobble
        result = self.apply_horizontal_wobble(result, frame_time)

        # 4. Tracking errors (ocasional)
        result = self.apply_tracking_errors(result)

        # 5. Position jitter
        jitter_offset = self.jitter.get_offset(frame_time, fps, self.intensity)
        result = self.jitter.apply_jitter(result, jitter_offset)

        # 6. Vertical jitter (ocasional)
        result = self.apply_vertical_jitter(result)

        # 7. Scanlines
        result = self.apply_scanlines(result)

        # 8. Noise/grain
        result = self.apply_noise(result)

        # 9. Color grading
        result = apply_color_grading(result, self.intensity)

        # 10. Sharpness reduction (último)
        result = self.reduce_sharpness(result)

        return result


def process_image(input_path: str, output_path: str, intensity: float = 0.5) -> None:
    """
    Procesa una imagen con efecto VHS.

    Args:
        input_path: Ruta al archivo de imagen de entrada
        output_path: Ruta al archivo de imagen de salida
        intensity: 0.0-1.0

    Raises:
        FileNotFoundError: Si el archivo de entrada no existe
        ValueError: Si el archivo no es una imagen válida
    """
    # Leer imagen
    frame = cv2.imread(input_path)

    if frame is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {input_path}")

    # Procesar
    vhs = VHSEffect(intensity=intensity)
    processed = vhs.process_frame(frame, frame_time=0.0)

    # Guardar
    cv2.imwrite(output_path, processed)
    print(f"Guardado: {output_path}")


def process_video(input_path: str, output_path: str,
                  intensity: float = 0.5,
                  show_progress: bool = True) -> None:
    """
    Procesa un video completo con efecto VHS.

    Args:
        input_path: Ruta al video de entrada
        output_path: Ruta al video de salida
        intensity: 0.0-1.0
        show_progress: Mostrar barra de progreso

    Raises:
        FileNotFoundError: Si el archivo de entrada no existe
        ValueError: Si el archivo no es un video válido
    """
    # Abrir video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {input_path}")

    # Obtener propiedades
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Crear writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Inicializar VHS
    vhs = VHSEffect(intensity=intensity)

    # Procesar frames
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calcular tiempo actual
        current_time = frame_count / fps

        # Aplicar efectos VHS
        processed = vhs.process_frame(frame, frame_time=current_time, fps=fps)

        # Escribir frame
        writer.write(processed)

        # Mostrar progreso
        if show_progress and frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"\rProcesando: {progress:.1f}% ({frame_count}/{total_frames})", end="")

        frame_count += 1

    # Liberar recursos
    cap.release()
    writer.release()

    if show_progress:
        print(f"\nCompletado: {output_path}")
