"""
VHS Text Overlays - Overlays de texto estilo VCR

Implementa elementos visuales típicos de grabadoras VHS como:
- Timecode (HH:MM:SS)
- Indicador REC parpadeante
- Indicador PLAY
- Fecha/hora estilo VCR
"""

import cv2
import numpy as np
from datetime import timedelta, datetime
from typing import Tuple, Optional


class VHSTextOverlay:
    """
    Overlays de texto estilo VCR de los años 80-90.

    Los VCRs típicamente mostraban:
    - Timecode en esquina superior izquierda
    - Indicador REC con círculo rojo parpadeante
    - Indicador PLAY durante reproducción
    - Fecha/hora de grabación

    El estilo usa fuentes monoespaciadas, colores saturados
    (cyan, rojo, blanco) y a veces fondo semi-transparente.
    """

    # Colores típicos de VCR (BGR)
    COLOR_WHITE = (200, 200, 200)
    COLOR_RED = (0, 0, 255)
    COLOR_CYAN = (255, 255, 0)
    COLOR_GREEN = (0, 255, 0)

    # Fuente monoespaciada
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2

    @staticmethod
    def add_timecode(frame: np.ndarray, time_seconds: float,
                     position: Tuple[int, int] = (20, 40),
                     with_background: bool = True) -> np.ndarray:
        """
        Añade timecode VCR (HH:MM:SS).

        Args:
            frame: Array BGR (H, W, 3)
            time_seconds: Tiempo en segundos a mostrar
            position: (x, y) posición del texto
            with_background: Añadir fondo semi-transparente

        Returns:
            Frame con timecode añadido
        """
        result = frame.copy()

        # Convertir tiempo a string HH:MM:SS
        time_str = str(timedelta(seconds=int(time_seconds)))
        if time_str.startswith("0:"):
            time_str = "0" + time_str  # 0:05:30 -> 00:05:30

        # Calcular tamaño del texto
        (text_width, text_height), baseline = cv2.getTextSize(
            time_str,
            VHSTextOverlay.FONT,
            VHSTextOverlay.FONT_SCALE,
            VHSTextOverlay.FONT_THICKNESS
        )

        x, y = position

        # Añadir fondo semi-transparente si se solicita
        if with_background:
            padding = 5
            overlay = result.copy()
            cv2.rectangle(
                overlay,
                (x - padding, y - text_height - padding),
                (x + text_width + padding, y + baseline + padding),
                (0, 0, 0),  # Negro
                -1  # Filled
            )
            # Blend con 60% opacidad
            result = cv2.addWeighted(overlay, 0.6, result, 0.4, 0)

        # Añadir texto
        cv2.putText(
            result,
            time_str,
            position,
            VHSTextOverlay.FONT,
            VHSTextOverlay.FONT_SCALE,
            VHSTextOverlay.COLOR_WHITE,
            VHSTextOverlay.FONT_THICKNESS
        )

        return result

    @staticmethod
    def add_rec_indicator(frame: np.ndarray,
                          position: Optional[Tuple[int, int]] = None,
                          show_circle: bool = True) -> np.ndarray:
        """
        Añade indicador REC con círculo rojo.

        El indicador REC típico de VCR consiste en:
        - Círculo rojo sólido (parpadea)
        - Texto "REC" en rojo

        Nota: El parpadeo se controla desde el código que llama
        (mostrar/ocultar cada 15 frames típicamente).

        Args:
            frame: Array BGR (H, W, 3)
            position: (x, y) posición, si None usa esquina superior derecha
            show_circle: Mostrar círculo rojo (para efecto parpadeo)

        Returns:
            Frame con indicador REC
        """
        result = frame.copy()
        H, W = frame.shape[:2]

        # Posición por defecto: esquina superior derecha
        if position is None:
            position = (W - 100, 35)

        x, y = position

        # Dibujar círculo rojo (si show_circle=True)
        if show_circle:
            circle_center = (x - 15, y - 5)
            cv2.circle(result, circle_center, 8, VHSTextOverlay.COLOR_RED, -1)

        # Añadir texto "REC"
        cv2.putText(
            result,
            "REC",
            position,
            VHSTextOverlay.FONT,
            VHSTextOverlay.FONT_SCALE,
            VHSTextOverlay.COLOR_RED,
            VHSTextOverlay.FONT_THICKNESS
        )

        return result

    @staticmethod
    def add_play_indicator(frame: np.ndarray,
                           position: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Añade indicador PLAY con triángulo.

        Args:
            frame: Array BGR (H, W, 3)
            position: (x, y) posición, si None usa esquina superior derecha

        Returns:
            Frame con indicador PLAY
        """
        result = frame.copy()
        H, W = frame.shape[:2]

        if position is None:
            position = (W - 100, 35)

        x, y = position

        # Dibujar triángulo (símbolo de play)
        triangle_pts = np.array([
            [x - 20, y - 12],
            [x - 20, y + 5],
            [x - 5, y - 3]
        ], np.int32)
        cv2.fillPoly(result, [triangle_pts], VHSTextOverlay.COLOR_GREEN)

        # Añadir texto "PLAY"
        cv2.putText(
            result,
            "PLAY",
            position,
            VHSTextOverlay.FONT,
            VHSTextOverlay.FONT_SCALE,
            VHSTextOverlay.COLOR_GREEN,
            VHSTextOverlay.FONT_THICKNESS
        )

        return result

    @staticmethod
    def add_date_stamp(frame: np.ndarray,
                       date: Optional[datetime] = None,
                       position: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Añade fecha/hora estilo VCR.

        Formato típico: "JAN 15 1987  3:45 PM"

        Args:
            frame: Array BGR (H, W, 3)
            date: datetime a mostrar, si None usa fecha actual
            position: (x, y) posición, si None usa esquina inferior derecha

        Returns:
            Frame con fecha añadida
        """
        result = frame.copy()
        H, W = frame.shape[:2]

        if date is None:
            date = datetime.now()

        if position is None:
            position = (W - 220, H - 20)

        # Formato VCR típico
        date_str = date.strftime("%b %d %Y  %I:%M %p").upper()

        # Añadir texto con sombra para mejor legibilidad
        x, y = position

        # Sombra
        cv2.putText(
            result,
            date_str,
            (x + 1, y + 1),
            VHSTextOverlay.FONT,
            0.5,
            (0, 0, 0),
            VHSTextOverlay.FONT_THICKNESS
        )

        # Texto principal en amarillo/naranja
        cv2.putText(
            result,
            date_str,
            position,
            VHSTextOverlay.FONT,
            0.5,
            VHSTextOverlay.COLOR_CYAN,
            1
        )

        return result

    @staticmethod
    def add_channel_number(frame: np.ndarray,
                           channel: int = 3,
                           position: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Añade número de canal estilo VCR.

        Los VCRs mostraban el canal de entrada (típicamente CH 3 o CH 4).

        Args:
            frame: Array BGR (H, W, 3)
            channel: Número de canal a mostrar
            position: (x, y) posición, si None usa esquina superior izquierda

        Returns:
            Frame con número de canal
        """
        result = frame.copy()

        if position is None:
            position = (20, 80)

        channel_str = f"CH {channel}"

        cv2.putText(
            result,
            channel_str,
            position,
            VHSTextOverlay.FONT,
            VHSTextOverlay.FONT_SCALE,
            VHSTextOverlay.COLOR_CYAN,
            VHSTextOverlay.FONT_THICKNESS
        )

        return result

    @staticmethod
    def add_sp_lp_indicator(frame: np.ndarray,
                            mode: str = "SP",
                            position: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Añade indicador de velocidad de grabación (SP/LP/EP).

        - SP (Standard Play): 2 horas por cinta
        - LP (Long Play): 4 horas por cinta
        - EP/SLP (Extended/Super Long Play): 6 horas por cinta

        Args:
            frame: Array BGR (H, W, 3)
            mode: "SP", "LP", o "EP"
            position: (x, y) posición

        Returns:
            Frame con indicador de modo
        """
        result = frame.copy()
        H, W = frame.shape[:2]

        if position is None:
            position = (20, H - 20)

        cv2.putText(
            result,
            mode.upper(),
            position,
            VHSTextOverlay.FONT,
            VHSTextOverlay.FONT_SCALE,
            VHSTextOverlay.COLOR_WHITE,
            VHSTextOverlay.FONT_THICKNESS
        )

        return result

    @classmethod
    def add_full_vcr_overlay(cls, frame: np.ndarray,
                              time_seconds: float,
                              is_recording: bool = True,
                              frame_count: int = 0,
                              show_date: bool = True) -> np.ndarray:
        """
        Añade overlay VCR completo con todos los elementos.

        Incluye:
        - Timecode
        - Indicador REC o PLAY (parpadeante)
        - Fecha/hora
        - Indicador SP/LP

        Args:
            frame: Array BGR (H, W, 3)
            time_seconds: Tiempo actual
            is_recording: True para REC, False para PLAY
            frame_count: Contador de frames (para parpadeo)
            show_date: Mostrar fecha/hora

        Returns:
            Frame con overlay VCR completo
        """
        result = frame.copy()

        # Timecode
        result = cls.add_timecode(result, time_seconds)

        # REC o PLAY (parpadeo cada 15 frames)
        show_indicator = (frame_count % 30) < 15

        if is_recording:
            result = cls.add_rec_indicator(result, show_circle=show_indicator)
        else:
            if show_indicator:
                result = cls.add_play_indicator(result)

        # Fecha/hora
        if show_date:
            result = cls.add_date_stamp(result)

        # Indicador SP
        result = cls.add_sp_lp_indicator(result, "SP")

        return result
