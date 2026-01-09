"""
VHS Effect - Sistema completo de efectos VHS en Python

Este paquete implementa artefactos visuales y movimientos del formato
analógico de video VHS de los años 80-90, basado en la física real del formato.

Uso básico (CPU):
    from vhs_effect import VHSEffect
    vhs = VHSEffect(intensity=0.7)
    processed = vhs.process_frame(frame, frame_time=0.0)

Uso GPU (CuPy acelerado):
    from vhs_effect import VHSEffectGPU
    vhs = VHSEffectGPU(intensity=0.7)
    processed = vhs.process_frame(frame, frame_time=0.0)

Requiere para GPU:
    pip install cupy-cuda12x  # o cupy-cuda11x
"""

from .vhs_processor import VHSEffect
from .displacement import VHSDisplacementMap
from .overlays import VHSTextOverlay
from .color_utils import ColorProcessor

# GPU-accelerated versions (con fallback a CPU)
try:
    from .vhs_processor_gpu import VHSEffectGPU, process_video_gpu
    from .color_utils_gpu import ColorProcessorGPU, HAS_CUPY
    GPU_AVAILABLE = HAS_CUPY
except ImportError:
    VHSEffectGPU = VHSEffect  # Fallback
    GPU_AVAILABLE = False

__version__ = "1.1.0"
__author__ = "VHS Effect System"
__all__ = [
    "VHSEffect",
    "VHSEffectGPU",
    "VHSDisplacementMap",
    "VHSTextOverlay",
    "ColorProcessor",
    "GPU_AVAILABLE"
]
