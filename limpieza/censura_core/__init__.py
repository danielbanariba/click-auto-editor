"""Motor de censura — carga datos YAML y expone detección/redacción."""

from limpieza.censura_core.loader import CensuraData, load_all
from limpieza.censura_core.normalizer import normalize_compact, normalize_word
from limpieza.censura_core.engine import (
    CensuraEngine,
    get_default_engine,
)

__all__ = [
    "CensuraData",
    "CensuraEngine",
    "load_all",
    "normalize_word",
    "normalize_compact",
    "get_default_engine",
]
