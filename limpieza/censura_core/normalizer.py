"""Normalización de texto: leetspeak, homóglifos, acentos y fillers."""

from __future__ import annotations

import re
from functools import lru_cache


def build_translator(leet_table: dict[str, str]) -> dict[int, str]:
    """Construye un mapa de traducción apto para str.translate()."""
    return {ord(src): dst for src, dst in leet_table.items()}


def normalize_word(word: str, translator: dict[int, str]) -> str:
    """Baja a minúsculas y aplica traducción (leet + acentos + fillers)."""
    return word.lower().translate(translator)


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_compact(text: str, translator: dict[int, str]) -> str:
    """Normaliza y colapsa todo non-alfanumérico.

    Útil para chequear slugs tipo 'fucking-death' o 'analcunt'.
    """
    return _NON_ALNUM.sub("", normalize_word(text, translator))


# Cache a nivel módulo para no reconstruir lru_cache en cada engine.
@lru_cache(maxsize=4096)
def _cached_normalize(word: str, translator_id: int, translator_tuple: tuple) -> str:
    translator = dict(translator_tuple)
    return word.lower().translate(translator)
