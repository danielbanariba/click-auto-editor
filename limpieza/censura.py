"""
Módulo de censura de texto para YouTube — API pública compatible.

Este módulo es un wrapper delgado sobre limpieza.censura_core, que
implementa la detección y redacción de groserías. Los datos (listas de
palabras por idioma, whitelist, leet mapping, ofuscaciones) viven en
YAMLs dentro de limpieza/censura_data/ — editables sin tocar código.

Soporta:
- Multilingüe (inglés y español por defecto; se puede extender vía YAML)
- Leetspeak (f*ck, sh1t, @ss, etc.)
- Acentos latinos (á → a, ñ → n, etc.)
- Homóglifos Unicode (griego/cirílico que se parecen a letras latinas)
- Whitelist con contexto metal (Cannibal Corpse, Napalm Death, etc.)
- Detección en slugs/URLs compactas (contains_profanity_fragment)

API:
    normalize_word, normalize_compact_text, is_whitelisted,
    should_censor, censor_word, censor_profanity, contains_profanity,
    contains_profanity_fragment, list_profanity,
    add_profanity_word, add_whitelist_word

Constantes expuestas (read-only; la fuente es YAML):
    PROFANITY_EXACT, PROFANITY_WORD_BOUNDARY, WHITELIST,
    VOWELLESS_PATTERNS, LEET_MAP
"""

from __future__ import annotations

from limpieza.censura_core import get_default_engine
from limpieza.censura_core.engine import _WORD_PATTERN

# ============================================================================
# Engine singleton — construido perezosamente al primer uso
# ============================================================================
_engine = get_default_engine()


# ============================================================================
# Constantes expuestas para compatibilidad con código viejo
# ============================================================================
PROFANITY_EXACT: set[str] = _engine.data.blacklist
PROFANITY_WORD_BOUNDARY: set[str] = _engine.data.word_boundary_only
WHITELIST: set[str] = _engine.data.whitelist
VOWELLESS_PATTERNS: dict[str, str] = _engine.data.obfuscations
LEET_MAP = str.maketrans(_engine.data.leet_table)


# ============================================================================
# API pública
# ============================================================================


def normalize_word(word: str) -> str:
    """Normaliza una palabra eliminando leetspeak y acentos."""
    return _engine.normalize(word)


def normalize_compact_text(text: str) -> str:
    """Normaliza un texto y elimina separadores para revisar slugs/URLs."""
    return _engine.normalize_compact(text)


def is_whitelisted(word: str) -> bool:
    """Verifica si una palabra está en la lista blanca."""
    return _engine.is_whitelisted(word)


def should_censor(word: str) -> bool:
    """Determina si una palabra debe ser censurada."""
    return _engine.should_censor(word)


def censor_word(word: str) -> str:
    """Censura una palabra manteniendo la primera letra visible."""
    return _engine.censor_word(word)


def censor_profanity(text: str) -> str:
    """Censura palabras ofensivas en un texto."""
    return _engine.censor_text(text)


def contains_profanity(text: str) -> bool:
    """Verifica si un texto contiene palabras ofensivas."""
    return _engine.contains_profanity(text)


def contains_profanity_fragment(text: str) -> bool:
    """Detecta groserías dentro de slugs o URLs compactas."""
    return _engine.contains_profanity_fragment(text)


def list_profanity(text: str) -> list[str]:
    """Lista todas las palabras ofensivas encontradas en un texto."""
    return _engine.list_profanity(text)


def add_profanity_word(word: str, require_boundary: bool = False) -> None:
    """Añade una palabra a la lista de censura en tiempo de ejecución."""
    _engine.add_profanity(word, require_boundary=require_boundary)


def add_whitelist_word(word: str) -> None:
    """Añade una palabra a la lista blanca en tiempo de ejecución."""
    _engine.add_whitelist(word)


__all__ = [
    "PROFANITY_EXACT",
    "PROFANITY_WORD_BOUNDARY",
    "WHITELIST",
    "VOWELLESS_PATTERNS",
    "LEET_MAP",
    "normalize_word",
    "normalize_compact_text",
    "is_whitelisted",
    "should_censor",
    "censor_word",
    "censor_profanity",
    "contains_profanity",
    "contains_profanity_fragment",
    "list_profanity",
    "add_profanity_word",
    "add_whitelist_word",
]
