"""
Detector y redactor de menciones a organizaciones extremistas o criminales.

Este módulo es COMPLEMENTARIO a limpieza/censura.py:
- censura.py    → groserías individuales (fuck, shit) con leetspeak/stemming
- extremismo.py → frases multi-palabra de organizaciones designadas, alias
                  de terroristas y grupos criminales violentos

Política de YouTube de referencia:
    "Organizaciones criminales o extremistas violentas — elogios o promociones"

Datos en limpieza/extremismo_data/*.yaml:
    terrorist_orgs.yaml      FTO designadas internacionalmente
    domestic_terror_us.yaml  Unabomber, McVeigh, grupos extrema derecha US
    cartels.yaml             Cárteles + líderes mediáticos
    mass_shooters.yaml       Columbine, Sandy Hook, atacantes notorios
    whitelist.yaml           Frases legítimas que NO se redactan

API:
    redact_extremism(text)   → texto con menciones censuradas con asteriscos
    contains_extremism(text) → bool
    list_extremism(text)     → lista de menciones detectadas (texto original)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).resolve().parent / "extremismo_data"

CATEGORY_FILES: tuple[str, ...] = (
    "terrorist_orgs.yaml",
    "terrorist_orgs_extended.yaml",
    "domestic_terror_us.yaml",
    "cartels.yaml",
    "mass_shooters.yaml",
    "mass_shooters_extended.yaml",
    "historical_atrocities.yaml",
    "nsbm_bands.yaml",
)

WHITELIST_FILE = "whitelist.yaml"
WHITELIST_SECTIONS: tuple[str, ...] = (
    "bands",
    "academic_context",
    "metal_themes",
    "geography",
    "personal_names_legitimate",
    "historical_locations",
)

# Variantes acentuadas/diacríticas por letra base. El regex compilado para
# cada término sustituye cada letra base por su character class, permitiendo
# matchear "Cartel" tanto en "cartel" como en "cártel" sin tocar el texto.
_ACCENT_CLASSES: dict[str, str] = {
    "a": "[aáàäâãåā]",
    "e": "[eéèëêēė]",
    "i": "[iíìïîī]",
    "o": "[oóòöôõøō]",
    "u": "[uúùüûū]",
    "n": "[nñ]",
    "c": "[cç]",
    "s": "[sß]",
    "y": "[yý]",
}

# Caracteres que se consideran "letra" para los boundaries del match.
# Incluye latín extendido para que "Kaczynski" termine antes de "í" si toca.
_WORD_CHAR_CLASS = r"[A-Za-z0-9À-ÖØ-öø-ÿĀ-ſ]"


@dataclass
class ExtremismoData:
    """Datos cargados de los YAMLs."""

    canonical_to_aliases: dict[str, set[str]] = field(default_factory=dict)
    term_to_category: dict[str, str] = field(default_factory=dict)
    all_terms_ordered: list[str] = field(default_factory=list)
    whitelist_phrases: list[str] = field(default_factory=list)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _char_class(c: str) -> str:
    """Devuelve el character class para un carácter — soporta acentos."""
    lower = c.lower()
    if lower in _ACCENT_CLASSES:
        return _ACCENT_CLASSES[lower]
    return re.escape(c)


def _build_pattern_for_phrase(phrase: str) -> re.Pattern | None:
    """Compila un regex para una frase multi-palabra.

    - Case-insensitive
    - Espacios entre palabras flexibles (espacio, tab, NBSP)
    - Acentos opcionales en vocales y "ñ"
    - Guion/apostrofe/underscore tratados como espacio
    - Boundaries de palabra en los extremos para no matchear dentro de
      palabras más largas
    """
    cleaned = phrase.strip()
    if not cleaned:
        return None

    # Tokenizar — separadores incluyen guion, apostrofe, underscore
    tokens = re.split(r"[\s\-'`´_]+", cleaned)
    tokens = [t for t in tokens if t]
    if not tokens:
        return None

    escaped_tokens: list[str] = []
    for token in tokens:
        escaped_tokens.append("".join(_char_class(c) for c in token))

    # Separadores tolerantes entre tokens: espacios, guion, apóstrofe, underscore
    body = r"[\s\-'`´_]+".join(escaped_tokens)

    pattern = (
        rf"(?<!{_WORD_CHAR_CLASS})"
        rf"({body})"
        rf"(?!{_WORD_CHAR_CLASS})"
    )
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


def load_data(data_dir: Path | None = None) -> ExtremismoData:
    """Carga todos los YAMLs y arma la estructura de datos."""
    base = Path(data_dir) if data_dir else DATA_DIR
    data = ExtremismoData()

    collected: list[tuple[str, str]] = []  # (term, category_key)

    for filename in CATEGORY_FILES:
        yaml_data = _load_yaml(base / filename)
        categories = yaml_data.get("categories", {})
        for cat_key, cat_data in categories.items():
            if not isinstance(cat_data, dict):
                continue
            for entry in cat_data.get("entries", []) or []:
                canonical = (entry.get("canonical") or "").strip()
                if not canonical:
                    continue
                aliases = [
                    a.strip()
                    for a in (entry.get("aliases") or [])
                    if a and a.strip()
                ]
                term_set = {canonical, *aliases}
                data.canonical_to_aliases[canonical] = term_set
                for term in term_set:
                    collected.append((term, cat_key))
                    data.term_to_category[term.casefold()] = cat_key

    # Orden: matches largos ganan a cortos cuando solapan
    collected.sort(key=lambda pair: -len(pair[0]))
    data.all_terms_ordered = [term for term, _ in collected]

    # Whitelist
    wl_yaml = _load_yaml(base / WHITELIST_FILE)
    for section in WHITELIST_SECTIONS:
        for phrase in wl_yaml.get(section, []) or []:
            if isinstance(phrase, str) and phrase.strip():
                data.whitelist_phrases.append(phrase.strip())

    return data


class ExtremismoEngine:
    """Motor de detección y redacción multi-palabra."""

    def __init__(self, data: ExtremismoData):
        self.data = data

        # Pre-compilar regex para cada término. Mantener el orden por longitud
        # descendente — el resolver de solapamientos se apoya en esto.
        self._term_patterns: list[tuple[str, re.Pattern]] = []
        for term in data.all_terms_ordered:
            pat = _build_pattern_for_phrase(term)
            if pat is not None:
                self._term_patterns.append((term, pat))

        # Whitelist
        self._whitelist_patterns: list[re.Pattern] = []
        for phrase in data.whitelist_phrases:
            pat = _build_pattern_for_phrase(phrase)
            if pat is not None:
                self._whitelist_patterns.append(pat)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------
    def _whitelist_spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for pat in self._whitelist_patterns:
            for m in pat.finditer(text):
                spans.append((m.start(), m.end()))
        return spans

    @staticmethod
    def _span_inside_any(
        start: int, end: int, spans: list[tuple[int, int]]
    ) -> bool:
        for s, e in spans:
            if start >= s and end <= e:
                return True
        return False

    def _raw_matches(self, text: str) -> list[tuple[int, int, str, str]]:
        """Devuelve (start, end, matched_text, term_canonical) sin resolver
        solapamientos. Filtra los matches cubiertos por whitelist."""
        wl_spans = self._whitelist_spans(text)
        out: list[tuple[int, int, str, str]] = []
        for term, pattern in self._term_patterns:
            for m in pattern.finditer(text):
                if self._span_inside_any(m.start(), m.end(), wl_spans):
                    continue
                out.append((m.start(), m.end(), m.group(0), term))
        return out

    def _resolve_matches(
        self, text: str
    ) -> list[tuple[int, int, str, str]]:
        """Resuelve solapamientos quedándose con el match más largo en cada
        rango. Si dos matches del mismo largo solapan, gana el primero."""
        raw = self._raw_matches(text)
        # Orden: por start ASC, longitud DESC (para que el primero en cada
        # posición sea el más largo)
        raw.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        resolved: list[tuple[int, int, str, str]] = []
        last_end = -1
        for start, end, matched, term in raw:
            if start >= last_end:
                resolved.append((start, end, matched, term))
                last_end = end
            elif end > last_end:
                # Solapa pero termina más tarde — descartamos para no
                # generar redacciones parciales con texto cortado raro
                continue
        return resolved

    @staticmethod
    def _censor_phrase(phrase: str) -> str:
        """Mantiene la primera letra de cada palabra; resto a asteriscos.

        Ejemplos:
            "Freedom Club"      → "F****** C***"
            "Cártel de Sinaloa" → "C***** d* S******"
            "Unabomber"         → "U********"
            "ETA"               → "E**"
            "FC"                → "**"
        """
        result: list[str] = []
        # Conservamos los separadores tal cual al volver a unir
        tokens = re.split(r"(\s+)", phrase)
        for token in tokens:
            if not token or token.isspace():
                result.append(token)
                continue
            if len(token) <= 2:
                result.append("*" * len(token))
            else:
                result.append(token[0] + "*" * (len(token) - 1))
        return "".join(result)

    # ------------------------------------------------------------------
    # API interna
    # ------------------------------------------------------------------
    def redact(self, text: str) -> str:
        if not text:
            return text
        matches = self._resolve_matches(text)
        if not matches:
            return text
        # Aplicamos de derecha a izquierda para preservar índices intactos
        result = text
        for start, end, matched, _ in sorted(matches, key=lambda x: -x[0]):
            result = result[:start] + self._censor_phrase(matched) + result[end:]
        return result

    def contains(self, text: str) -> bool:
        if not text:
            return False
        return bool(self._resolve_matches(text))

    def list_terms(self, text: str) -> list[str]:
        if not text:
            return []
        return [matched for _, _, matched, _ in self._resolve_matches(text)]

    def list_detail(self, text: str) -> list[dict]:
        """Variante diagnóstica: cada match con su categoría y término canónico.

        Útil para reportes — no se usa en el pipeline pero ayuda a depurar
        falsos positivos.
        """
        if not text:
            return []
        out: list[dict] = []
        for start, end, matched, term in self._resolve_matches(text):
            out.append(
                {
                    "match": matched,
                    "term": term,
                    "category": self.data.term_to_category.get(
                        term.casefold(), "unknown"
                    ),
                    "start": start,
                    "end": end,
                }
            )
        return out


# ----------------------------------------------------------------------
# Singleton default
# ----------------------------------------------------------------------
_DEFAULT_ENGINE: ExtremismoEngine | None = None


def get_default_engine() -> ExtremismoEngine:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = ExtremismoEngine(load_data())
    return _DEFAULT_ENGINE


def reset_default_engine() -> None:
    """Fuerza recarga de datos — útil en tests."""
    global _DEFAULT_ENGINE
    _DEFAULT_ENGINE = None


# ============================================================================
# API pública
# ============================================================================
def redact_extremism(text: str) -> str:
    """Redacta menciones de organizaciones/individuos extremistas en `text`.

    Las frases detectadas se censuran preservando la primera letra de cada
    palabra y reemplazando el resto por asteriscos, en simetría con
    `limpieza.censura.censor_word`.
    """
    return get_default_engine().redact(text)


def contains_extremism(text: str) -> bool:
    """True si `text` contiene al menos una mención de extremismo detectable."""
    return get_default_engine().contains(text)


def list_extremism(text: str) -> list[str]:
    """Lista de spans detectados en `text` tal como aparecen (sin censurar)."""
    return get_default_engine().list_terms(text)


def list_extremism_detail(text: str) -> list[dict]:
    """Variante diagnóstica con categoría y término canónico por match."""
    return get_default_engine().list_detail(text)


__all__ = [
    "redact_extremism",
    "contains_extremism",
    "list_extremism",
    "list_extremism_detail",
    "ExtremismoEngine",
    "ExtremismoData",
    "load_data",
    "get_default_engine",
    "reset_default_engine",
]
