#!/usr/bin/env python3
"""Canonicalizacion de nombres de pais a un codigo ISO y a un nombre en ingles.

El objetivo es que distintas variantes del mismo pais (ingles, espanol,
abreviaturas, gentilicios) colapsen a UN solo nombre estable, para no generar
playlists duplicadas en YouTube.

No se usa busqueda fuzzy: es peligrosa (p.ej. "UK" matchea "Uganda"). Solo
coincidencias exactas contra nombres en ingles/espanol (CLDR via babel),
codigos ISO de dos letras, y un mapa de alias manual.
"""
import unicodedata

try:
    import pycountry
except Exception:  # pragma: no cover - dependencia declarada en requirements
    pycountry = None

try:
    from babel import Locale
except Exception:  # pragma: no cover - dependencia declarada en requirements
    Locale = None


# Alias manuales: gentilicios, abreviaturas y variantes que CLDR no cubre.
_ALIAS = {
    "usa": "US",
    "u.s.a": "US",
    "u.s.a.": "US",
    "united states of america": "US",
    "eeuu": "US",
    "ee.uu": "US",
    "ee.uu.": "US",
    "uk": "GB",
    "u.k": "GB",
    "u.k.": "GB",
    "england": "GB",
    "inglaterra": "GB",
    "gran bretana": "GB",
    "great britain": "GB",
    "britain": "GB",
    "russian": "RU",
    "ruso": "RU",
    "rusa": "RU",
    "czech republic": "CZ",
    "republica checa": "CZ",
    "korea": "KR",
    "south korea": "KR",
    "corea del sur": "KR",
    "corea": "KR",
    "holland": "NL",
    "holanda": "NL",
}


def _norm(value):
    """Minusculas sin acentos ni espacios sobrantes, para comparar."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.casefold().split()).strip()


def _build_reverse_map():
    mapping = {}
    if Locale is None:
        return mapping
    for loc in ("en", "es"):
        try:
            territories = Locale.parse(loc).territories
        except Exception:  # pragma: no cover
            continue
        for code, name in territories.items():
            if len(code) == 2 and code.isalpha():
                mapping.setdefault(_norm(name), code.upper())
    return mapping


_REVERSE = _build_reverse_map()


def _valid_codes():
    if pycountry is None:
        return set()
    return {c.alpha_2 for c in pycountry.countries}


_CODES = _valid_codes()


def canonicalizar_pais(value):
    """Devuelve el codigo ISO alpha-2 del pais, o None si no se reconoce."""
    if not value:
        return None
    key = _norm(value)
    if not key:
        return None
    if key in _ALIAS:
        return _ALIAS[key]
    if len(key) == 2 and key.isalpha():
        code = key.upper()
        if not _CODES or code in _CODES:
            return code
    if key in _REVERSE:
        return _REVERSE[key]
    if pycountry is not None:
        try:
            return pycountry.countries.lookup(str(value)).alpha_2
        except LookupError:
            return None
    return None


def nombre_pais_en(value):
    """Devuelve el nombre canonico en ingles del pais.

    Si no se reconoce, conserva el valor original (limpiado) para no perderlo.
    """
    code = canonicalizar_pais(value)
    if not code:
        return value.strip() if isinstance(value, str) else value
    if Locale is not None:
        try:
            name = Locale.parse("en").territories.get(code)
            if name:
                return name
        except Exception:  # pragma: no cover
            pass
    if pycountry is not None:
        country = pycountry.countries.get(alpha_2=code)
        if country and getattr(country, "name", None):
            return country.name
    return value.strip() if isinstance(value, str) else value
