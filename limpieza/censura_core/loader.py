"""Carga los datos de censura desde archivos YAML.

Los datos viven en limpieza/censura_data/:
- leet.yaml         — mapeo leetspeak + homóglifos + acentos + fillers
- whitelist.yaml    — falsos positivos y términos metal
- {lang}.yaml       — blacklists por idioma (en, es, pt, fr, it, de, ...)
- obfuscations.yaml — variantes curadas (f*ck → fck, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

DATA_DIR = Path(__file__).resolve().parent.parent / "censura_data"

# Idiomas que el loader carga por defecto. Agregar uno es tan simple como
# crear el archivo {lang}.yaml y sumarlo acá.
DEFAULT_LANGUAGES: tuple[str, ...] = ("en", "es", "pt", "fr", "it", "de")

# Map código ISO → nombre de algoritmo Snowball para stemming.
# Los códigos no listados acá no recibirán stemming.
SNOWBALL_ALGORITHMS: dict[str, str] = {
    "en": "english",
    "es": "spanish",
    "pt": "portuguese",
    "fr": "french",
    "it": "italian",
    "de": "german",
    "nl": "dutch",
    "ru": "russian",
    "sv": "swedish",
    "no": "norwegian",
    "da": "danish",
    "fi": "finnish",
    "ro": "romanian",
    "tr": "turkish",
    "hu": "hungarian",
}


@dataclass
class CensuraData:
    """Contenedor de datos cargados de los YAMLs."""

    blacklist: set[str] = field(default_factory=set)
    whitelist: set[str] = field(default_factory=set)
    word_boundary_only: set[str] = field(default_factory=set)
    obfuscations: dict[str, str] = field(default_factory=dict)
    leet_table: dict[str, str] = field(default_factory=dict)
    # Idiomas cargados y sus stems precomputados
    loaded_languages: tuple[str, ...] = ()
    # stems[lang] = {stem -> set of canonical words}
    stems_by_lang: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    # Palabras ambiguas cross-idioma — solo se activan cuando language
    # detection confirma el idioma correcto. Evita falsos positivos como
    # FR "queue" (polla) matcheando textos en inglés ("queue" = cola).
    ambiguous_by_lang: dict[str, set[str]] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Detecta inconsistencias — retorna lista de warnings."""
        warnings: list[str] = []

        collision = self.blacklist & self.whitelist
        if collision:
            warnings.append(
                f"Colisión blacklist ∩ whitelist: {sorted(collision)}"
            )

        boundary_in_exact = self.word_boundary_only & self.blacklist
        if boundary_in_exact:
            warnings.append(
                f"word_boundary_only redundante con blacklist: "
                f"{sorted(boundary_in_exact)}"
            )

        boundary_in_wl = self.word_boundary_only & self.whitelist
        if boundary_in_wl:
            warnings.append(
                f"word_boundary_only colisiona con whitelist: "
                f"{sorted(boundary_in_wl)}"
            )

        # Obfuscations deberían apuntar a palabras en blacklist
        orphan_obf = {
            variant: canon
            for variant, canon in self.obfuscations.items()
            if canon not in self.blacklist and canon not in self.word_boundary_only
        }
        if orphan_obf:
            warnings.append(
                f"obfuscations apuntan a palabras fuera de blacklist: {orphan_obf}"
            )

        return warnings


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _iter_blacklist_words(lang_data: dict) -> Iterable[str]:
    """Extrae palabras de las categorías severity=explicit de un idioma."""
    categories = lang_data.get("categories", {})
    for cat in categories.values():
        severity = cat.get("severity", "explicit")
        if severity != "explicit":
            # Reservado para futuro modo estricto — por ahora solo explicit.
            continue
        for word in cat.get("words", []):
            yield word


def _build_leet_table(leet_yaml: dict) -> dict[str, str]:
    """Combina leetspeak + homoglyphs + accents + fillers en un solo mapping."""
    table: dict[str, str] = {}
    table.update(leet_yaml.get("leetspeak", {}))
    table.update(leet_yaml.get("homoglyphs", {}))
    table.update(leet_yaml.get("accents", {}))
    for filler in leet_yaml.get("fillers", []):
        table[filler] = ""
    return table


def _auto_generate_vowelless(words: Iterable[str]) -> dict[str, str]:
    """Para cada palabra base, genera su variante sin vocales.

    Ejemplo: fuck → fck, shit → sht, dick → dck.
    Salta palabras muy cortas (<=3) para evitar falsos positivos.
    """
    vowels = set("aeiou")
    table: dict[str, str] = {}
    for word in words:
        if len(word) <= 3:
            continue
        stripped = "".join(ch for ch in word if ch.lower() not in vowels)
        if 2 <= len(stripped) < len(word):
            # No pisar variantes curadas del YAML (se mergean arriba)
            table.setdefault(stripped, word)
    return table


def _build_stems(words: Iterable[str], lang_code: str) -> dict[str, set[str]]:
    """Precomputa stems Snowball para todas las palabras de un idioma.

    Retorna dict: stem -> set de palabras canonical que lo generan.
    Si snowballstemmer no está disponible o el idioma no tiene algoritmo,
    retorna dict vacío (el engine hará fallback silencioso).
    """
    algo = SNOWBALL_ALGORITHMS.get(lang_code)
    if not algo:
        return {}
    try:
        import snowballstemmer
    except ImportError:
        return {}

    stemmer = snowballstemmer.stemmer(algo)
    result: dict[str, set[str]] = {}
    for word in words:
        stem = stemmer.stemWord(word)
        # Solo guardamos stems que reducen efectivamente la palabra
        # (o que son idénticos a la palabra — la palabra ya era stem).
        if not stem or len(stem) < 3:
            # Stems muy cortos (<3) causan demasiados falsos positivos
            # cross-lingual. No los usamos.
            continue
        result.setdefault(stem, set()).add(word)
    return result


def load_all(
    data_dir: Path | None = None,
    languages: Iterable[str] = DEFAULT_LANGUAGES,
    raise_on_warning: bool = False,
) -> CensuraData:
    """Carga todos los YAMLs y devuelve un CensuraData listo para el engine."""
    base = Path(data_dir) if data_dir else DATA_DIR

    data = CensuraData()

    # --- leet.yaml ---
    leet_yaml = _load_yaml(base / "leet.yaml")
    data.leet_table = _build_leet_table(leet_yaml)

    # Normalizador local: baja a minúsculas y aplica el leet_table.
    # Esto asegura que las palabras en blacklist/whitelist estén guardadas
    # en su forma YA normalizada, para que coincidan con input normalizado.
    def _norm(word: str) -> str:
        return word.lower().translate({ord(k): v for k, v in data.leet_table.items()})

    # --- whitelist.yaml ---
    wl_yaml = _load_yaml(base / "whitelist.yaml")
    for word in wl_yaml.get("false_positives", []):
        data.whitelist.add(_norm(word))
    for word in wl_yaml.get("metal_bands_and_themes", []):
        data.whitelist.add(_norm(word))

    # --- idiomas ---
    # Cargamos cada idioma y trackeamos qué palabras le pertenecen
    # para precomputar stems por idioma.
    words_by_lang: dict[str, set[str]] = {}
    loaded: list[str] = []
    for lang in languages:
        path = base / f"{lang}.yaml"
        if not path.exists():
            continue
        lang_data = _load_yaml(path)
        lang_words: set[str] = set()
        for word in _iter_blacklist_words(lang_data):
            normalized = _norm(word)
            data.blacklist.add(normalized)
            lang_words.add(normalized)
        for word in lang_data.get("word_boundary_only", []):
            data.word_boundary_only.add(_norm(word))

        # Palabras ambiguas (cross-lingual). Solo se activan cuando
        # language detection confirma este idioma.
        ambiguous: set[str] = set()
        for cat in lang_data.get("ambiguous_categories", {}).values():
            for word in cat.get("words", []):
                ambiguous.add(_norm(word))
        if ambiguous:
            data.ambiguous_by_lang[lang] = ambiguous

        words_by_lang[lang] = lang_words
        loaded.append(lang)

    data.loaded_languages = tuple(loaded)

    # --- Precomputar stems por idioma ---
    for lang, words in words_by_lang.items():
        # Excluir whitelist antes de stemmear: evita stems que coincidan
        # con falsos positivos o términos metal.
        stemmable = words - data.whitelist
        data.stems_by_lang[lang] = _build_stems(stemmable, lang)

    # --- obfuscations curadas ---
    obf_yaml = _load_yaml(base / "obfuscations.yaml")
    curated = {_norm(k): _norm(v) for k, v in obf_yaml.get("obfuscations", {}).items()}

    # --- auto-generar versiones sin vocales ---
    auto = _auto_generate_vowelless(data.blacklist)

    # Las curadas tienen prioridad sobre las auto-generadas.
    merged: dict[str, str] = {}
    merged.update(auto)
    merged.update(curated)
    data.obfuscations = merged

    warnings = data.validate()
    if warnings and raise_on_warning:
        raise ValueError("Datos de censura inconsistentes: " + "; ".join(warnings))

    return data
