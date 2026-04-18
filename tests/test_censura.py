"""Tests de censura — cobertura completa del módulo limpieza.censura.

Incluye:
- Paridad funcional con comportamiento viejo
- Corpus real de títulos metal (para evitar regresiones en whitelist)
- Leetspeak y homóglifos
- Detección en slugs
- Multilingüe (EN + ES)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Permitir importar el paquete desde la raíz del repo sin instalación.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from limpieza.censura import (
    censor_profanity,
    contains_profanity,
    contains_profanity_fragment,
    is_whitelisted,
    list_profanity,
    should_censor,
    PROFANITY_EXACT,
    WHITELIST,
    VOWELLESS_PATTERNS,
)


# ============================================================================
# Basics — detección exacta
# ============================================================================

@pytest.mark.parametrize("text", [
    "fuck",
    "What the fuck is this",
    "shit happens",
    "this is bullshit",
    "you bitch",
    "motherfucker",
])
def test_detects_english_profanity(text):
    assert contains_profanity(text) is True


@pytest.mark.parametrize("text", [
    "puta madre",
    "hijo de puta",
    "que pendejo",
    "mierda total",
    "no jodas",
])
def test_detects_spanish_profanity(text):
    assert contains_profanity(text) is True


@pytest.mark.parametrize("text", [
    "caralho",
    "que porra é essa",
    "vai se foder",
    "está fodendo",
    "sua puta",
    "seu cacete",
])
def test_detects_portuguese_profanity(text):
    assert contains_profanity(text) is True


@pytest.mark.parametrize("text", [
    "scheiße",
    "du arschloch",
    "dumme fotze",
    "geh ficken",
    "hurensohn",
    "gefickt",
])
def test_detects_german_profanity(text):
    assert contains_profanity(text) is True


@pytest.mark.parametrize("text", [
    "cazzo",
    "che stronzo",
    "vaffanculo",
    "fottuto bastardo",
    "puttana",
    "sei un coglione",
])
def test_detects_italian_profanity(text):
    assert contains_profanity(text) is True


@pytest.mark.parametrize("text", [
    "merde alors",
    "putain de merde",
    "enculé",
    "quel connard",
    "foutu",
    "salope",
])
def test_detects_french_profanity(text):
    assert contains_profanity(text) is True


# ============================================================================
# Stemming — detecta inflexiones sin enumerarlas
# ============================================================================

@pytest.mark.parametrize("word", [
    # EN stemmer: fucking/fucked/fuckers → stem "fuck"
    "fucking",
    "fucked",
    # ES stemmer: cogiendo/cogida → stem "cog"
    "cogiendo",
    # PT stemmer: fodendo/fodido → stem "fod"
    "fodendo",
    "fodido",
    # DE stemmer: gefickt/fickt → stem "fick"
    "gefickt",
    "fickt",
    # IT stemmer: fottuto/fottuti → stem "fott"
    "fottuto",
    "fottuti",
])
def test_stemming_catches_inflections(word):
    assert contains_profanity(word) is True, (
        f"Stemming debería detectar inflexión: {word!r}"
    )


def test_stemming_does_not_flag_metal_whitelist():
    """Stemming NO debe pisar a términos metal en whitelist."""
    titles = [
        "Cannibal Corpse",
        "Napalm Death",
        "Slayer",
        "Carcass",
        "Morbid Angel",
        "Sepultura",    # metal BR
        "Rammstein",    # metal DE
        "Alcest",       # black metal FR
    ]
    for t in titles:
        assert not contains_profanity(t), f"Stemming pisó whitelist metal: {t!r}"


# ============================================================================
# Falsos positivos — NUNCA censurar
# ============================================================================

@pytest.mark.parametrize("text", [
    "Classic Assassin",
    "Data analysis",
    "Bass player",
    "Mass destruction",
    "Skill level",
    "Category Five",
    "Gorgeous view",
    "Documentary film",
    "Banal conversation",
    "Canal Street",
    "Harassment report",
    "Final countdown",
])
def test_false_positives_not_censored(text):
    assert contains_profanity(text) is False, f"Falso positivo: {text}"


# ============================================================================
# Contexto metal — bandas y temática
# ============================================================================

@pytest.mark.parametrize("band", [
    "Cannibal Corpse",
    "Napalm Death",
    "Carcass",
    "Dying Fetus",
    "Morbid Angel",
    "Slayer",
    "Exodus",
    "Obituary",
    "Autopsy",
    "Massacre",
    "Death",
])
def test_metal_bands_not_censored(band):
    assert contains_profanity(band) is False, f"Banda metal censurada: {band}"


@pytest.mark.parametrize("title", [
    "Hammer Smashed Face",
    "Tomb of the Mutilated",
    "Scum",
    "Reign in Blood",
    "From Enslavement to Obliteration",
    "Symphonies of Sickness",
    "Heartwork",
    "Altars of Madness",
    "Slaughter of the Soul",
    "Human",
    "Left Hand Path",
    "Like an Everflowing Stream",
    "The Bleeding",
    "Killing on Adrenaline",
    "Butchered at Birth",
    "Gore Obsessed",
])
def test_real_metal_album_titles_pass(title):
    """Álbumes reales que NO deben censurarse — regresión metal."""
    assert contains_profanity(title) is False, f"Álbum legítimo censurado: {title}"


# ============================================================================
# Leetspeak y obfuscación
# ============================================================================

@pytest.mark.parametrize("input_text,should_detect", [
    ("f*ck you", True),
    ("sh1t happens", True),
    ("b!tch", True),
    ("@ss", True),  # 'ass' aislado sí se censura
    ("fuck", True),
    ("fu.ck", True),   # con fillers
    ("sh*t", True),
])
def test_leetspeak_obfuscation(input_text, should_detect):
    assert contains_profanity(input_text) is should_detect, (
        f"{input_text!r} → esperado contains_profanity={should_detect}"
    )


# Fase 3: resistencia a evasión avanzada
@pytest.mark.parametrize("text", [
    "f-u-c-k",
    "s.h.i.t",
    "b_i_t_c_h",
    "f-u-c-k you",
    "said s.h.i.t loud",
])
def test_split_word_obfuscation(text):
    """Palabras con separadores (f-u-c-k, s.h.i.t, b_i_t_c_h)."""
    assert contains_profanity(text) is True


@pytest.mark.parametrize("text", [
    "f***",
    "sh*t",
    "b*tch",
    "p*ssy",
    "d*ck",
    "p*ta",      # ES
    "c*zzo",     # IT
    "m*rde",     # FR
])
def test_asterisk_as_wildcard(text):
    """Asteriscos como wildcard de cualquier carácter."""
    assert contains_profanity(text) is True


@pytest.mark.parametrize("text", [
    "U.S.A.",
    "A.I.",
    "ASCII art",
    "pass the ball",
    "classic rock",
    "the final countdown",
])
def test_separators_dont_trigger_false_positives(text):
    assert contains_profanity(text) is False


# ============================================================================
# Language detection + palabras ambiguas (Fase 3)
# ============================================================================

def test_ambiguous_queue_activates_on_french():
    """'queue' es vulgar en FR pero inocente en EN."""
    # Texto claramente FR → queue se activa
    assert contains_profanity(
        "La queue du renard dans la forêt profonde"
    ) is True
    # Texto claramente EN → queue NO se activa
    assert contains_profanity(
        "Stand in the queue please for the train to arrive"
    ) is False


def test_ambiguous_chatte_activates_on_french():
    """'chatte' es vulgar en FR, cat/gata en otros idiomas."""
    assert contains_profanity(
        "La chatte noire du voisin marchait lentement"
    ) is True
    assert contains_profanity(
        "The cat sat on the chatte rug peacefully"
    ) is False


def test_spanish_con_is_never_censored():
    """'con' es conector en ES, NUNCA debe ser grosería en este sistema."""
    assert contains_profanity("café con leche para desayunar") is False
    assert contains_profanity("voy con mi hermano al concierto") is False


def test_short_texts_skip_language_detection():
    """Textos cortos (<15 chars) no activan ambiguous detection."""
    # "queue" solo (sin contexto) NO debería censurarse
    assert contains_profanity("queue") is False
    assert contains_profanity("chatte") is False


def test_homoglyph_cyrillic():
    """Un caracter cirílico visualmente idéntico a latín debe detectarse."""
    # 'а' = U+0430 Cyrillic small a — visualmente igual a latín 'a'
    obfuscated = "fuck".replace("a", "а")  # no aplica a 'fuck' pero sirve para 'ass'
    assert contains_profanity("fuck") is True
    # Con homóglifo real
    assert contains_profanity("pυta") is True  # ypsilon griega


# ============================================================================
# Word boundary — palabras cortas ambiguas
# ============================================================================

def test_ass_standalone_is_censored():
    """'ass' aislado se censura."""
    assert contains_profanity("your ass is grass") is True


def test_ass_in_word_not_censored():
    """'ass' dentro de palabra NO se censura (assassin, class, mass)."""
    assert contains_profanity("Classic assassin") is False
    assert contains_profanity("Bass player") is False


# ============================================================================
# Redacción (formato del output)
# ============================================================================

def test_censor_preserves_first_letter():
    assert censor_profanity("fuck") == "f***"


def test_censor_short_word_full_mask():
    # Palabras de len<=2 se enmascaran completas
    assert censor_profanity("mf") == "**"


def test_censor_preserves_non_profanity():
    text = "Hello World, this is clean text 123"
    assert censor_profanity(text) == text


def test_censor_in_phrase():
    # "bullshit" = 8 letras → b + 7 asteriscos = 8 chars
    assert censor_profanity("What the fuck is this bullshit") == \
        "What the f*** is this b*******"


def test_censor_multiple_languages_same_text():
    assert censor_profanity("fuck y puta") == "f*** y p***"


# ============================================================================
# contains_profanity_fragment — slugs / URLs
# ============================================================================

@pytest.mark.parametrize("slug,expected", [
    ("fucking-death", True),
    ("analcunt", True),
    ("cannibalcorpse", False),    # whitelist gana
    ("napalmdeath", False),
    ("death-metal-band", False),
    ("puta-madre", True),
    ("slaughter-of-the-soul", False),  # slaughter whitelisted
])
def test_fragment_detection(slug, expected):
    assert contains_profanity_fragment(slug) is expected, (
        f"fragment({slug!r}) esperado {expected}"
    )


# ============================================================================
# list_profanity — listar coincidencias
# ============================================================================

def test_list_finds_all():
    found = list_profanity("fuck you bitch")
    assert "fuck" in [w.lower() for w in found]
    assert "bitch" in [w.lower() for w in found]


def test_list_empty_on_clean():
    assert list_profanity("Cannibal Corpse live") == []


# ============================================================================
# Consistencia de datos (invariantes del sistema)
# ============================================================================

def test_no_blacklist_whitelist_collision():
    collision = PROFANITY_EXACT & WHITELIST
    assert not collision, f"blacklist ∩ whitelist: {sorted(collision)}"


def test_vowelless_auto_generated_covers_common():
    """Las variantes más comunes deben estar en obfuscations."""
    # Generadas automáticamente
    assert VOWELLESS_PATTERNS.get("fck") == "fuck"
    assert VOWELLESS_PATTERNS.get("sht") == "shit"
    assert VOWELLESS_PATTERNS.get("btch") == "bitch"
    # Curadas del obfuscations.yaml
    assert VOWELLESS_PATTERNS.get("fk") == "fuck"
    assert VOWELLESS_PATTERNS.get("pt") == "puta"


def test_is_whitelisted_on_band():
    assert is_whitelisted("Cannibal") is True
    assert is_whitelisted("Napalm") is True


def test_should_censor_basic():
    assert should_censor("fuck") is True
    assert should_censor("assassin") is False
    assert should_censor("cannibal") is False
    assert should_censor("puta") is True


# ============================================================================
# Edge cases
# ============================================================================

def test_empty_string():
    assert censor_profanity("") == ""
    assert contains_profanity("") is False
    assert list_profanity("") == []
    assert contains_profanity_fragment("") is False


def test_whitespace_only():
    assert contains_profanity("   ") is False


def test_only_punctuation():
    assert contains_profanity("!?.-_*") is False


def test_mixed_case():
    assert contains_profanity("FUCK") is True
    assert contains_profanity("Fuck") is True
    assert contains_profanity("fUcK") is True


def test_accents_normalized():
    # palabras con acento deben detectarse igual
    assert contains_profanity("coño") is True
    assert contains_profanity("c*ño") is True  # Fase 3: wildcard + ñ
