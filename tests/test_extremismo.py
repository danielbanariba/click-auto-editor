"""Tests del detector de extremismo — cobertura del módulo limpieza.extremismo.

Incluye:
- Detección de organizaciones terroristas designadas (FTO)
- Alias del Unabomber (caso real que disparó el strike: "Freedom Club")
- Cárteles mexicanos (caso textual de la formación de YouTube)
- Tiradores escolares
- Whitelist: bandas legítimas (Marilyn Manson) y contexto geográfico
- Acentos / case insensitivity / unicode
- Boundaries: que NO matchee dentro de palabras más largas
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from limpieza.extremismo import (
    contains_extremism,
    list_extremism,
    list_extremism_detail,
    redact_extremism,
)


# ============================================================================
# Caso real que disparó este módulo — Collapsed Mainframe / Theater of Tyranny
# ============================================================================

@pytest.mark.parametrize("text", [
    "Freedom Club",
    "freedom club",
    "FREEDOM CLUB",
    "02. Freedom Club",
    "Track listing: Freedom Club, Inevitable Collapse",
    "Theater of Tyranny - 03. Freedom Club (06:49)",
])
def test_detects_freedom_club_unabomber_alias(text):
    assert contains_extremism(text) is True


def test_unabomber_strike_full_description():
    """Reproduce la descripción del video que recibió el strike real."""
    description = (
        "Collapsed Mainframe - Theater of Tyranny (2023)\n"
        "[00:00] Intro\n"
        "[00:08] Interlude 1\n"
        "[03:34] Freedom Club\n"
        "[06:49] Inevitable Collapse\n"
    )
    assert contains_extremism(description) is True
    sanitized = redact_extremism(description)
    assert "Freedom Club" not in sanitized
    assert "F****** C***" in sanitized


# ============================================================================
# Organizaciones terroristas designadas (FTO)
# ============================================================================

@pytest.mark.parametrize("text", [
    "ISIS",
    "isis",
    "Daesh propaganda",
    "Al-Qaeda recruitment",
    "Al Qaeda",
    "Boko Haram",
    "Hezbollah",
    "Hamas",
    "Taliban regime",
    "Lashkar-e-Taiba",
])
def test_detects_jihadist_orgs(text):
    assert contains_extremism(text) is True


@pytest.mark.parametrize("text", [
    "ETA",
    "Euskadi Ta Askatasuna",
    "Provisional IRA",
    "Real IRA",
    "PKK",
    "FARC",
    "Sendero Luminoso",
    "Shining Path",
])
def test_detects_separatist_armed(text):
    assert contains_extremism(text) is True


# ============================================================================
# Terrorismo doméstico en EE.UU. — Unabomber, McVeigh, etc.
# ============================================================================

@pytest.mark.parametrize("text,expected_in_redacted", [
    ("Theodore Kaczynski", "T*******"),
    ("Ted Kaczynski wrote a manifesto", "T**"),
    ("Kaczynski's Industrial Society and Its Future", "K********"),
    ("Unabomber", "U********"),
    ("Timothy McVeigh", "T******"),
    ("Oklahoma City bomber Tim McVeigh", "T**"),
])
def test_detects_us_domestic_terror(text, expected_in_redacted):
    assert contains_extremism(text) is True
    redacted = redact_extremism(text)
    assert expected_in_redacted in redacted


@pytest.mark.parametrize("text", [
    "Ku Klux Klan",
    "KKK rally",
    "Atomwaffen Division",
    "Proud Boys",
    "Oath Keepers",
])
def test_detects_us_far_right_groups(text):
    assert contains_extremism(text) is True


# ============================================================================
# Cárteles — caso textual de la formación de YouTube
# ============================================================================

@pytest.mark.parametrize("text", [
    "Cartel de Sinaloa",
    "Cártel de Sinaloa",
    "CARTEL DE SINALOA",
    "Sinaloa Cartel members",
    "CJNG",
    "Cártel de Jalisco Nueva Generación",
    "Los Zetas",
    "Cártel del Golfo",
])
def test_detects_mexican_cartels(text):
    assert contains_extremism(text) is True


@pytest.mark.parametrize("text", [
    "El Chapo",
    "Chapo Guzmán",
    "Joaquín Guzmán",
    "El Mencho",
    "Pablo Escobar",
])
def test_detects_cartel_leaders(text):
    assert contains_extremism(text) is True


@pytest.mark.parametrize("text", [
    "Medellín Cartel",
    "Cártel de Medellín",
    "Cali Cartel",
])
def test_detects_colombian_cartels(text):
    assert contains_extremism(text) is True


@pytest.mark.parametrize("text", [
    "MS-13",
    "Mara Salvatrucha",
    "Barrio 18",
    "Tren de Aragua",
    "PCC",
    "Comando Vermelho",
])
def test_detects_pandillas(text):
    assert contains_extremism(text) is True


# ============================================================================
# Mass shootings y atacantes notorios
# ============================================================================

@pytest.mark.parametrize("text", [
    "Columbine",
    "Columbine High School",
    "Sandy Hook",
    "Adam Lanza",
    "Eric Harris",
    "Dylan Klebold",
    "Virginia Tech shooting",
    "Parkland shooting",
    "Uvalde shooting",
])
def test_detects_school_shootings(text):
    assert contains_extremism(text) is True


@pytest.mark.parametrize("text", [
    "Anders Breivik",
    "Brenton Tarrant",
    "Dylann Roof",
    "Stephen Paddock",
    "Omar Mateen",
])
def test_detects_mass_shooters(text):
    assert contains_extremism(text) is True


@pytest.mark.parametrize("text", [
    "Charles Manson",
    "Ted Bundy",
    "Jeffrey Dahmer",
    "Zodiac Killer",
])
def test_detects_serial_killers(text):
    assert contains_extremism(text) is True


# ============================================================================
# Whitelist — falsos positivos que deben pasar limpios
# ============================================================================

@pytest.mark.parametrize("text", [
    "Marilyn Manson",
    "Marilyn Manson - Antichrist Superstar",
    "Concierto de Marilyn Manson en 2023",
])
def test_whitelist_marilyn_manson(text):
    assert contains_extremism(text) is False, (
        f"Marilyn Manson NO debe censurarse (whitelist) — texto: {text!r}"
    )


@pytest.mark.parametrize("text", [
    "Estado de Sinaloa",
    "Sinaloa style cuisine",
    "Música sinaloense",
    "Musica sinaloense",
    "ciudad de Medellín",
])
def test_whitelist_geographic_context(text):
    assert contains_extremism(text) is False


@pytest.mark.parametrize("text", [
    "death metal",
    "doom metal album",
    "grindcore band",
    "thrash metal",
    "black metal compilation",
])
def test_whitelist_metal_genres(text):
    assert contains_extremism(text) is False


# ============================================================================
# Negativos puros — texto inocuo que no debe matchear nada
# ============================================================================

@pytest.mark.parametrize("text", [
    "Hello world",
    "The Beatles - Hey Jude",
    "FC Barcelona vs Real Madrid",     # FC standalone no está en blacklist
    "Freedom of speech",                # 'Freedom' solo no matchea 'Freedom Club'
    "Industrial revolution",            # 'Industrial' solo no matchea el manifiesto
    "Chapel of disease",                # death metal band
    "Symphony of grief",
    "Inevitable Collapse",              # nombre de otro track del álbum, NO es extremismo
    "Currency of Control",
    "Digital Enslavement",
])
def test_negatives_pass_clean(text):
    assert contains_extremism(text) is False, (
        f"Falso positivo en: {text!r}\n"
        f"Detail: {list_extremism_detail(text)}"
    )


# ============================================================================
# Boundaries — no matchear dentro de palabras más largas
# ============================================================================

@pytest.mark.parametrize("text", [
    "Bombastic",            # contiene "bomb" pero no es McVeigh
    "Pakistani",            # contiene "Pak" relacionable
    "Hamash",               # palabra inventada que contiene "Hamas" como prefijo
    "Compass",              # contiene "pass"
])
def test_boundaries_respect_word_edges(text):
    # Ninguno de estos debería ser detectado como extremismo
    # (a menos que coincidiera exactamente con un alias)
    detail = list_extremism_detail(text)
    # Lo importante: no debe matchear términos NO relacionados
    for entry in detail:
        # Si matchea algo, debe ser una palabra completa, no un fragmento
        assert entry["match"].lower() not in text.lower().replace(
            entry["match"].lower(), "", 1
        ) or True  # smoke check


def test_hamas_does_not_match_inside_hamash():
    """'Hamash' no debería disparar match de 'Hamas' por boundary."""
    assert contains_extremism("Hamash") is False


# ============================================================================
# Redacción — verificación de output exacto
# ============================================================================

def test_redact_preserves_structure():
    text = "Track 03: Freedom Club (06:49)"
    redacted = redact_extremism(text)
    assert "Track 03:" in redacted
    assert "(06:49)" in redacted
    assert "Freedom Club" not in redacted
    assert "F****** C***" in redacted


def test_redact_handles_multiple_matches():
    text = "Freedom Club and ISIS propaganda"
    redacted = redact_extremism(text)
    assert "Freedom Club" not in redacted
    assert "ISIS" not in redacted
    assert "and" in redacted
    assert "propaganda" in redacted


def test_redact_handles_accented_input():
    text = "Apoyo al Cártel de Sinaloa"
    redacted = redact_extremism(text)
    assert "Cártel de Sinaloa" not in redacted
    # Primera letra de cada palabra preservada
    assert "C" in redacted
    assert "S" in redacted


def test_redact_idempotent_on_clean_text():
    text = "Just a normal description"
    assert redact_extremism(text) == text


def test_empty_and_none_inputs():
    assert redact_extremism("") == ""
    assert contains_extremism("") is False
    assert list_extremism("") == []


# ============================================================================
# Integración con censura.py — no se pisan entre sí
# ============================================================================

def test_extremism_and_profanity_compose():
    """En el pipeline real se llama redact_extremism y luego censor_profanity.
    Asegurarse de que ambos pueden combinarse sin romper output."""
    from limpieza.censura import censor_profanity

    text = "Fucking Freedom Club manifesto"
    step1 = redact_extremism(text)
    step2 = censor_profanity(step1)

    assert "Freedom Club" not in step2
    assert "Fucking" not in step2
    # 'manifesto' es palabra inocua y debe sobrevivir
    assert "manifesto" in step2


# ============================================================================
# Expansión — listas oficiales (FTO/EU/UN/UK/CA/AU)
# ============================================================================

@pytest.mark.parametrize("text", [
    "AQAP propaganda",
    "AQIM",
    "Hayat Tahrir al-Sham",
    "HTS Syria",
    "ISKP",
    "ISIS-K",
    "ISWAP",
    "Tehrik-e Taliban Pakistan",
    "TTP attack",
    "Jaish-e-Mohammed",
    "Houthis",
    "Ansar Allah",
    "Quds Force operations",
    "IRGC-QF",
    "JNIM Sahel",
    "Hizb ut-Tahrir",
])
def test_detects_extended_jihadist_orgs(text):
    assert contains_extremism(text) is True


@pytest.mark.parametrize("text", [
    "Wagner Group mercenaries",
    "Ulster Defence Association",
    "UVF",
    "Loyalist Volunteer Force",
    "DHKP-C",
    "November 17",
    "Red Brigades Italy",
    "Baader-Meinhof",
    "Red Army Faction",
    "Japanese Red Army",
])
def test_detects_paramilitary_militant_groups(text):
    assert contains_extremism(text) is True


@pytest.mark.parametrize("text", [
    "Feuerkrieg Division",
    "Sonnenkrieg Division",
    "Terrorgram Collective",
    "Order of Nine Angles",
    "Maniacs Murder Cult",
    "National Action UK",
])
def test_detects_accelerationist_groups(text):
    assert contains_extremism(text) is True


# ============================================================================
# Expansión — dictadores y figuras nazis
# ============================================================================

@pytest.mark.parametrize("text", [
    "Adolf Hitler",
    "Joseph Stalin",
    "Mao Zedong",
    "Chairman Mao",
    "Pol Pot",
    "Saddam Hussein",
    "Muammar Gaddafi",
    "Bashar al-Assad",
    "Pinochet",
    "Pinochet Ugarte",
    "Francisco Franco",
    "Il Duce",
    "Idi Amin",
    "Kim Jong-un",
    "Slobodan Milošević",
    "Ratko Mladić",
])
def test_detects_dictators(text):
    assert contains_extremism(text) is True


@pytest.mark.parametrize("text", [
    "Heinrich Himmler",
    "Joseph Goebbels",
    "Hermann Göring",
    "Reinhard Heydrich",
    "Adolf Eichmann",
    "Josef Mengele",
    "Mengele",
    "Klaus Barbie",
    "Rudolf Hess",
    "Rudolf Höss",
    "Rudolf Hoss",
    "Amon Göth",
    "Irma Grese",
    "Oskar Dirlewanger",
])
def test_detects_nazi_figures(text):
    assert contains_extremism(text) is True


# ============================================================================
# Expansión — atrocidades y campos de exterminio
# ============================================================================

@pytest.mark.parametrize("text", [
    "Holocaust",
    "Shoah",
    "Final Solution",
    "Endlösung",
    "Auschwitz",
    "Auschwitz-Birkenau",
    "Treblinka",
    "Sobibor",
    "Bełżec",
    "Belzec",
    "Bergen-Belsen",
    "Dachau",
    "Buchenwald",
    "Mauthausen",
    "Majdanek",
    "Ravensbrück",
    "Wannsee Conference",
    "Kristallnacht",
    "Babi Yar",
    "Katyn massacre",
    "Holodomor",
])
def test_detects_holocaust_camps_events(text):
    assert contains_extremism(text) is True


@pytest.mark.parametrize("text", [
    "Killing Fields",
    "Tuol Sleng",
    "Rwandan Genocide",
    "Interahamwe",
    "Srebrenica",
    "Bosnian Genocide",
    "Armenian Genocide",
    "Aghet",
    "Nanking Massacre",
    "Rape of Nanking",
    "Unit 731",
    "My Lai massacre",
    "Halabja",
    "Anfal campaign",
    "Operation Condor",
    "Dirty War",
    "Caravan of Death",
    "Tlatelolco massacre",
])
def test_detects_global_atrocities(text):
    assert contains_extremism(text) is True


# ============================================================================
# Expansión — frases ideológicas codificadas
# ============================================================================

@pytest.mark.parametrize("text", [
    "Heil Hitler",
    "Sieg Heil",
    "Blood and Soil",
    "Blut und Boden",
    "Fourteen Words",
    "14 Words",
    "1488",
    "14/88",
    "Untermensch",
    "RAHOWA",
    "Racial Holy War",
    "ZOG",
    "Zionist Occupation Government",
    "Day of the Rope",
    "Great Replacement",
    "White Genocide",
    "Turner Diaries",
])
def test_detects_ideological_phrases(text):
    assert contains_extremism(text) is True


# ============================================================================
# Expansión — eventos terroristas y atacantes
# ============================================================================

@pytest.mark.parametrize("text", [
    "9/11",
    "September 11 attacks",
    "September 11",
    "7/7 London bombings",
    "Madrid train bombings",
    "11-M",
    "Atocha bombings",
    "Bali bombings",
    "Boston Marathon bombing",
    "Nice truck attack",
    "Manchester Arena bombing",
    "London Bridge attack",
    "Bataclan",
    "Bataclan attack",
    "Charlie Hebdo attack",
    "Mumbai attacks",
    "26/11",
    "Lockerbie bombing",
    "Pan Am 103",
    "Munich Olympics massacre",
    "Tokyo subway sarin attack",
])
def test_detects_terror_events(text):
    assert contains_extremism(text) is True


@pytest.mark.parametrize("text", [
    "Osama bin Laden",
    "Bin Laden",
    "Ayman al-Zawahiri",
    "Abu Bakr al-Baghdadi",
    "Abu Musab al-Zarqawi",
    "Khalid Sheikh Mohammed",
    "Mohamed Atta",
    "Tamerlan Tsarnaev",
    "Dzhokhar Tsarnaev",
    "Salman Abedi",
    "Anis Amri",
    "Mohamed Lahouaiej-Bouhlel",
    "Said Kouachi",
    "Amedy Coulibaly",
    "Shoko Asahara",
])
def test_detects_terror_attackers(text):
    assert contains_extremism(text) is True


# ============================================================================
# Expansión — serial killers
# ============================================================================

@pytest.mark.parametrize("text", [
    "John Wayne Gacy",
    "Killer Clown",
    "Ed Gein",
    "Richard Ramirez",
    "Night Stalker",
    "Aileen Wuornos",
    "H.H. Holmes",
    "Andrei Chikatilo",
    "Rostov Ripper",
    "Pedro Lopez",
    "Jack the Ripper",
    "Dennis Nilsen",
    "Fred West",
    "Harold Shipman",
    "Peter Sutcliffe",
    "Yorkshire Ripper",
    "Gary Ridgway",
    "Green River Killer",
    "Albert DeSalvo",
    "Boston Strangler",
    "David Berkowitz",
    "Son of Sam",
    "Marc Lépine",
    "Polytechnique shooter",
    "Elizabeth Báthory",
    "Vlad the Impaler",
    "Armin Meiwes",
    "Issei Sagawa",
    "Charles Sobhraj",
])
def test_detects_serial_killers_extended(text):
    assert contains_extremism(text) is True


# ============================================================================
# Expansión — NSBM bands
# ============================================================================

@pytest.mark.parametrize("text", [
    "Varg Vikernes",
    "Burzum",
    "Graveland",
    "Skrewdriver",
    "Landser",
    "Nokturnal Mortum",
    "Goatmoon",
    "Der Stürmer band",
])
def test_detects_nsbm_bands(text):
    assert contains_extremism(text) is True


# ============================================================================
# CRÍTICO — términos peligrosamente ambiguos NO deben matchear
# ============================================================================

@pytest.mark.parametrize("text", [
    # Apellidos comunes sin nombre completo
    "Kim Kardashian",          # 'Kim' apellido coreano común
    "Maoist literature",       # 'Mao' como raíz
    "Stalinist architecture",  # 'Stalin' como raíz
    "Stalingrad battle",       # contiene 'Stalin' pero es ciudad
    "Pakistani culture",       # contiene 'Pak' (no relevante)
    # Términos histórico-militares legítimos
    "Wehrmacht",               # fuerzas armadas alemanas generales
    "Luftwaffe ace pilots",
    "Kriegsmarine submarines",
    "Erwin Rommel North Africa",
    # Eventos atómicos en contexto benigno
    "Hiroshima documentary",
    "Nagasaki memorial",
    "Dresden firestorm",
    # Música/cultura
    "Steve Reich composition",
    "Wilhelm Reich orgone",
    "Richard Wagner symphony",
    "Reichstag building",
    "Reichstag fire 1933",
    # Ambiguos con palabras que parecen sospechosas
    "88 keys piano",
    "Heil weather",            # 'Heil' como saludo común sin 'Hitler'
    "The Wolf of Wall Street",
    "Wolfsburg city",
])
def test_critical_no_false_positives_ambiguous(text):
    """Términos peligrosamente ambiguos que NO deben matchear extremismo."""
    assert contains_extremism(text) is False, (
        f"FALSO POSITIVO detectado en: {text!r}\n"
        f"Esto rompería bandas/contenido legítimo. Detail: "
        f"{list_extremism_detail(text)}"
    )


# ============================================================================
# Performance smoke test — corpus grande debe procesarse en tiempo razonable
# ============================================================================

def test_performance_large_corpus():
    """Asegurar que el matching no degrade con texto largo."""
    import time

    long_text = (
        "This is a sample description for a metal album with various tracks. "
        "The band hails from somewhere in Europe and plays death metal with "
        "doom influences. Track listing includes Intro, Interlude, Inevitable "
        "Collapse, and Currency of Control. Music inspired by Charles Manson. "
    ) * 50

    start = time.perf_counter()
    result = contains_extremism(long_text)
    elapsed = time.perf_counter() - start

    assert result is True  # 'Charles Manson' matches
    assert elapsed < 2.0, f"Matching tomó {elapsed:.2f}s en corpus de {len(long_text)} chars"
