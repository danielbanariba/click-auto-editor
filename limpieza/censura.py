"""
Módulo centralizado de censura de texto para YouTube.

Detecta y censura palabras ofensivas evitando falsos positivos.
Soporta:
- Leetspeak (f*ck, sh1t, @ss, etc.)
- Variantes con acentos
- Palabras en inglés y español
- Protección contra falsos positivos (assassin, classic, etc.)
"""

import re
from typing import Optional

# ============================================================================
# CONFIGURACIÓN DE PALABRAS
# ============================================================================

# Palabras a censurar (base normalizada, sin leetspeak)
PROFANITY_EXACT = {
    # Inglés - vulgaridades sexuales
    "fuck",
    "fucking",
    "fucked",
    "fucker",
    "fucks",
    "motherfucker",
    "motherfucking",
    "mf",
    "shit",
    "shitty",
    "shitting",
    "bullshit",
    "bitch",
    "bitches",
    "bitchy",
    "asshole",
    "arsehole",
    "dick",
    "dickhead",
    "dicks",
    "cock",
    "cocks",
    "cockring",
    "cockrings",
    "pussy",
    "pussies",
    "pussys",
    "cunt",
    "cunts",
    "slut",
    "sluts",
    "slutty",
    "whore",
    "whores",
    "whorecraft",
    "whorrified",
    "bastard",
    "bastards",
    "cum",
    "cumming",
    "cumshot",
    "cums",
    "piss",
    "pissing",
    "porn",
    "porno",
    "pornstar",
    "blowjob",
    "handjob",
    "rimjob",
    "fisting",
    "fisted",
    "tits",
    "titty",
    "titties",
    "boobs",
    "boob",
    "nudes",
    "nude",
    # Inglés - contenido sexual explícito
    "orgy",
    "orgies",
    "snuff",
    "squirting",
    "squirt",
    "onlyfans",
    "femboy",
    "femboys",
    "stepsis",
    "stepbro",
    "stepmom",
    "stepdad",
    "incest",
    "hentai",
    "furry",
    "furries",
    "yiff",
    "creampie",
    "gangbang",
    "threesome",
    "foursome",
    "bdsm",
    "bondage",
    "dildo",
    "dildos",
    "vibrator",
    "masturbate",
    "masturbating",
    "masturbation",
    "ejaculate",
    "ejaculation",
    "orgasm",
    "orgasms",
    "erection",
    "erect",
    "testicle",
    "testicles",
    "testicular",
    "susticles",  # Variante meme de testicles
    "balls",
    "ballsack",
    "nutsack",
    "privates",  # Referencia a genitales
    "scrotum",
    "clitoris",
    "clit",
    "vagina",
    "vaginal",
    "penis",
    "penises",
    "anus",
    "anal",
    "rectum",
    "rectal",
    "sodomy",
    "sodomize",
    "sodomized",
    "fellatio",
    "cunnilingus",
    "doggystyle",
    "missionary",
    "cowgirl",
    "facial",
    "bukake",
    "bukkake",
    "semen",
    "sperm",
    "spermatic",
    "precum",
    "horny",
    "horniest",
    "aroused",
    "arousal",
    "erotic",
    "erotica",
    "sexy",
    "sexiest",
    "xxx",
    "nsfw",
    "lewd",
    "kinky",
    "kink",
    "fetish",
    "fetishes",
    # Actos sexuales compuestos
    "skullfuck",
    "skullfucked",
    "skullfucking",
    "facefuck",
    "facefucked",
    "facefucking",
    "throatfuck",
    "throatfucked",
    "throatfucking",
    "assfuck",
    "assfucked",
    "assfucking",
    "buttfuck",
    "buttfucked",
    "buttfucking",
    "mindfuck",
    "mindfucked",
    "mindfucking",
    "clusterfuck",
    "insemination",
    "inseminated",
    "inseminate",
    "impregnate",
    "impregnated",
    "impregnation",
    # Términos de prostitución/explotación
    "hooker",
    "hookers",
    "prostitute",
    "prostitutes",
    "prostitution",
    "pimp",
    "pimps",
    "pimping",
    # Genital/anatomía explícita
    "genital",
    "genitals",
    "genitalia",
    "boner",
    "boners",
    "erect",
    "erection",
    "womb",
    "uterus",
    "uterine",
    "ovary",
    "ovaries",
    "umbilical",
    "amniotic",
    # Fluidos corporales ofensivos
    "faeces",
    "feces",
    "faecal",
    "fecal",
    "faecaloma",
    "diarrhoea",
    "diarrhea",
    "vomit",
    "vomiting",
    "urine",
    "urinate",
    "urinating",
    "excrement",
    "defecate",
    "defecating",
    "defecation",
    # Inglés - violencia extrema / gore explícito
    "rape",
    "raped",
    "raping",
    "rapist",
    "necro",
    "necrophilia",
    "necrophiliac",
    "necroparty",
    "pedophile",
    "pedophilia",
    "pedo",
    "molest",
    "molested",
    "molesting",
    "molester",
    "snuff",
    "torture",
    "tortured",
    "torturing",
    "crucifixion",
    "crucified",
    "crucify",
    "mutilate",
    "mutilated",
    "mutilation",
    "mutilating",
    "dismember",
    "dismembered",
    "dismemberment",
    "decapitate",
    "decapitated",
    "decapitation",
    "disembowel",
    "disemboweled",
    "disembowelment",
    "eviscerate",
    "eviscerated",
    "evisceration",
    "fetal",
    "abortion",
    "aborted",
    "infanticide",
    "genocide",
    "holocaust",
    "terrorist",
    "terrorism",
    # Gore médico/anatómico gráfico
    "cadaver",
    "cadaveric",
    "cadavers",
    "corpse",
    "corpses",
    "carcass",
    "carcasses",
    "carrion",
    "putrid",
    "putridity",
    "putrefaction",
    "putrefied",
    "putrescent",
    "rotten",
    "rotting",
    "rot",
    "decompose",
    "decomposed",
    "decomposing",
    "decomposition",
    "decay",
    "decayed",
    "decaying",
    "necrotic",
    "necrosis",
    "necrotizing",
    "gangrene",
    "gangrenous",
    "septic",
    "sepsis",
    "septicemia",
    "septicemic",
    "suppurate",
    "suppurated",
    "suppuration",
    "pus",
    "purulent",
    "pustule",
    "pustules",
    "abscess",
    "abscesses",
    "lesion",
    "lesions",
    "maggot",
    "maggots",
    "vermin",
    "verminous",
    "infest",
    "infested",
    "infestation",
    # Violencia gráfica
    "beheaded",
    "behead",
    "beheading",
    "impale",
    "impaled",
    "impalement",
    "disembowel",
    "disemboweled",
    "castrate",
    "castrated",
    "castration",
    "amputation",
    "amputate",
    "amputated",
    "lacerate",
    "lacerated",
    "laceration",
    "sever",
    "severed",
    "severing",
    "hack",
    "hacked",
    "hacking",
    "slash",
    "slashed",
    "slashing",
    "stab",
    "stabbed",
    "stabbing",
    "butcher",
    "butchered",
    "butchering",
    "slaughter",
    "slaughtered",
    "slaughtering",
    "massacre",
    "massacred",
    "carnage",
    "bloodbath",
    "gory",
    "goriest",
    "viscera",
    "visceral",
    "entrails",
    "intestine",
    "intestines",
    "intestinal",
    "bowel",
    "bowels",
    # Enfermedades/condiciones ofensivas
    "syphilis",
    "syphilitic",
    "herpes",
    "gonorrhea",
    "chlamydia",
    "plague",
    "leprosy",
    "leper",
    "cholera",
    "typhoid",
    "anthrax",
    "ebola",
    "smallpox",
    # Términos de abuso/explotación
    "retard",
    "retarded",
    "cripple",
    "crippled",
    # Canibalismo (cannibal está en whitelist para Cannibal Corpse)
    "cannibalism",
    "cannibalistic",
    "anthropophagy",
    "anthropophagous",
    "anthropophagia",
    "devour",
    "devoured",
    "devouring",
    # Español - vulgaridades
    "puta",
    "putas",
    "putita",
    "puteria",
    "puto",
    "putos",
    "pendejo",
    "pendejos",
    "pendeja",
    "pendejas",
    "cabron",
    "cabrones",
    "cabrona",
    "cabronas",
    "mierda",
    "mierdas",
    "verga",
    "vergas",
    "polla",
    "pollas",
    "pene",
    "penes",
    "culo",
    "culos",
    "ano",
    "anos",
    "coño",
    "coños",
    "follar",
    "follando",
    "follada",
    "follado",
    "chingar",
    "chingada",
    "chingado",
    "chingados",
    "chingadera",
    "coger",
    "cogida",
    "cogiendo",
    "mamar",
    "mamada",
    "mamadas",
    "chupar",
    "chupada",
    "nalgas",
    "tetas",
    "panocha",
    "concha",
    "huevos",
    "cojones",
    "ojete",
    "culero",
    "culera",
    "joder",
    "jodido",
    "jodida",
    "carajo",
    "hostia",
    "hostias",
    "cagar",
    "cagada",
    "cagando",
    "mear",
    "meada",
    "meando",
    "pedo",
    "pedos",
    "porra",
    "polvo",
    # Español - violencia
    "violacion",
    "violador",
    "violada",
    "violado",
    "violar",
}

# Palabras que solo se censuran si aparecen solas (no como subcadena)
# Esto evita falsos positivos como "assassin", "classic", "grass"
# NOTA: Para contexto musical (death metal, grindcore), no censuramos
#       palabras como "kill", "murder", "gore" que son temáticas comunes.
PROFANITY_WORD_BOUNDARY = {
    "ass",  # Evita censurar: assassin, classic, bass, grass, mass, pass
    "anal",  # Evita censurar: analysis, banal, canal
    "suck",  # Evita censurar: sucker (contexto musical ok)
}

# Palabras que NUNCA deben censurarse (falsos positivos conocidos)
WHITELIST = {
    # Palabras con "ass"
    "assassin",
    "assassins",
    "assassination",
    "assassinate",
    "class",
    "classic",
    "classical",
    "classify",
    "classification",
    "bass",
    "bassist",
    "bassline",
    "basshead",
    "grass",
    "grassroots",
    "mass",
    "massacre",
    "massive",
    "masses",
    "pass",
    "passed",
    "passing",
    "passage",
    "passenger",
    "compass",
    "trespass",
    "surpass",
    "bypass",
    "embassy",
    "embarrass",
    "harass",
    "harassment",
    "crass",
    "brass",
    "cassette",
    "cassandra",
    "assault",  # Contexto musical válido
    # Palabras con "anal"
    "analysis",
    "analyze",
    "analytical",
    "analyst",
    "banal",
    "canal",
    "analgesia",
    "analog",
    "analogue",
    "finale",
    "final",
    "finals",
    # Palabras con "kill" - temática común en metal
    "skill",
    "skilled",
    "skillful",
    "skills",
    "kill",  # Temática de death metal
    "killed",
    "killer",
    "killers",
    "killing",  # Killing Joke, Dying Fetus - Killing on Adrenaline
    "overkill",  # Banda de thrash metal
    "roadkill",
    # Otras palabras temáticas de metal que no censuramos
    "murder",
    "murdered",
    "murderer",
    "murderous",
    "fist",
    "fists",  # Puño, temática común en metal
    # Palabras con "gore"
    "category",
    "categories",
    "gorgeous",
    "allegory",
    "gregory",
    # Palabras con "cum"
    "document",
    "documents",
    "documentary",
    "accumulate",
    "circumstance",
    "circumstances",
    "cucumber",
    "cumulative",
    "vacuum",
    # Nombres de bandas/artistas y términos temáticos de metal
    "cannibal",  # Cannibal Corpse
    "corpse",  # Cannibal Corpse, Corpse, etc.
    "carcass",  # Banda de grindcore
    "napalm",  # Napalm Death
    "obituary",  # Banda de death metal
    "slayer",  # Banda de thrash metal
    "exodus",  # Banda de thrash metal
    "massacre",  # Banda de death metal
    "autopsy",  # Banda de death metal
    "morbid",  # Morbid Angel
    "death",  # Banda de death metal
    "dying",  # Dying Fetus
    "fetus",  # Dying Fetus
    "butcher",
    "butchered",
    "butchering",  # Términos comunes en metal
    "slaughter",
    "slaughtered",  # Términos comunes en metal
    "blood",
    "bloody",
    "bleeding",
    "dead",
    "deadly",
    "evil",
    "dark",
    "darkness",
    "doom",
    "doomed",
    "grave",
    "graveyard",
    "tomb",
    "tombs",
    "death",
    "deathly",
    "demon",
    "demons",
    "demonic",
    "satan",
    "satanic",
    "satanism",
    "hell",
    "hellish",
    "infernal",
    "inferno",
    "abyss",
    "chaos",
    "chaotic",
    "brutal",
    "brutality",
    "extreme",
    "savage",
    "vicious",
    "violent",
    "violence",
    "war",
    "warfare",
    "destruction",
    "destructive",
    "annihilation",
    "annihilate",
}

# ============================================================================
# MAPEO LEETSPEAK Y ACENTOS
# ============================================================================

LEET_MAP = str.maketrans(
    {
        # Leetspeak común
        "@": "a",
        "4": "a",
        "$": "s",
        "5": "s",
        "!": "i",
        "1": "i",
        "0": "o",
        "3": "e",
        "7": "t",
        # Acentos y caracteres especiales
        "á": "a",
        "à": "a",
        "â": "a",
        "ä": "a",
        "å": "a",
        "ã": "a",
        "æ": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ï": "i",
        "ó": "o",
        "ò": "o",
        "ô": "o",
        "ö": "o",
        "õ": "o",
        "ø": "o",
        "ú": "u",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ñ": "n",
        "ç": "c",
        # Caracteres de relleno (se eliminan para normalizar)
        "*": "",
        "-": "",
        "_": "",
        ".": "",
        "'": "",
        "`": "",
        "´": "",
    }
)

# Patrones de palabras censuradas sin vocales (para detectar f*ck, sh*t, etc.)
# Mapea la versión sin vocales -> palabra original
VOWELLESS_PATTERNS = {
    # fuck variantes (f***, f**k, f*ck, etc.)
    "fck": "fuck",
    "fk": "fuck",
    "fuc": "fuck",
    "fuk": "fuck",
    "fcking": "fucking",
    "fkng": "fucking",
    "fckng": "fucking",
    "fukng": "fucking",
    "fckd": "fucked",
    "fkd": "fucked",
    "fukd": "fucked",
    "fckr": "fucker",
    "fkr": "fucker",
    # shit variantes
    "sht": "shit",
    "sht": "shit",
    "shtt": "shit",
    "shttng": "shitting",
    "shtting": "shitting",
    "shttng": "shitting",
    # bitch variantes
    "btch": "bitch",
    "bch": "bitch",
    "btc": "bitch",
    # dick variantes (d***, d**k, d*ck)
    "dck": "dick",
    "dk": "dick",
    "dc": "dick",
    # cock variantes (c***, c**k, c*ck)
    "cck": "cock",
    "ck": "cock",
    "cc": "cock",
    "cckrng": "cockring",
    "cckrngs": "cockrings",
    # pussy variantes (p****, p***y, p*ssy)
    "pss": "pussy",
    "pssy": "pussy",
    "psy": "pussy",
    "ps": "pussy",
    # cunt variantes (c***, c**t, c*nt)
    "cnt": "cunt",
    "ct": "cunt",
    "cn": "cunt",
    "cnts": "cunts",
    "cts": "cunts",
    # cum variantes (c**, c*m)
    "cm": "cum",
    "cmming": "cumming",
    "cmmng": "cumming",
    "cmmg": "cumming",
    # slut/whore variantes
    "slt": "slut",
    "slts": "sluts",
    "whr": "whore",
    "whrs": "whores",
    # anal variantes (a***)
    "nl": "anal",
    "anl": "anal",
    # ass variantes (a**)
    "ss": "ass",
    # español
    "pt": "puta",
    "pts": "putas",
    "mrd": "mierda",
    "vrg": "verga",
    "clo": "culo",
    "plln": "pollon",
}

_COMPACT_PROFANITY_TOKENS = tuple(
    sorted(
        {
            word
            for word in PROFANITY_EXACT
            if len(word) >= 4
            and word not in WHITELIST
            and word not in PROFANITY_WORD_BOUNDARY
        },
        key=len,
        reverse=True,
    )
)

# ============================================================================
# FUNCIONES DE CENSURA
# ============================================================================


def normalize_word(word: str) -> str:
    """Normaliza una palabra eliminando leetspeak y acentos."""
    return word.lower().translate(LEET_MAP)


def normalize_compact_text(text: str) -> str:
    """Normaliza un texto y elimina separadores para revisar slugs/URLs."""
    return re.sub(r"[^a-z0-9]+", "", normalize_word(text))


def is_whitelisted(word: str) -> bool:
    """Verifica si una palabra está en la lista blanca."""
    normalized = normalize_word(word)
    return normalized in WHITELIST


def should_censor(word: str) -> bool:
    """
    Determina si una palabra debe ser censurada.

    Returns:
        True si la palabra debe censurarse, False si no.
    """
    # Primero verificar whitelist
    if is_whitelisted(word):
        return False

    normalized = normalize_word(word)

    # Coincidencia exacta con palabras de censura
    if normalized in PROFANITY_EXACT:
        return True

    # Palabras que requieren límite de palabra (ya están aisladas aquí)
    if normalized in PROFANITY_WORD_BOUNDARY:
        return True

    # Detectar patrones sin vocales (f*ck -> fck -> fuck)
    if normalized in VOWELLESS_PATTERNS:
        return True

    return False


def censor_word(word: str) -> str:
    """
    Censura una palabra manteniendo la primera letra visible.

    Ejemplos:
        "fuck" -> "f***"
        "shit" -> "s***"
        "mf" -> "**"
    """
    if len(word) <= 2:
        return "*" * len(word)
    return word[0] + "*" * (len(word) - 1)


# Patrón compartido para captura de palabras con asteriscos
# Usamos lookbehind/lookahead en lugar de \b para manejar asteriscos correctamente
_WORD_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z@\$!?'´`][A-Za-z0-9@\$!?\*'´`]*(?![A-Za-z0-9])", re.UNICODE
)


def censor_profanity(text: str) -> str:
    """
    Censura palabras ofensivas en un texto.

    Esta función detecta y censura palabras ofensivas mientras
    evita falsos positivos (ej: "assassin" no se censura).

    Args:
        text: Texto a procesar.

    Returns:
        Texto con palabras ofensivas censuradas.

    Ejemplos:
        >>> censor_profanity("What the fuck")
        "What the f***"
        >>> censor_profanity("Classic Assassin")
        "Classic Assassin"  # Sin cambios
    """
    if not text:
        return text

    def replace_match(match: re.Match) -> str:
        word = match.group(0)
        if should_censor(word):
            return censor_word(word)
        return word

    return _WORD_PATTERN.sub(replace_match, text)


def contains_profanity(text: str) -> bool:
    """
    Verifica si un texto contiene palabras ofensivas.

    Args:
        text: Texto a verificar.

    Returns:
        True si contiene palabras ofensivas, False si no.
    """
    if not text:
        return False

    for match in _WORD_PATTERN.finditer(text):
        if should_censor(match.group(0)):
            return True

    return False


def contains_profanity_fragment(text: str) -> bool:
    """
    Detecta groserías dentro de slugs o URLs compactas.

    Ejemplos:
        "fucking-death" -> True
        "analcunt" -> True
        "cannibalcorpse" -> False
    """
    if not text:
        return False

    if contains_profanity(text):
        return True

    compact = normalize_compact_text(text)
    if not compact or compact in WHITELIST:
        return False

    for token in _COMPACT_PROFANITY_TOKENS:
        if token in compact:
            return True

    return False


def list_profanity(text: str) -> list[str]:
    """
    Lista todas las palabras ofensivas encontradas en un texto.

    Args:
        text: Texto a analizar.

    Returns:
        Lista de palabras ofensivas encontradas (originales, no normalizadas).
    """
    if not text:
        return []

    found = []
    for match in _WORD_PATTERN.finditer(text):
        word = match.group(0)
        if should_censor(word):
            found.append(word)

    return found


# ============================================================================
# UTILIDADES ADICIONALES
# ============================================================================


def add_profanity_word(word: str, require_boundary: bool = False) -> None:
    """
    Añade una palabra a la lista de censura en tiempo de ejecución.

    Args:
        word: Palabra a añadir (se normalizará automáticamente).
        require_boundary: Si True, solo censura cuando es palabra completa.
    """
    normalized = normalize_word(word)
    if require_boundary:
        PROFANITY_WORD_BOUNDARY.add(normalized)
    else:
        PROFANITY_EXACT.add(normalized)


def add_whitelist_word(word: str) -> None:
    """
    Añade una palabra a la lista blanca en tiempo de ejecución.

    Args:
        word: Palabra que nunca debe censurarse.
    """
    normalized = normalize_word(word)
    WHITELIST.add(normalized)


# ============================================================================
# PRUEBAS RÁPIDAS
# ============================================================================

if __name__ == "__main__":
    # Pruebas de funcionamiento
    test_cases = [
        # (input, expected_censored, description)
        ("What the fuck", True, "Palabra exacta"),
        ("f*ck you", True, "Leetspeak con asterisco"),
        ("sh1t happens", True, "Leetspeak con número"),
        ("Classic Assassin", False, "Falso positivo - assassin"),
        ("Bass player", False, "Falso positivo - bass"),
        ("Mass destruction", False, "Falso positivo - mass"),
        ("Skill level", False, "Falso positivo - skill"),
        ("Data analysis", False, "Falso positivo - analysis"),
        ("That's bullshit", True, "Palabra compuesta"),
        ("Puta madre", True, "Español"),
        ("Hijo de puta", True, "Español en frase"),
        ("Category Five", False, "Falso positivo - category"),
        ("Gorgeous view", False, "Falso positivo - gorgeous"),
        ("Cannibal Corpse", False, "Nombre de banda - whitelist"),
        ("Napalm Death", False, "Nombre de banda - whitelist"),
        ("Your ass is grass", True, "ass aislado = censurar"),
        ("Anal Cunt", True, "Banda con nombre explícito"),
    ]

    print("=" * 60)
    print("TEST DE CENSURA")
    print("=" * 60)

    passed = 0
    failed = 0

    for text, should_contain, desc in test_cases:
        has_profanity = contains_profanity(text)
        censored = censor_profanity(text)

        if has_profanity == should_contain:
            status = "✓"
            passed += 1
        else:
            status = "✗"
            failed += 1

        print(f"{status} {desc}")
        print(f"  Input:    '{text}'")
        print(f"  Output:   '{censored}'")
        print(f"  Expected: {'contiene' if should_contain else 'limpio'}")
        print()

    print("=" * 60)
    print(f"Resultado: {passed}/{passed + failed} pruebas pasadas")
    print("=" * 60)
