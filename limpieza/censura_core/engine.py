"""Motor de censura — decide y aplica redacción sobre texto."""

from __future__ import annotations

import itertools
import re
from functools import lru_cache

from limpieza.censura_core.loader import CensuraData, load_all
from limpieza.censura_core.normalizer import build_translator


# Patrón Unicode para detectar palabras incluyendo leetspeak y obfuscación.
# Usamos lookbehind/lookahead en lugar de \b para manejar asteriscos correctamente.
_WORD_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[A-Za-z@\$!?'´`\u00c0-\u024f\u0370-\u03ff\u0400-\u04ff]"
    r"[A-Za-z0-9@\$!?\*'´`\u00c0-\u024f\u0370-\u03ff\u0400-\u04ff]*"
    r"(?![A-Za-z0-9])",
    re.UNICODE,
)

# Detecta palabras obfuscadas con separadores tipo "f-u-c-k", "s.h.i.t", "b_i_t_c_h".
# Requiere al menos 3 letras separadas por [-._] para evitar falsos positivos con
# abreviaciones cortas ("A.I.", "U.S."). Todas las letras individuales de 1 char.
_SPLIT_WORD_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[A-Za-z](?:[-._][A-Za-z]){2,}"
    r"(?![A-Za-z0-9])",
    re.UNICODE,
)

_SPLIT_SEPARATORS = re.compile(r"[-._]")

_VOWELS = "aeiou"


class CensuraEngine:
    """Motor principal de detección y redacción de groserías."""

    def __init__(self, data: CensuraData):
        self.data = data
        self._translator = build_translator(data.leet_table)
        self._word_re = _WORD_PATTERN

        # Tokens utilizados para detección compacta (slugs/URLs).
        # Solo palabras "largas" que no requieren word-boundary y no están
        # en whitelist, ordenadas por longitud descendente para que matches
        # largos ganen a cortos.
        self._compact_tokens: tuple[str, ...] = tuple(
            sorted(
                {
                    w
                    for w in data.blacklist
                    if len(w) >= 4
                    and w not in data.whitelist
                    and w not in data.word_boundary_only
                },
                key=len,
                reverse=True,
            )
        )

        # Aho-Corasick automaton para match O(N+M) sobre compact text.
        # Construido en load-time, se reutiliza en cada contains_profanity_fragment.
        self._ac_automaton = self._build_aho_corasick(self._compact_tokens)

        # Índice blacklist + obfuscations por longitud — acelera matching
        # posicional cuando hay asteriscos como wildcard.
        self._blacklist_by_len: dict[int, tuple[str, ...]] = {}
        all_words = data.blacklist | set(data.obfuscations.keys())
        for w in all_words:
            self._blacklist_by_len.setdefault(len(w), ())
        for w in all_words:
            self._blacklist_by_len[len(w)] = self._blacklist_by_len[len(w)] + (w,)

        # Stemmers por idioma (lazy-inicializados, uno por idioma cargado).
        self._stemmers: dict[str, object] = {}
        self._init_stemmers()

        # Language detector opcional — si está disponible, activa palabras
        # ambiguas cross-idioma solo cuando detecta el idioma correcto.
        self._lang_detector = None
        self._init_language_detector()

        # Cache decisiones por palabra — el mismo texto suele tener
        # palabras repetidas (ej: "fuck" en una estrofa) y el normalize
        # no es gratis.
        self._should_censor_cached = lru_cache(maxsize=8192)(self._should_censor_impl)

    @staticmethod
    def _build_aho_corasick(tokens: tuple[str, ...]):
        """Construye un automaton Aho-Corasick. Retorna None si la lib no está."""
        try:
            import ahocorasick
        except ImportError:
            return None
        if not tokens:
            return None
        automaton = ahocorasick.Automaton()
        for token in tokens:
            automaton.add_word(token, token)
        automaton.make_automaton()
        return automaton

    def _init_stemmers(self) -> None:
        """Inicializa stemmers Snowball para los idiomas cargados."""
        try:
            import snowballstemmer
        except ImportError:
            return
        from limpieza.censura_core.loader import SNOWBALL_ALGORITHMS
        for lang in self.data.loaded_languages:
            algo = SNOWBALL_ALGORITHMS.get(lang)
            if algo and self.data.stems_by_lang.get(lang):
                self._stemmers[lang] = snowballstemmer.stemmer(algo)

    def _init_language_detector(self) -> None:
        """Construye detector lingua si hay ambiguous words y la lib está."""
        if not self.data.ambiguous_by_lang:
            return
        try:
            from lingua import Language, LanguageDetectorBuilder
        except ImportError:
            return

        # Mapeo código ISO → Language enum de lingua.
        code_to_lang = {
            "en": Language.ENGLISH, "es": Language.SPANISH,
            "pt": Language.PORTUGUESE, "fr": Language.FRENCH,
            "it": Language.ITALIAN, "de": Language.GERMAN,
            "nl": Language.DUTCH, "ru": Language.RUSSIAN,
            "sv": Language.SWEDISH, "no": Language.NYNORSK,
            "da": Language.DANISH, "fi": Language.FINNISH,
            "ro": Language.ROMANIAN, "tr": Language.TURKISH,
            "hu": Language.HUNGARIAN,
        }
        langs = [code_to_lang[c] for c in self.data.loaded_languages if c in code_to_lang]
        if len(langs) < 2:
            return

        # Builder es pesado pero se construye una vez por engine.
        self._lang_detector = (
            LanguageDetectorBuilder.from_languages(*langs).build()
        )

    def _detect_active_ambiguous(self, text: str) -> frozenset[str]:
        """Detecta idioma y retorna set de ambiguous del idioma (o vacío).

        Solo se activa si:
        - Hay detector cargado
        - Texto >= 15 chars (evitar falsos positivos en strings cortas)
        - El idioma detectado tiene palabras ambiguas definidas
        """
        if self._lang_detector is None or len(text) < 15:
            return frozenset()
        lang = self._lang_detector.detect_language_of(text)
        if lang is None:
            return frozenset()
        code = lang.iso_code_639_1.name.lower()
        return frozenset(self.data.ambiguous_by_lang.get(code, set()))

    def _should_censor_with_ambiguous(
        self, word: str, ambiguous: frozenset[str]
    ) -> bool:
        if self.should_censor(word):
            return True
        if not ambiguous:
            return False
        normalized = self.normalize(word)
        if normalized in self.data.whitelist:
            return False
        return normalized in ambiguous

    # ------------------------------------------------------------------
    # Normalización
    # ------------------------------------------------------------------
    def normalize(self, word: str) -> str:
        return word.lower().translate(self._translator)

    def normalize_compact(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", self.normalize(text))

    def is_whitelisted(self, word: str) -> bool:
        return self.normalize(word) in self.data.whitelist

    # ------------------------------------------------------------------
    # Decisión de censura
    # ------------------------------------------------------------------
    def _should_censor_impl(self, word: str) -> bool:
        # Si la palabra tiene asteriscos, probar match posicional contra
        # blacklist (cada `*` es un wildcard de un carácter). Lo hacemos
        # ANTES de normalizar porque normalize borra los asteriscos (filler).
        # Ejemplo: "f***" con 3 asteriscos matchea "fuck" (longitud 4... ajustar).
        lower = word.lower()
        if "*" in lower:
            if self._match_asterisk_wildcard(lower):
                return True

        normalized = self.normalize(word)

        # Whitelist siempre gana.
        if normalized in self.data.whitelist:
            return False

        if normalized in self.data.blacklist:
            return True

        if normalized in self.data.word_boundary_only:
            return True

        if normalized in self.data.obfuscations:
            return True

        # Último recurso: stemming por idioma. Detecta inflexiones como
        # "fucking" → stem "fuck", "fodendo" → stem "fod" (match con
        # "foda/foder/fodido"). Solo se activa si hay stemmers cargados.
        # Ignorar palabras muy cortas para evitar falsos positivos.
        #
        # Clave: si stem == normalized, NO matchear por stem — si fuera
        # grosería estaría en blacklist directo. Esto evita falsos positivos
        # como "ball" (stem de "balls") o "tit" (stem de "tits").
        if len(normalized) >= 4:
            for lang, stemmer in self._stemmers.items():
                stem = stemmer.stemWord(normalized)
                if stem and stem != normalized and stem in self.data.stems_by_lang[lang]:
                    return True

        return False

    def _match_asterisk_wildcard(self, word: str) -> bool:
        """Chequea si `word` (con `*` como wildcards) matchea blacklist.

        Cada asterisco representa exactamente UN carácter cualquiera. El
        match se hace posición a posición contra palabras de la misma
        longitud en blacklist y obfuscations.

        Ejemplos:
            "f***"  → matchea "fuck" (4 letras, posiciones 1/2/3 libres)
            "c*ño"  → primero aplica translator a "c" y "ño" (ñ→n), luego
                      matchea "cono" (c?no con ?=o)
            "sh*t"  → matchea "shit" (o "shut" si estuviera en blacklist)

        No intenta colisiones cross-idioma excluyendo whitelist: si la
        palabra matcheada está en whitelist, no censura.
        """
        # Aplicamos translator a todos los caracteres EXCEPTO `*` (filler
        # los borraría). Construimos el patrón posicional preservando `*`.
        pattern_chars: list[str] = []
        for c in word:
            if c == "*":
                pattern_chars.append("*")
            else:
                # Traducir este carácter sin eliminarlo
                translated = c.translate(self._translator)
                if translated == "":
                    # Era un filler no-asterisco — ignorar posición
                    continue
                pattern_chars.append(translated)
        if "*" not in pattern_chars:
            return False
        n = len(pattern_chars)

        # Buscar en blacklist + obfuscations palabras de misma longitud.
        for candidate in self._blacklist_by_len.get(n, ()):
            if candidate in self.data.whitelist:
                continue
            if all(
                p == "*" or p == c
                for p, c in zip(pattern_chars, candidate)
            ):
                return True
        return False

    def should_censor(self, word: str) -> bool:
        return self._should_censor_cached(word)

    # ------------------------------------------------------------------
    # Redacción
    # ------------------------------------------------------------------
    @staticmethod
    def censor_word(word: str) -> str:
        """f*** para palabras largas, ** para palabras de <=2 caracteres."""
        if len(word) <= 2:
            return "*" * len(word)
        return word[0] + "*" * (len(word) - 1)

    def censor_text(self, text: str) -> str:
        if not text:
            return text

        # Primero colapsar palabras obfuscadas con separadores
        # ("f-u-c-k" → "f***") antes del pase general.
        text = self._censor_split_words(text)

        ambiguous = self._detect_active_ambiguous(text)

        def replace(match: re.Match) -> str:
            word = match.group(0)
            if self._should_censor_with_ambiguous(word, ambiguous):
                return self.censor_word(word)
            return word

        return self._word_re.sub(replace, text)

    def _censor_split_words(self, text: str) -> str:
        """Detecta palabras como "f-u-c-k" y las reemplaza por su censura."""
        def replace(m: re.Match) -> str:
            split = m.group(0)
            joined = _SPLIT_SEPARATORS.sub("", split)
            if self.should_censor(joined):
                return self.censor_word(joined)
            return split
        return _SPLIT_WORD_PATTERN.sub(replace, text)

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------
    def contains_profanity(self, text: str) -> bool:
        if not text:
            return False
        ambiguous = self._detect_active_ambiguous(text)
        # Chequeo por palabras normales (+ ambiguas si aplica)
        for m in self._word_re.finditer(text):
            if self._should_censor_with_ambiguous(m.group(0), ambiguous):
                return True
        # Chequeo por palabras con separadores (f-u-c-k)
        for m in _SPLIT_WORD_PATTERN.finditer(text):
            joined = _SPLIT_SEPARATORS.sub("", m.group(0))
            if self.should_censor(joined):
                return True
        return False

    def list_profanity(self, text: str) -> list[str]:
        if not text:
            return []
        found: list[str] = []
        for match in self._word_re.finditer(text):
            word = match.group(0)
            if self.should_censor(word):
                found.append(word)
        for match in _SPLIT_WORD_PATTERN.finditer(text):
            split = match.group(0)
            joined = _SPLIT_SEPARATORS.sub("", split)
            if self.should_censor(joined):
                found.append(split)
        return found

    def contains_profanity_fragment(self, text: str) -> bool:
        """Detección en slugs/URLs donde las palabras aparecen pegadas."""
        if not text:
            return False

        if self.contains_profanity(text):
            return True

        compact = self.normalize_compact(text)
        if not compact or compact in self.data.whitelist:
            return False

        # Aho-Corasick si disponible — O(N+M) vs O(N×M) del loop.
        if self._ac_automaton is not None:
            for _, _ in self._ac_automaton.iter(compact):
                return True
            return False

        # Fallback cuando pyahocorasick no está instalado.
        for token in self._compact_tokens:
            if token in compact:
                return True
        return False

    # ------------------------------------------------------------------
    # Extensiones runtime
    # ------------------------------------------------------------------
    def add_profanity(self, word: str, require_boundary: bool = False) -> None:
        normalized = self.normalize(word)
        target = self.data.word_boundary_only if require_boundary else self.data.blacklist
        target.add(normalized)
        self._should_censor_cached.cache_clear()

    def add_whitelist(self, word: str) -> None:
        self.data.whitelist.add(self.normalize(word))
        self._should_censor_cached.cache_clear()


# ----------------------------------------------------------------------
# Singleton default para callers que no quieren pasarse data
# ----------------------------------------------------------------------
_DEFAULT_ENGINE: CensuraEngine | None = None


def get_default_engine() -> CensuraEngine:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = CensuraEngine(load_all())
    return _DEFAULT_ENGINE


def reset_default_engine() -> None:
    """Fuerza recarga de datos — útil en tests."""
    global _DEFAULT_ENGINE
    _DEFAULT_ENGINE = None
