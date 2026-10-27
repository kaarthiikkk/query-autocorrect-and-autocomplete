from language_detector import detect_language
from romanized_detector import detect_romanized_language

try:
    from data_loader import LANGUAGE_CODE_MAP, SUPPORTED_LANGUAGES
except ImportError:
    LANGUAGE_CODE_MAP = {
        "en": "english", "ta": "tamil", "ta_rom": "tamil_romanized",
        "es": "spanish", "de": "german", "fr": "french",
        "it": "italian", "pt": "portuguese", "nl": "dutch"
    }
    SUPPORTED_LANGUAGES = [
        "english", "tamil", "tamil_romanized", "spanish", "german",
        "french", "italian", "portuguese", "dutch"
    ]

EUROPEAN_LANGUAGES = ["spanish", "german", "french", "italian", "portuguese", "dutch"]

# Distinctive character sets
STRONG_CHAR_CUES = {
    "spanish": set("ñ¿¡"),
    "german": set("ßäöü"),
    "portuguese": set("ãõ"),
    "french": set("œæ"),
}

# Exclusive/strong distinctive vocabulary
EUROPEAN_DISTINCTIVE_VOCAB = {
    "spanish": {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "del", "al", "por", "para", "con",
        "como", "su", "sus", "este", "esta", "estos", "estas", "pero", "más", "gracias", "bueno",
        "hola", "donde", "cuando", "amigo", "amigos", "todos", "español", "información", "buenos",
        "dias", "día", "porfavor", "muy", "también", "después", "están", "días", "informacion",
        "informes", "informe"
    },
    "german": {
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "einem", "einen", "eines",
        "und", "nicht", "von", "mit", "für", "im", "auf", "ist", "sich", "auch", "als", "nach",
        "wie", "bei", "oder", "aber", "deutsch", "danke", "bitte", "heute", "immer", "wieder",
        "jahr", "jahre", "neue", "neuen", "wir", "ihr", "alles", "schon", "sehr", "über",
        "information", "informationen", "guten", "morgen", "abend", "informatik", "menschen"
    },
    "french": {
        "le", "la", "les", "un", "une", "des", "du", "et", "que", "qui", "dans", "pour",
        "pas", "sur", "ce", "cette", "ces", "avec", "tout", "tous", "faire", "mais", "nous",
        "vous", "ils", "elles", "leur", "leurs", "bonjour", "merci", "français", "être", "avoir",
        "plus", "même", "bien", "aussi", "autre", "sans", "très", "peut", "comme", "quand",
        "informations", "informatique"
    },
    "italian": {
        "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "da", "su", "per", "tra", "fra",
        "del", "dello", "della", "dei", "degli", "delle", "nel", "nello", "nella", "nei", "negli", "nelle",
        "non", "che", "sono", "questo", "questa", "quello", "quella", "anche", "ciao", "grazie",
        "italiano", "tutto", "tutti", "molto", "sempre", "tempo", "casa", "cosa", "cose",
        "come", "più", "ancora", "dopo", "bene", "prima", "informazioni", "informazione"
    },
    "portuguese": {
        "o", "os", "as", "um", "uma", "uns", "umas", "do", "da", "dos", "das", "no", "na",
        "nos", "nas", "em", "pelo", "pela", "pelos", "pelas", "com", "para", "por", "como",
        "mais", "mas", "se", "não", "está", "estão", "são", "obrigado", "obrigada", "português",
        "muito", "fazer", "isso", "aqui", "tudo", "todos", "toda", "todas", "você", "vocês",
        "informações", "informação"
    },
    "dutch": {
        "het", "een", "van", "en", "op", "te", "dat", "die", "voor", "met", "zijn", "niet",
        "aan", "om", "als", "er", "maar", "bij", "ook", "tot", "je", "we", "ze", "informatie",
        "nederlands", "alsjeblieft", "dank", "dankjewel", "goed", "veel", "jaar", "nieuwe",
        "heel", "wel", "over", "door", "wat", "mijn", "geen", "naar", "hebben", "worden"
    },
}

# Exclusive single-word strong triggers
EXCLUSIVE_TRIGGERS = {
    "gracias": "spanish", "buenos": "spanish", "español": "spanish", "información": "spanish",
    "danke": "german", "bitte": "german", "deutsch": "german", "morgen": "german",
    "merci": "french", "bonjour": "french", "français": "french",
    "grazie": "italian", "ciao": "italian", "italiano": "italian", "informazione": "italian", "informazioni": "italian",
    "obrigado": "portuguese", "obrigada": "portuguese", "português": "portuguese", "informação": "portuguese", "informações": "portuguese",
    "dankjewel": "dutch", "nederlands": "dutch", "informatie": "dutch", "alsjeblieft": "dutch",
}


def detect_european_language(text):
    """Detects if Latin-script text has evidence of Spanish, German, French, Italian, Portuguese, or Dutch."""
    text_lower = text.lower().strip()
    if not text_lower:
        return None, 0.0

    # 1. Exclusive trigger word lookup
    words = text_lower.split()
    for w in words:
        if w in EXCLUSIVE_TRIGGERS:
            return EXCLUSIVE_TRIGGERS[w], 0.95

    # 2. Character-level strong signals
    for lang, char_set in STRONG_CHAR_CUES.items():
        if any(c in char_set for c in text_lower):
            return lang, 0.95

    # 3. Dutch 'ij' digraph (excluding foreign proper nouns like beijing)
    if "ij" in text_lower and not any(text_lower.startswith(p) for p in ["beij", "fij"]):
        return "dutch", 0.90

    # 4. Suffix & distinctive stopword scoring
    scores = {lang: 0.0 for lang in EUROPEAN_LANGUAGES}
    for w in words:
        for lang, vocab in EUROPEAN_DISTINCTIVE_VOCAB.items():
            if w in vocab:
                scores[lang] += 1.0

        if w.endswith("ción") or w.endswith("ciones"):
            scores["spanish"] += 1.5
        elif w.endswith("zione") or w.endswith("zioni"):
            scores["italian"] += 1.5
        elif w.endswith("ção") or w.endswith("ções"):
            scores["portuguese"] += 1.5
        elif w.endswith("tie") and len(w) > 5:
            scores["dutch"] += 0.9
        elif w.endswith("keit") or w.endswith("heit") or w.endswith("schaft"):
            scores["german"] += 1.5

    best_lang, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score >= 0.8:
        conf = min(0.95, 0.65 + best_score * 0.15)
        return best_lang, conf

    return None, 0.0


def analyze_token(token, target_language=None):
    """Classify one token for language-specific candidate generation."""
    token = token.strip()
    if not token:
        return {
            "token": token,
            "language": "unknown",
            "type": "unknown",
            "confidence": 0.0
        }

    # Respect explicit target_language selection if provided
    if target_language and str(target_language).lower() not in {"auto", "none"}:
        canon = LANGUAGE_CODE_MAP.get(str(target_language).lower(), str(target_language).lower())
        if "_romanized" in canon:
            base_lang = canon.replace("_romanized", "")
            return {"token": token, "language": base_lang, "type": "romanized", "confidence": 1.0}
        if canon in {"tamil", "hindi", "telugu"} and detect_language(token) == "english":
            return {"token": token, "language": canon, "type": "romanized", "confidence": 1.0}
        return {"token": token, "language": canon, "type": "native_script", "confidence": 1.0}

    script_language = detect_language(token)

    if script_language in {"tamil", "hindi", "telugu"}:
        return {
            "token": token,
            "language": script_language,
            "type": "native_script",
            "confidence": 1.0
        }

    if script_language == "mixed":
        return {
            "token": token,
            "language": "mixed",
            "type": "code_switched",
            "confidence": 1.0
        }

    # Romanized Indian language check (Tamil, Hindi, Telugu)
    romanized = detect_romanized_language(token)
    if romanized["language"] != "unknown":
        return {
            "token": token,
            "language": romanized["language"],
            "type": "romanized",
            "confidence": romanized["confidence"]
        }

    # European foreign language check
    euro_lang, euro_conf = detect_european_language(token)
    if euro_lang:
        return {
            "token": token,
            "language": euro_lang,
            "type": "native_script",
            "confidence": euro_conf
        }

    if script_language == "english":
        return {
            "token": token,
            "language": "english",
            "type": "native_script",
            "confidence": 1.0
        }

    return {
        "token": token,
        "language": "unknown",
        "type": "unknown",
        "confidence": 0.0
    }


def analyze_language(query, target_language=None):
    """Combines script-based, European language, and Romanized language detection."""
    query = query.strip()
    if not query:
        return {
            "language": "unknown",
            "type": "unknown",
            "confidence": 0.0
        }

    # Explicit target_language override
    if target_language and str(target_language).lower() not in {"auto", "none"}:
        canon = LANGUAGE_CODE_MAP.get(str(target_language).lower(), str(target_language).lower())
        if "_romanized" in canon:
            base_lang = canon.replace("_romanized", "")
            return {"language": base_lang, "type": "romanized", "confidence": 1.0}
        if canon in {"tamil", "hindi", "telugu"} and detect_language(query) == "english":
            return {"language": canon, "type": "romanized", "confidence": 1.0}
        return {"language": canon, "type": "native_script", "confidence": 1.0}

    # 1. Script-based detection
    script_language = detect_language(query)
    if script_language in {"tamil", "hindi", "telugu"}:
        return {
            "language": script_language,
            "type": "native_script",
            "confidence": 1.0
        }

    if script_language == "mixed":
        return {
            "language": "mixed",
            "type": "code_switched",
            "confidence": 1.0
        }

    # 2. Romanized Indian language detection
    romanized = detect_romanized_language(query)
    if romanized["language"] != "unknown":
        return {
            "language": romanized["language"],
            "type": "romanized",
            "confidence": romanized["confidence"]
        }

    # 3. European language detection
    euro_lang, euro_conf = detect_european_language(query)
    if euro_lang:
        return {
            "language": euro_lang,
            "type": "native_script",
            "confidence": euro_conf
        }

    # 4. English fallback for Latin-script text
    if script_language == "english":
        return {
            "language": "english",
            "type": "native_script",
            "confidence": 1.0
        }

    return {
        "language": "unknown",
        "type": "unknown",
        "confidence": 0.0
    }
