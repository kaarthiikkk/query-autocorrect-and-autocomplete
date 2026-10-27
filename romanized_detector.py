import os
import csv
from corrector import damerau_levenshtein

# Starter vocabulary for detecting
# Romanized Indian language words.

ROMANIZED_WORDS = {
    "tamil": {
        "venum",
        "vendum",
        "enna",
        "epdi",
        "eppadi",
        "irukku",
        "iruka",
        "sapadu",
        "saapadu",
        "sapda",
        "ponga",
        "vaanga",
        "inge",
        "anga",
        "romba",
        "nalla",
        "illai",
        "illa",
        "vandha",
        "varum",
        "pogum",
        "enakku",
        "enaku",
        "ungalukku",
        "thanni",
        "kudikka",
        "biryani",
        "briyani",
        "kadai",
        "thevai",
        "venom"
    },

    "telugu": {
        "namaskaram",
        "namaste",
        "dhanyavadalu",
        "kavali",
        "kaavali",
        "undi",
        "unnadi",
        "ela",
        "elaa",
        "enti",
        "enduku",
        "naaku",
        "naku",
        "neeku",
        "meeru",
        "nenu",
        "manam",
        "ledu",
        "vachindi",
        "vastundi",
        "veldam",
        "vellali",
        "cheppu",
        "cheyyali",
        "tinnava",
        "tinandi",
        "bagundi",
        "bagunnara",
        "unnav",
        "unnaru"
    },

    "hindi": {
        "namaste",
        "namaskar",
        "shukriya",
        "dhanyawad",
        "dhanyavad",
        "hai",
        "hain",
        "kya",
        "kaise",
        "kaisa",
        "kaisi",
        "mujhe",
        "mujhko",
        "aap",
        "aapko",
        "mera",
        "meri",
        "mere",
        "chahiye",
        "nahi",
        "nahin",
        "acha",
        "accha",
        "theek",
        "thik",
        "bhai",
        "karo",
        "karna",
        "kyun",
        "kahan",
        "jana",
        "jaana",
        "khana"
    }
}

HIGH_FREQ_TAMIL = [
    "venum", "vendum", "biryani", "briyani", "sapda", "saapadu",
    "sapadu", "kadai", "thevai", "thanni", "kudikka", "irukku",
    "illai", "illa", "enakku", "enaku", "ungalukku", "romba",
    "nalla", "ponga", "vaanga", "inge", "anga", "enna", "epdi", "eppadi"
]

_COMMON_EN_EXCLUDE = {
    "the", "in", "to", "of", "and", "a", "is", "for", "with", "on",
    "free", "best", "new", "mp", "com", "hd", "mb", "songs", "song",
    "movie", "video", "download", "lyrics", "tamil", "chennai", "nagar",
    "india", "mp3", "actor", "actress", "python", "java", "system",
    "search", "buy", "shop", "online", "phone", "laptop", "near", "login"
}

_ROMANIZED_TAMIL_EXTENDED = False


def _extend_romanized_vocabulary():
    """Load top distinctive Romanized Tamil words from data/tamil_romanized_distinctive.txt or ta_rom."""
    global _ROMANIZED_TAMIL_EXTENDED
    if _ROMANIZED_TAMIL_EXTENDED:
        return
    _ROMANIZED_TAMIL_EXTENDED = True

    txt_path = os.path.join(os.path.dirname(__file__), "data", "tamil_romanized_distinctive.txt")
    if os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip().lower()
                    if w:
                        ROMANIZED_WORDS["tamil"].add(w)
            return
        except Exception:
            pass

    csv_path = os.path.join(os.path.dirname(__file__), "data", "tamil_romanized.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                loaded = 0
                for row in reader:
                    w = row.get("word", "").strip().lower()
                    if len(w) >= 3 and w not in _COMMON_EN_EXCLUDE:
                        ROMANIZED_WORDS["tamil"].add(w)
                        loaded += 1
                        if loaded >= 30000:
                            break
        except Exception:
            pass

    # Load Hindi Romanized
    hi_csv = os.path.join(os.path.dirname(__file__), "data", "hindi_romanized.csv")
    if os.path.exists(hi_csv):
        try:
            with open(hi_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                loaded = 0
                for row in reader:
                    w = row.get("word", "").strip().lower()
                    if len(w) >= 3 and w not in _COMMON_EN_EXCLUDE:
                        ROMANIZED_WORDS["hindi"].add(w)
                        loaded += 1
                        if loaded >= 15000:
                            break
        except Exception:
            pass

    # Load Telugu Romanized
    te_csv = os.path.join(os.path.dirname(__file__), "data", "telugu_romanized.csv")
    if os.path.exists(te_csv):
        try:
            with open(te_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                loaded = 0
                for row in reader:
                    w = row.get("word", "").strip().lower()
                    if len(w) >= 3 and w not in _COMMON_EN_EXCLUDE:
                        ROMANIZED_WORDS["telugu"].add(w)
                        loaded += 1
                        if loaded >= 15000:
                            break
        except Exception:
            pass


_extend_romanized_vocabulary()


def detect_romanized_language(query):
    """
    Detects likely Indian language from
    Romanized/English-script words, with typo tolerance
    for high-frequency Romanized vocabulary.

    Returns:
        language
        confidence
    """
    words = query.lower().split()

    scores = {
        "tamil": 0,
        "telugu": 0,
        "hindi": 0
    }

    for word in words:
        matched = False
        for language in ROMANIZED_WORDS:
            if word in ROMANIZED_WORDS[language]:
                scores[language] += 1
                matched = True

        # Typo tolerance: if not an exact match, check 1-edit distance
        # to high-frequency Romanized Tamil words (e.g. vemum -> venum)
        if not matched and len(word) >= 3:
            for hf in HIGH_FREQ_TAMIL:
                if damerau_levenshtein(word, hf, max_dist=1) <= 1:
                    scores["tamil"] += 1
                    break

    best_language = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_language]

    if best_score == 0:
        return {
            "language": "unknown",
            "confidence": 0.0
        }

    total_matches = sum(scores.values())

    confidence = best_score / total_matches

    return {
        "language": best_language,
        "confidence": round(confidence, 3)
    }