import csv
import collections
import os

from trie import Trie


import unicodedata

DATA_DIR = "data"

LANGUAGE_CODE_MAP = {
    "en": "english",
    "ta": "tamil",
    "ta_rom": "tamil_romanized",
    "hi": "hindi",
    "hi_rom": "hindi_romanized",
    "te": "telugu",
    "te_rom": "telugu_romanized",
    "es": "spanish",
    "de": "german",
    "fr": "french",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
}

LANGUAGE_FILE_MAP = {
    "english": "english.csv",
    "tamil": "tamil_small.csv",
    "tamil_romanized": "tamil_romanized.csv",
    "ta_rom": "tamil_romanized.csv",
    "hindi": "hindi.csv",
    "hindi_romanized": "hindi_romanized.csv",
    "hi": "hindi.csv",
    "hi_rom": "hindi_romanized.csv",
    "telugu": "telugu.csv",
    "telugu_romanized": "telugu_romanized.csv",
    "te": "telugu.csv",
    "te_rom": "telugu_romanized.csv",
    "spanish": "spanish.csv",
    "german": "german.csv",
    "french": "french.csv",
    "italian": "italian.csv",
    "portuguese": "portuguese.csv",
    "dutch": "dutch.csv",
    "es": "spanish.csv",
    "de": "german.csv",
    "fr": "french.csv",
    "it": "italian.csv",
    "pt": "portuguese.csv",
    "nl": "dutch.csv",
}

SUPPORTED_LANGUAGES = [
    "english",
    "tamil",
    "tamil_romanized",
    "hindi",
    "hindi_romanized",
    "telugu",
    "telugu_romanized",
    "spanish",
    "german",
    "french",
    "italian",
    "portuguese",
    "dutch",
]

_LOADED_CACHE = {}


def load_dictionary_and_trie(language="english", reload=False):
    """
    Loads a language dictionary into:
    1. Trie
    2. Dictionary grouped by word length

    Supports both:
        word,count

    and:
        word,frequency
    """
    canonical = LANGUAGE_CODE_MAP.get(str(language).lower(), str(language).lower())
    if not reload and canonical in _LOADED_CACHE:
        return _LOADED_CACHE[canonical]

    filename = LANGUAGE_FILE_MAP.get(canonical, LANGUAGE_FILE_MAP.get(language, f"{canonical}.csv"))
    filepath = os.path.join(
        DATA_DIR,
        filename
    )

    print(
        f"\nLoading dictionary for "
        f"'{canonical}' from {filepath}..."
    )

    trie = Trie()
    dictionary_by_length = collections.defaultdict(list)

    try:
        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                # Read the word with NFC normalization
                raw_w = row.get("word", "")
                word = unicodedata.normalize("NFC", raw_w.strip().lower()) if raw_w else ""

                if not word:
                    continue

                # Support count or frequency
                try:
                    if row.get("count") is not None:
                        frequency = int(
                            row.get("count")
                        )

                    elif row.get("frequency") is not None:
                        frequency = int(
                            row.get("frequency")
                        )

                    else:
                        frequency = 1

                except (ValueError, TypeError):
                    frequency = 1

                # Insert into Trie
                trie.insert(
                    word,
                    frequency
                )

                # Group words by length
                dictionary_by_length[
                    len(word)
                ].append(word)

    except FileNotFoundError:

        print(
            f"ERROR: Dictionary file "
            f"'{filepath}' not found."
        )

        return None, None

    print(
        f"Dictionary for '{canonical}' "
        f"loaded successfully!"
    )

    total_words = sum(
        len(words)
        for words in dictionary_by_length.values()
    )

    print(
        f"Total words loaded: {total_words}"
    )

    _LOADED_CACHE[canonical] = (trie, dictionary_by_length)
    if language != canonical:
        _LOADED_CACHE[language] = (trie, dictionary_by_length)
    return trie, dictionary_by_length