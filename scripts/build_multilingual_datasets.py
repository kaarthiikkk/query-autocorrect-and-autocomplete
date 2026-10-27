"""Reproducible dataset generator for multilingual vocabularies using wordfreq.

Generates frequency-ranked vocabulary files for:
  - Spanish (es) -> data/spanish.csv
  - German (de)  -> data/german.csv
  - French (fr)  -> data/french.csv
  - Italian (it) -> data/italian.csv
  - Portuguese (pt) -> data/portuguese.csv
  - Dutch (nl)   -> data/dutch.csv

Records source provenance, licensing, and frequency mapping in
data/multilingual_datasets_manifest.json.
"""

import csv
import datetime
import json
import os
import sys
import unicodedata

try:
    import wordfreq
except ImportError:
    print("ERROR: wordfreq package not found. Please install it with 'pip install wordfreq'.")
    sys.exit(1)


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

LANGUAGES = [
    {
        "code": "es",
        "name": "spanish",
        "label": "Spanish",
        "target_count": 20000,
    },
    {
        "code": "de",
        "name": "german",
        "label": "German",
        "target_count": 20000,
    },
    {
        "code": "fr",
        "name": "french",
        "label": "French",
        "target_count": 20000,
    },
    {
        "code": "it",
        "name": "italian",
        "label": "Italian",
        "target_count": 20000,
    },
    {
        "code": "pt",
        "name": "portuguese",
        "label": "Portuguese",
        "target_count": 20000,
    },
    {
        "code": "nl",
        "name": "dutch",
        "label": "Dutch",
        "target_count": 20000,
    },
]


def clean_word(raw_word):
    """Normalize word to NFC lowercase, preserving accents while removing noise."""
    if not raw_word or not isinstance(raw_word, str):
        return None
    # NFC normalisation ensures accented letters like é, ñ, ü are composite single glyphs
    normalized = unicodedata.normalize("NFC", raw_word.strip().lower())
    if not normalized:
        return None
    # Must have at least one alphabetic character
    if not any(c.isalpha() for c in normalized):
        return None
    # Exclude purely numeric, punctuation, or corrupted tokens
    if any(c in "\x00\r\n\t" for c in normalized):
        return None
    return normalized


def build_language_vocabulary(lang_cfg):
    """Build a cleaned, frequency-ranked vocabulary dataset for a language."""
    code = lang_cfg["code"]
    name = lang_cfg["name"]
    target_count = lang_cfg["target_count"]

    print(f"\n[Dataset Builder] Extracting vocabulary for {lang_cfg['label']} ({code})...")

    # Request candidates from wordfreq to allow for filtering
    raw_list = wordfreq.top_n_list(code, target_count + 15000, wordlist="best")
    if not raw_list:
        raw_list = wordfreq.top_n_list(code, target_count + 15000, wordlist="default")

    def strip_accents(s):
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

    # Pass 1: collect unique cleaned candidates and their frequencies
    candidates = []
    seen = set()
    accented_canonical = {}

    for raw in raw_list:
        w = clean_word(raw)
        if not w or w in seen:
            continue
        seen.add(w)

        p = wordfreq.word_frequency(w, code, wordlist="best")
        zipf = wordfreq.zipf_frequency(w, code, wordlist="best")
        scaled_count = max(1, int(round(p * 1_000_000_000)))

        cand = {
            "word": w,
            "frequency": scaled_count,
            "zipf": round(zipf, 4),
            "prob": p,
            "stripped": strip_accents(w)
        }
        candidates.append(cand)
        # Track accented forms and their frequency
        if w != cand["stripped"]:
            if cand["stripped"] not in accented_canonical or scaled_count > accented_canonical[cand["stripped"]]["frequency"]:
                accented_canonical[cand["stripped"]] = cand

    # Pass 2: filter out unaccented internet typos (e.g. 'informacion' when canonical 'información' is 10x higher)
    cleaned_rows = []
    for cand in candidates:
        w = cand["word"]
        stripped = cand["stripped"]
        freq = cand["frequency"]

        # If w is purely ASCII unaccented, but an accented canonical counterpart has substantially higher frequency, skip w
        if w == stripped and stripped in accented_canonical:
            canon_freq = accented_canonical[stripped]["frequency"]
            if canon_freq >= freq * 2:
                continue

        cleaned_rows.append(cand)
        if len(cleaned_rows) >= target_count:
            break


    # Write persistent CSV: word,frequency
    csv_path = os.path.join(DATA_DIR, f"{name}.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        for item in cleaned_rows:
            writer.writerow([item["word"], item["frequency"]])

    print(f"[Dataset Builder] Wrote {len(cleaned_rows)} rows to {csv_path}")

    return {
        "language_code": code,
        "language_name": name,
        "file_path": f"data/{name}.csv",
        "row_count": len(cleaned_rows),
        "min_frequency": cleaned_rows[-1]["frequency"] if cleaned_rows else 0,
        "max_frequency": cleaned_rows[0]["frequency"] if cleaned_rows else 0,
        "min_zipf": cleaned_rows[-1]["zipf"] if cleaned_rows else 0.0,
        "max_zipf": cleaned_rows[0]["zipf"] if cleaned_rows else 0.0,
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    manifest = {
        "generator": "scripts/build_multilingual_datasets.py",
        "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "source": "wordfreq (Python library by Robyn Speer / Luminoso Technologies)",
        "source_url": "https://pypi.org/project/wordfreq/",
        "source_version": getattr(wordfreq, "__version__", "unknown"),
        "license": "Creative Commons Attribution-ShareAlike 4.0 / MIT",
        "frequency_representation": {
            "scale": "Integer frequency count derived as round(P * 1,000,000,000)",
            "rationale": "Matches existing English and Tamil integer frequency scaling in ranker.py log-frequency calculation without dominating hybrid weights."
        },
        "languages": {}
    }

    for lang_cfg in LANGUAGES:
        stats = build_language_vocabulary(lang_cfg)
        manifest["languages"][lang_cfg["name"]] = stats

    manifest_path = os.path.join(DATA_DIR, "multilingual_datasets_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[Dataset Builder] Successfully generated manifest: {manifest_path}")
    print("[Dataset Builder] All 6 language vocabularies ready!")


if __name__ == "__main__":
    main()
