"""Automated test suite verifying the 6 foreign languages (Spanish, German, French, Italian, Portuguese, Dutch).

Validates:
  1. Dataset integrity (20,000 unique entries, valid integer frequencies, accent preservation)
  2. Trie autocomplete and Damerau-Levenshtein spelling correction across all 6 languages
  3. Multilingual search routing and target_language override
  4. FastAPI REST endpoints (/api/v1/health, /api/v1/suggest, /api/v1/correct, /api/v1/autocomplete)
"""

import csv
import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader import load_dictionary_and_trie, SUPPORTED_LANGUAGES, LANGUAGE_FILE_MAP
from corrector import correct_query
from suggestor import autocomplete
from unified_language_detector import analyze_language, analyze_token
from multilingual_search import multilingual_search
import api


NEW_LANGUAGES = ["spanish", "german", "french", "italian", "portuguese", "dutch"]


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


# ============================================================================
# 1. DATASET INTEGRITY TESTS
# ============================================================================

@pytest.mark.parametrize("lang", NEW_LANGUAGES)
def test_dataset_file_structure(lang):
    filename = LANGUAGE_FILE_MAP[lang]
    filepath = os.path.join("data", filename)
    assert os.path.exists(filepath), f"Dataset file missing: {filepath}"

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["word", "frequency"], f"Invalid header in {filepath}: {header}"

        words = []
        frequencies = []
        for row in reader:
            assert len(row) == 2, f"Malformed row in {filepath}: {row}"
            w, freq = row[0].strip(), int(row[1])
            assert len(w) > 0, "Empty word found"
            assert freq > 0, f"Non-positive frequency found for {w}: {freq}"
            words.append(w)
            frequencies.append(freq)

        assert len(words) == 20000, f"Expected 20,000 unique entries for {lang}, got {len(words)}"
        assert len(set(words)) == len(words), f"Duplicates found in {filepath}"
        # Assert descending frequency order
        assert frequencies == sorted(frequencies, reverse=True), f"Entries in {filepath} not sorted by frequency"


def test_accent_preservation_in_datasets():
    # Spanish: más, información
    trie_es, dict_es = load_dictionary_and_trie("spanish")
    assert trie_es.search("información") is not None
    assert trie_es.search("más") is not None

    # German: für, über
    trie_de, dict_de = load_dictionary_and_trie("german")
    assert trie_de.search("für") is not None
    assert trie_de.search("über") is not None

    # French: été, même
    trie_fr, dict_fr = load_dictionary_and_trie("french")
    assert trie_fr.search("information") is not None

    # Italian: perché, così
    trie_it, dict_it = load_dictionary_and_trie("italian")
    assert trie_it.search("informazione") is not None

    # Portuguese: informação, você
    trie_pt, dict_pt = load_dictionary_and_trie("portuguese")
    assert trie_pt.search("informação") is not None

    # Dutch: informatie
    trie_nl, dict_nl = load_dictionary_and_trie("dutch")
    assert trie_nl.search("informatie") is not None


# ============================================================================
# 2. AUTOCOMPLETE & SPELLING CORRECTION TESTS
# ============================================================================

def test_spanish_autocomplete_and_correction():
    trie, d = load_dictionary_and_trie("spanish")
    ac = autocomplete(trie, "infor", k=5)
    assert any("inform" in w for w in ac)

    corr = correct_query(d, trie, "informacion", threshold=2, max_results=3)
    assert "información" in corr


def test_german_autocomplete_and_correction():
    trie, d = load_dictionary_and_trie("german")
    ac = autocomplete(trie, "informa", k=5)
    assert any("information" in w for w in ac)

    corr = correct_query(d, trie, "informazion", threshold=2, max_results=3)
    assert "information" in corr


def test_french_autocomplete_and_correction():
    trie, d = load_dictionary_and_trie("french")
    ac = autocomplete(trie, "inform", k=5)
    assert any("information" in w for w in ac)

    corr = correct_query(d, trie, "informasion", threshold=2, max_results=3)
    assert "information" in corr


def test_italian_autocomplete_and_correction():
    trie, d = load_dictionary_and_trie("italian")
    ac = autocomplete(trie, "infor", k=5)
    assert any("informazione" in w for w in ac)

    corr = correct_query(d, trie, "informazzione", threshold=2, max_results=3)
    assert "informazione" in corr


def test_portuguese_autocomplete_and_correction():
    trie, d = load_dictionary_and_trie("portuguese")
    ac = autocomplete(trie, "infor", k=5)
    assert any("informa" in w for w in ac)

    corr = correct_query(d, trie, "informasao", threshold=2, max_results=3)
    assert "informação" in corr


def test_dutch_autocomplete_and_correction():
    trie, d = load_dictionary_and_trie("dutch")
    ac = autocomplete(trie, "inform", k=5)
    assert any("informatie" in w for w in ac)

    corr = correct_query(d, trie, "informasie", threshold=2, max_results=3)
    assert "informatie" in corr


# ============================================================================
# 3. LANGUAGE DETECTION & ROUTING TESTS
# ============================================================================

def test_language_detection_and_override():
    # Spanish with accent
    det_es = analyze_language("información")
    assert det_es["language"] == "spanish"

    # German with umlaut
    det_de = analyze_language("über alles")
    assert det_de["language"] == "german"

    # Portuguese with tilde
    det_pt = analyze_language("informação")
    assert det_pt["language"] == "portuguese"

    # Dutch with ij
    det_nl = analyze_language("zijn")
    assert det_nl["language"] == "dutch"

    # Explicit target_language override
    override_fr = analyze_language("inform", target_language="french")
    assert override_fr["language"] == "french"

    override_it = analyze_language("inform", target_language="it")
    assert override_it["language"] == "italian"


# ============================================================================
# 4. FASTAPI ENDPOINT TESTS
# ============================================================================

def test_api_health_includes_all_languages(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    for l in NEW_LANGUAGES:
        assert l in data["supported_languages"]
        assert data["models_loaded"].get(f"{l}_trie") is True


def test_api_suggest_with_explicit_language(client):
    # Spanish query with explicit language
    res = client.post("/api/v1/suggest", json={"query": "informacion", "language": "spanish", "top_k": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["detected_language"] == "spanish"
    assert any("información" in s for s in data["ranked_suggestions"])

    # German query with explicit language
    res = client.post("/api/v1/suggest", json={"query": "informazion", "language": "german", "top_k": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["detected_language"] == "german"
    assert any("information" in s for s in data["ranked_suggestions"])


def test_api_correct_endpoint_multilingual(client):
    res = client.post("/api/v1/correct", json={"query": "informasion", "language": "french", "threshold": 2})
    assert res.status_code == 200
    data = res.json()
    assert data["language"] == "french"
    assert "information" in data["corrections"]


def test_api_autocomplete_endpoint_multilingual(client):
    res = client.get("/api/v1/autocomplete?q=inform&language=dutch&k=5")
    assert res.status_code == 200
    data = res.json()
    assert data["language"] == "dutch"
    assert "informatie" in data["candidates"]


def test_api_invalid_language_handling(client):
    res = client.post("/api/v1/suggest", json={"query": "hello", "language": "klingon_nonexistent"})
    assert res.status_code == 400
