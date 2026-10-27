import os
import sys

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from data_loader import load_dictionary_and_trie, SUPPORTED_LANGUAGES
from language_detector import detect_language
from unified_language_detector import analyze_language
from romanized_lexicon import lookup_romanized_candidates
from multilingual_search import multilingual_search

def test_datasets_exist_and_load():
    """Verify all 4 new dataset indices load properly into Trie structures."""
    assert "hindi" in SUPPORTED_LANGUAGES
    assert "hindi_romanized" in SUPPORTED_LANGUAGES
    assert "telugu" in SUPPORTED_LANGUAGES
    assert "telugu_romanized" in SUPPORTED_LANGUAGES

    trie_hi, dict_hi = load_dictionary_and_trie("hindi")
    assert trie_hi is not None
    assert len(dict_hi) > 0

    trie_hi_rom, dict_hi_rom = load_dictionary_and_trie("hindi_romanized")
    assert trie_hi_rom is not None
    assert len(dict_hi_rom) > 0

    trie_te, dict_te = load_dictionary_and_trie("telugu")
    assert trie_te is not None
    assert len(dict_te) > 0

    trie_te_rom, dict_te_rom = load_dictionary_and_trie("telugu_romanized")
    assert trie_te_rom is not None
    assert len(dict_te_rom) > 0

def test_script_detection():
    """Verify Unicode-level script detector identifies Devanagari and Telugu scripts."""
    assert detect_language("नमस्ते") == "hindi"
    assert detect_language("समाचार") == "hindi"
    assert detect_language("నమస్కారం") == "telugu"
    assert detect_language("తెలుగు") == "telugu"
    assert detect_language("வணக்கம்") == "tamil"
    assert detect_language("hello") == "english"

def test_unified_language_analysis():
    """Verify unified detector correctly tags native script vs romanized."""
    # Hindi Native
    hi_native = analyze_language("नमस्ते भारत")
    assert hi_native["language"] == "hindi"
    assert hi_native["type"] == "native_script"

    # Telugu Native
    te_native = analyze_language("నమస్కారం వార్తలు")
    assert te_native["language"] == "telugu"
    assert te_native["type"] == "native_script"

    # Hindi Romanized
    hi_rom = analyze_language("kya haal hai bhai")
    assert hi_rom["language"] == "hindi"
    assert hi_rom["type"] == "romanized"

    # Telugu Romanized
    te_rom = analyze_language("meeru ela unnaru")
    assert te_rom["language"] == "telugu"
    assert te_rom["type"] == "romanized"

def test_romanized_lexicon_bridge():
    """Verify curated Romanized lexicons bridge to native script equivalents."""
    hi_res = lookup_romanized_candidates("hindi", "namaste")
    assert "नमस्ते" in hi_res

    te_res = lookup_romanized_candidates("telugu", "namaskaram")
    assert "నమస్కారం" in te_res

@pytest.fixture(scope="module")
def lang_data():
    return {
        "hindi": load_dictionary_and_trie("hindi"),
        "hindi_romanized": load_dictionary_and_trie("hindi_romanized"),
        "telugu": load_dictionary_and_trie("telugu"),
        "telugu_romanized": load_dictionary_and_trie("telugu_romanized"),
    }

def test_multilingual_search_native_hindi(lang_data):
    """Verify multilingual search generates valid autocomplete suggestions for Hindi."""
    res = multilingual_search("नम", lang_data, target_language="hindi")
    assert res["language"] == "hindi"
    assert len(res["top_k_suggestions"]) > 0

def test_multilingual_search_native_telugu(lang_data):
    """Verify multilingual search generates valid autocomplete suggestions for Telugu."""
    res = multilingual_search("నమ", lang_data, target_language="telugu")
    assert res["language"] == "telugu"
    assert len(res["top_k_suggestions"]) > 0

def test_multilingual_search_romanized_hindi_bridge(lang_data):
    """Verify romanized Hindi search bridges to native script suggestions."""
    res = multilingual_search("namaste", lang_data, target_language="hindi")
    assert res["language"] == "hindi"
    assert any("नमस्ते" in s for s in res["top_k_suggestions"])

def test_multilingual_search_romanized_telugu_bridge(lang_data):
    """Verify romanized Telugu search bridges to native script suggestions."""
    res = multilingual_search("namaskaram", lang_data, target_language="telugu")
    assert res["language"] == "telugu"
    assert any("నమస్కారం" in s for s in res["top_k_suggestions"])
