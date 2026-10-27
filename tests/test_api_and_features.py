import os
import sys

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient

import api

@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


# ============================================================================
# 1. SYSTEM HEALTH & METADATA
# ============================================================================

def test_health_endpoint(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "english_trie" in data["models_loaded"]
    assert data["models_loaded"]["english_trie"] is True
    assert data["models_loaded"]["ml_ranker"] is True


# ============================================================================
# 2. SUGGESTION PIPELINE (Features 1–15)
# ============================================================================

def test_suggest_clean_english(client):
    res = client.post("/api/v1/suggest", json={"query": "python", "top_k": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["detected_language"] == "english"
    assert "python" in data["ranked_suggestions"]
    assert data["processing_latency_ms"] > 0.0


def test_suggest_misspelling_correction(client):
    res = client.post("/api/v1/suggest", json={"query": "pythno", "top_k": 5})
    assert res.status_code == 200
    data = res.json()
    assert "python" in data["ranked_suggestions"]
    assert len(data["candidate_details"]) > 0


def test_suggest_prefix_autocomplete(client):
    res = client.post("/api/v1/suggest", json={"query": "pyth", "top_k": 5})
    assert res.status_code == 200
    data = res.json()
    assert "python" in data["ranked_suggestions"]
    assert len(data["autocomplete_candidates"]) > 0


def test_suggest_tamil_native_script(client):
    res = client.post("/api/v1/suggest", json={"query": "வணக்கம்", "top_k": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["detected_language"] == "tamil"
    assert data["language_type"] == "native_script"
    assert len(data["ranked_suggestions"]) > 0


def test_suggest_romanized_tamil(client):
    res = client.post("/api/v1/suggest", json={"query": "venum", "top_k": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["detected_language"] == "tamil"
    assert "வேண்டும்" in data["ranked_suggestions"]


def test_suggest_romanized_tamil_briyani_vemum(client):
    res = client.post("/api/v1/suggest", json={"query": "briyani vemum", "top_k": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["detected_language"] == "tamil"
    assert data["ranked_suggestions"][0] == "biryani venum"
    assert "biryani venum" in data["ranked_suggestions"]


def test_suggest_romanized_tamil_vemum_correction(client):
    res = client.post("/api/v1/suggest", json={"query": "vemum", "top_k": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["detected_language"] == "tamil"
    assert "venum" in data["ranked_suggestions"]
    assert "வேண்டும்" in data["ranked_suggestions"]


def test_suggest_empty_query(client):
    res = client.post("/api/v1/suggest", json={"query": "", "top_k": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["ranked_suggestions"] == []
    assert data["confidence_level"] == "low"


def test_suggest_whitespace_only(client):
    res = client.post("/api/v1/suggest", json={"query": "    ", "top_k": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["ranked_suggestions"] == []


def test_suggest_extremely_long_query(client):
    long_q = "system " * 30
    res = client.post("/api/v1/suggest", json={"query": long_q, "top_k": 5})
    assert res.status_code == 200
    assert "ranked_suggestions" in res.json()


# ============================================================================
# 3. SPELLCHECKING & AUTOCOMPLETE
# ============================================================================

def test_correct_endpoint(client):
    res = client.post("/api/v1/correct", json={"query": "pythno", "threshold": 2, "max_results": 5})
    assert res.status_code == 200
    data = res.json()
    assert "python" in data["corrections"]
    assert data["is_exact_match"] is False


def test_autocomplete_endpoint(client):
    res = client.get("/api/v1/autocomplete?q=py&k=5")
    assert res.status_code == 200
    data = res.json()
    assert len(data["candidates"]) > 0
    assert "python" in data["candidates"]


# ============================================================================
# 4. ONLINE FEEDBACK & PERSONALIZATION
# ============================================================================

def test_feedback_endpoint(client):
    res = client.post("/api/v1/feedback", json={
        "query": "pythn",
        "chosen_word": "python",
        "rejected_words": ["pylon"],
        "user_id": "pytest_user",
        "interaction_type": "select"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["learned_word"] == "python"
    assert data["user_id"] == "pytest_user"


# ============================================================================
# 5. METRICS, COMPARISON & EXPLAINABILITY
# ============================================================================

def test_metrics_endpoint(client):
    res = client.get("/api/v1/metrics")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "available"
    assert data["robustness_and_latency"] is not None
    assert data["ablation_and_baselines"] is not None


def test_compare_endpoint(client):
    res = client.post("/api/v1/compare", json={"query": "pythn", "top_k": 5})
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) >= 4
    alg_names = [r["algorithm_name"] for r in data["results"]]
    assert any("Prefix Trie" in a for a in alg_names)
    assert any("Damerau-Levenshtein" in a for a in alg_names)
    assert any("Proposed" in a for a in alg_names)


def test_explain_endpoint(client):
    res = client.post("/api/v1/explain", json={"query": "buy python", "user_id": "alice"})
    assert res.status_code == 200
    data = res.json()
    assert "dynamic_weights_used" in data
    assert "default_weights" in data
    assert len(data["candidates"]) > 0
    first_c = data["candidates"][0]
    assert "signals" in first_c
    assert "lexical_prefix" in first_c["signals"]
    assert "semantic_similarity" in first_c["signals"]


def test_trending_and_users(client):
    r_tr = client.get("/api/v1/trending")
    assert r_tr.status_code == 200
    assert "trending" in r_tr.json()

    r_usr = client.get("/api/v1/users")
    assert r_usr.status_code == 200
    assert "users" in r_usr.json()
