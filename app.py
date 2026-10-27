import csv
import collections
from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from trie import Trie
from data_loader import load_dictionary_and_trie
from multilingual_search import multilingual_search
from ml_ranker import load_model
from feedback_learner import get_feedback_learner
from temporal_ranker import get_temporal_ranker
from personalized_ranker import get_personalized_ranker


app = Flask(__name__)


# ---------------------------------
# Load dictionaries & ML model once
# ---------------------------------

trie_en, dict_en = load_dictionary_and_trie("english")
trie_ta, dict_ta = load_dictionary_and_trie("tamil")
trie_rom, dict_rom = load_dictionary_and_trie("tamil_romanized")

language_data = {
    "english": (trie_en, dict_en),
    "tamil": (trie_ta, dict_ta),
    "tamil_romanized": (trie_rom, dict_rom),
    "spanish": load_dictionary_and_trie("spanish"),
    "german": load_dictionary_and_trie("german"),
    "french": load_dictionary_and_trie("french"),
    "italian": load_dictionary_and_trie("italian"),
    "portuguese": load_dictionary_and_trie("portuguese"),
    "dutch": load_dictionary_and_trie("dutch"),
}

ml_model = load_model("models/ml_ranker.pkl")
learner = get_feedback_learner()
temporal_ranker = get_temporal_ranker()
personalized_ranker = get_personalized_ranker()


# ---------------------------------
# Web page
# ---------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------
# Suggestions API (Features 1–15)
# ---------------------------------

@app.route("/get_suggestions", methods=["POST"])
def get_suggestions():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    user_id = str(data.get("user_id", "default")).strip().lower()
    target_language = data.get("language")

    if not query:
        return jsonify({
            "autocomplete": [],
            "corrections": [],
            "ranked_results": [],
            "ml_reranked_results": [],
            "ml_reranking_used": False,
            "dynamic_weights": {},
            "weight_adaptation_reason": [],
            "dynamic_weighting_used": False,
            "intent": "unknown",
            "language": "unknown",
            "temporal_detected": False,
            "temporal_type": None,
            "temporal_signals": [],
            "temporal_confidence": 0.0,
            "temporal_results": [],
            "personalized_results": [],
            "personalized_candidates": [],
            "top_k_suggestions": [],
            "user_id": user_id,
        })

    # Record search query arrival for trending velocity tracking (Feature 14)
    temporal_ranker.record_query(query)

    user_freq = learner.get_user_frequencies()
    res = multilingual_search(
        query,
        language_data,
        user_freq,
        autocomplete_k=5,
        correction_k=5,
        ml_model=ml_model,
        user_id=user_id,
        top_k=5,
        target_language=target_language
    )


    return jsonify({
        "autocomplete": res.get("autocomplete", []),
        "corrections": res.get("corrections", []),
        "ranked_results": res.get("ranked_results", []),
        "ml_reranked_results": res.get("ml_reranked_results", []),
        "ml_reranking_used": res.get("ml_reranking_used", False),
        "dynamic_weights": res.get("dynamic_weights", {}),
        "weight_adaptation_reason": res.get("weight_adaptation_reason", []),
        "dynamic_weighting_used": res.get("dynamic_weighting_used", False),
        "confidence_score": res.get("confidence_score", 0.0),
        "confidence_level": res.get("confidence_level", "low"),
        "is_ambiguous": res.get("is_ambiguous", False),
        "ambiguity_score": res.get("ambiguity_score", 0.0),
        "ambiguity_type": res.get("ambiguity_type", []),
        "confidence_reasons": res.get("confidence_reasons", []),
        "temporal_detected": res.get("temporal_detected", False),
        "temporal_type": res.get("temporal_type"),
        "temporal_signals": res.get("temporal_signals", []),
        "temporal_confidence": res.get("temporal_confidence", 0.0),
        "temporal_results": res.get("temporal_results", []),
        "temporal_candidates": res.get("temporal_candidates", []),
        "personalized_results": res.get("personalized_results", []),
        "personalized_candidates": res.get("personalized_candidates", []),
        "top_k_suggestions": res.get("top_k_suggestions", []),
        "user_id": user_id,
        "intent": res.get("intent", "unknown"),
        "language": res.get("language", "unknown")
    })


# ---------------------------------
# Feedback API (Features 12, 14, 15)
# ---------------------------------

@app.route("/record_feedback", methods=["POST"])
def record_feedback():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    chosen_word = data.get("chosen_word") or data.get("word")
    rejected_words = data.get("rejected_words", [])
    interaction_type = data.get("interaction_type", "select")
    user_id = str(data.get("user_id", "default")).strip().lower()
    intent = data.get("intent")
    language = data.get("language")

    # Feature 12: global user preference learning
    learner.record_interaction(
        query=query,
        chosen_word=chosen_word,
        rejected_words=rejected_words,
        interaction_type=interaction_type
    )

    # Feature 14: temporal / trending recording
    if chosen_word:
        temporal_ranker.record_selection(chosen_word, query=query)

    # Feature 15: per-user personalized interaction recording
    personalized_ranker.record_user_interaction(
        user_id=user_id,
        query=query,
        chosen_word=chosen_word,
        rejected_words=rejected_words,
        intent=intent,
        language=language
    )

    return jsonify({
        "status": "success",
        "learned_word": chosen_word,
        "user_id": user_id,
        "summary": learner.get_feedback_summary(chosen_word)
    })


# ---------------------------------
# Feature 14: Trending API
# ---------------------------------

@app.route("/get_trending", methods=["GET"])
def get_trending():
    trending = temporal_ranker.get_top_trending(limit=8)
    return jsonify({"trending": trending})


# ---------------------------------
# Feature 15: Personalization Profile APIs
# ---------------------------------

@app.route("/get_users", methods=["GET"])
def get_users():
    users = personalized_ranker.list_users()
    return jsonify({"users": users})


@app.route("/get_user_profile", methods=["POST"])
def get_user_profile():
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id", "default")).strip().lower()
    profile = personalized_ranker.get_user_profile(user_id)
    return jsonify({"profile": profile})


@app.route("/reset_profile", methods=["POST"])
def reset_profile():
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id", "default")).strip().lower()
    personalized_ranker.reset_user_profile(user_id)
    return jsonify({"status": "success", "user_id": user_id})


if __name__ == "__main__":
    app.run(debug=True)