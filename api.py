"""FastAPI REST Service for Multilingual Search Intelligence Lab.

Exposes RESTful endpoints conforming to the research specification:
  - GET  /api/v1/health       - Health check & system metadata
  - POST /api/v1/suggest      - End-to-end predictive search (Features 1–15)
  - POST /api/v1/correct      - Damerau–Levenshtein spelling corrections
  - GET  /api/v1/autocomplete - Prefix Trie autocomplete candidates
  - POST /api/v1/feedback     - Online feedback & personalization recording
  - GET  /api/v1/metrics      - Offline robustness, latency & memory metrics
  - POST /api/v1/compare      - Algorithm comparison (Trie, Norvig, SymSpell, Hybrid, ML, Proposed)
  - POST /api/v1/explain      - 7-Signal scoring breakdown & dynamic weights
  - GET  /api/v1/trending     - Real-time trending query spikes
  - GET  /api/v1/users        - Active user profile list
  - GET  /api/v1/ablation     - Component ablation study benchmarks
"""

import collections
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Core search engine modules
from data_loader import load_dictionary_and_trie
from ml_ranker import load_model, rerank_candidates
from multilingual_search import multilingual_search
from unified_language_detector import analyze_language
from intent_detector import detect_intent
from corrector import correct_query, damerau_levenshtein
from suggestor import autocomplete
from feedback_learner import get_feedback_learner
from temporal_ranker import get_temporal_ranker
from personalized_ranker import get_personalized_ranker
from dynamic_weights import get_dynamic_weights, DEFAULT_HYBRID_WEIGHTS
from ranker import rank_multilingual_candidates

DATA_DIR = "data"
ROBUSTNESS_METRICS_FILE = os.path.join(DATA_DIR, "robustness_latency_memory_results.json")
ABLATION_METRICS_FILE = os.path.join(DATA_DIR, "ablation_and_comparison_results.json")

# ============================================================================
# PYDANTIC DATA MODELS
# ============================================================================

class SuggestRequest(BaseModel):
    query: str = Field(..., description="Input query string to search / autocorrect")
    language: Optional[str] = Field(None, description="Optional forced language ('english' or 'tamil')")
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of suggestions to return")
    user_id: str = Field("default", description="User profile ID for personalized ranking")
    context: Optional[List[str]] = Field(None, description="Session context query history")


class CandidateItem(BaseModel):
    rank: int
    word: str
    final_score: float
    confidence_score: Optional[float] = None
    is_personalized: bool = False
    is_bursting: bool = False
    personalization_reason: Optional[str] = None
    temporal_reason: Optional[str] = None
    source: Optional[str] = None


class SuggestResponse(BaseModel):
    original_query: str
    detected_language: str
    language_type: str
    language_confidence: float
    detected_intent: str
    error_classification: Optional[str] = None
    autocomplete_candidates: List[str]
    spelling_corrections: List[str]
    ranked_suggestions: List[str]
    candidate_details: List[CandidateItem]
    confidence_level: str
    confidence_score: float
    is_ambiguous: bool
    ambiguity_reasons: List[str]
    temporal_detected: bool
    temporal_type: Optional[str] = None
    processing_latency_ms: float
    user_id: str


class CorrectRequest(BaseModel):
    query: str = Field(..., description="Query to spellcheck")
    language: str = Field("english", description="Target language ('english' or 'tamil')")
    threshold: int = Field(2, ge=1, le=3, description="Max edit distance threshold")
    max_results: int = Field(5, ge=1, le=20, description="Max corrections to return")


class CorrectResponse(BaseModel):
    query: str
    language: str
    is_exact_match: bool
    corrections: List[str]
    latency_ms: float


class AutocompleteResponse(BaseModel):
    prefix: str
    language: str
    candidates: List[str]
    latency_ms: float


class FeedbackRequest(BaseModel):
    query: str = Field(..., description="Original input query")
    chosen_word: str = Field(..., description="Selected candidate word")
    rejected_words: List[str] = Field(default_factory=list, description="Suggestions displayed but rejected")
    interaction_type: str = Field("select", description="Type of interaction: 'select' or 'accept'")
    user_id: str = Field("default", description="User profile ID")
    intent: Optional[str] = Field(None, description="Query intent")
    language: Optional[str] = Field(None, description="Query language")


class FeedbackResponse(BaseModel):
    status: str
    learned_word: str
    user_id: str
    summary: Dict[str, Any]


class CompareRequest(BaseModel):
    query: str = Field(..., description="Query to compare across ranking algorithms")
    top_k: int = Field(5, ge=1, le=10)
    methods: Optional[List[str]] = Field(
        default=["trie", "corrector", "hybrid", "ml", "proposed"],
        description="Algorithms to benchmark"
    )


class AlgorithmResult(BaseModel):
    algorithm_name: str
    suggestions: List[str]
    latency_ms: float
    top_score: Optional[float] = None
    notes: str


class CompareResponse(BaseModel):
    query: str
    results: List[AlgorithmResult]


class ExplainRequest(BaseModel):
    query: str = Field(..., description="Query to inspect ranking breakdown for")
    user_id: str = Field("default", description="User profile ID")


# ============================================================================
# FASTAPI APP & STATE INITIALIZATION
# ============================================================================

from contextlib import asynccontextmanager

# Shared memory application state
state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    print("[API Startup] Loading language resources, tries, and ML models...")
    trie_en, dict_en = load_dictionary_and_trie("english")
    trie_ta, dict_ta = load_dictionary_and_trie("tamil")
    trie_rom, dict_rom = load_dictionary_and_trie("tamil_romanized")
    ml_model = load_model("models/ml_ranker.pkl")

    state["language_data"] = {
        "english": (trie_en, dict_en),
        "tamil": (trie_ta, dict_ta),
        "tamil_romanized": (trie_rom, dict_rom),
        "hindi": load_dictionary_and_trie("hindi"),
        "hindi_romanized": load_dictionary_and_trie("hindi_romanized"),
        "telugu": load_dictionary_and_trie("telugu"),
        "telugu_romanized": load_dictionary_and_trie("telugu_romanized"),
        "spanish": load_dictionary_and_trie("spanish"),
        "german": load_dictionary_and_trie("german"),
        "french": load_dictionary_and_trie("french"),
        "italian": load_dictionary_and_trie("italian"),
        "portuguese": load_dictionary_and_trie("portuguese"),
        "dutch": load_dictionary_and_trie("dutch"),
    }
    state["ml_model"] = ml_model
    state["learner"] = get_feedback_learner()
    state["temporal_ranker"] = get_temporal_ranker()
    state["personalized_ranker"] = get_personalized_ranker()
    print("[API Startup] System initialized successfully with all 13 language variants!")
    yield


app = FastAPI(
    title="Multilingual Search Intelligence Lab API",
    description="Research-grade REST API for Multilingual Predictive Search, Hybrid Ranking, and Personalization.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/api/v1/health", tags=["System"])
def get_health() -> Dict[str, Any]:
    """Return service status and basic version information."""
    lang_data = state.get("language_data", {})
    trie_en, _ = lang_data.get("english", (None, None))
    trie_ta, _ = lang_data.get("tamil", (None, None))

    return {
        "status": "healthy",
        "service": "Multilingual Search Intelligence Lab",
        "version": "1.0.0",
        "models_loaded": {
            "english_trie": trie_en is not None,
            "tamil_trie": trie_ta is not None,
            "hindi_trie": lang_data.get("hindi", (None, None))[0] is not None,
            "hindi_romanized_trie": lang_data.get("hindi_romanized", (None, None))[0] is not None,
            "telugu_trie": lang_data.get("telugu", (None, None))[0] is not None,
            "telugu_romanized_trie": lang_data.get("telugu_romanized", (None, None))[0] is not None,
            "spanish_trie": lang_data.get("spanish", (None, None))[0] is not None,
            "german_trie": lang_data.get("german", (None, None))[0] is not None,
            "french_trie": lang_data.get("french", (None, None))[0] is not None,
            "italian_trie": lang_data.get("italian", (None, None))[0] is not None,
            "portuguese_trie": lang_data.get("portuguese", (None, None))[0] is not None,
            "dutch_trie": lang_data.get("dutch", (None, None))[0] is not None,
            "ml_ranker": state.get("ml_model") is not None,
            "feedback_learner": state.get("learner") is not None,
            "temporal_ranker": state.get("temporal_ranker") is not None,
            "personalized_ranker": state.get("personalized_ranker") is not None,
        },
        "supported_languages": [
            "english", "tamil", "tamil_romanized",
            "hindi", "hindi_romanized", "telugu", "telugu_romanized",
            "spanish", "german", "french", "italian", "portuguese", "dutch",
            "code_switched"
        ],
    }



@app.post("/api/v1/suggest", response_model=SuggestResponse, tags=["Search"])
def post_suggest(req: SuggestRequest) -> SuggestResponse:
    """Generate predictive search suggestions, spell checks, ML rankings, and personalized recommendations."""
    raw_query = req.query.strip()
    if not raw_query:
        return SuggestResponse(
            original_query="",
            detected_language="unknown",
            language_type="unknown",
            language_confidence=0.0,
            detected_intent="unknown",
            error_classification=None,
            autocomplete_candidates=[],
            spelling_corrections=[],
            ranked_suggestions=[],
            candidate_details=[],
            confidence_level="low",
            confidence_score=0.0,
            is_ambiguous=False,
            ambiguity_reasons=["Empty query supplied"],
            temporal_detected=False,
            temporal_type=None,
            processing_latency_ms=0.0,
            user_id=req.user_id,
        )

    t0 = time.perf_counter_ns()

    # Track query arrival for temporal/trending burst detector
    state["temporal_ranker"].record_query(raw_query)

    # Validate & resolve optional target language
    target_lang = None
    if req.language and str(req.language).lower() not in {"auto", "none"}:
        from data_loader import LANGUAGE_CODE_MAP, load_dictionary_and_trie
        raw_target = str(req.language).lower()
        target_lang = LANGUAGE_CODE_MAP.get(raw_target, raw_target)
        if target_lang not in state["language_data"]:
            t, d = load_dictionary_and_trie(target_lang)
            if t:
                state["language_data"][target_lang] = (t, d)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported language '{req.language}'. Available: {list(state['language_data'].keys())}"
                )

    # Invoke full 15-feature multilingual search pipeline
    user_freq = state["learner"].get_user_frequencies()
    res = multilingual_search(
        query=raw_query,
        language_data=state["language_data"],
        user_freq=user_freq,
        autocomplete_k=req.top_k,
        correction_k=req.top_k,
        context=req.context,
        ml_model=state["ml_model"],
        user_id=req.user_id,
        top_k=req.top_k,
        target_language=target_lang,
    )

    t1 = time.perf_counter_ns()
    elapsed_ms = round((t1 - t0) / 1_000_000.0, 3)

    # Format candidate details
    pers_cands = res.get("personalized_candidates", [])
    cand_details = [
        CandidateItem(
            rank=idx + 1,
            word=c.get("word", ""),
            final_score=round(float(c.get("final_personalized_score", c.get("final_score", 0.0))), 4),
            confidence_score=round(float(res.get("confidence_score", 0.0)), 4),
            is_personalized=bool(c.get("is_personalized", False)),
            is_bursting=bool(c.get("is_bursting", False)),
            personalization_reason=c.get("personalization_reason"),
            temporal_reason=c.get("temporal_reason"),
            source=c.get("source"),
        )
        for idx, c in enumerate(pers_cands[:req.top_k])
    ]

    # Detect error classification
    error_class = "none"
    if res.get("corrections"):
        error_class = "typo" if res.get("corrections")[0] != raw_query.lower() else "exact_match"
    elif not res.get("autocomplete"):
        error_class = "unrecognized_or_oov"

    return SuggestResponse(
        original_query=raw_query,
        detected_language=res.get("language", "unknown"),
        language_type=res.get("type", "native_script"),
        language_confidence=float(res.get("confidence", 0.0)),
        detected_intent=res.get("intent", "general_search"),
        error_classification=error_class,
        autocomplete_candidates=res.get("autocomplete", [])[:req.top_k],
        spelling_corrections=res.get("corrections", [])[:req.top_k],
        ranked_suggestions=res.get("top_k_suggestions", [])[:req.top_k],
        candidate_details=cand_details,
        confidence_level=res.get("confidence_level", "low"),
        confidence_score=round(float(res.get("confidence_score", 0.0)), 4),
        is_ambiguous=bool(res.get("is_ambiguous", False)),
        ambiguity_reasons=res.get("confidence_reasons", []),
        temporal_detected=bool(res.get("temporal_detected", False)),
        temporal_type=res.get("temporal_type"),
        processing_latency_ms=elapsed_ms,
        user_id=req.user_id,
    )


@app.post("/api/v1/correct", response_model=CorrectResponse, tags=["Spelling"])
def post_correct(req: CorrectRequest) -> CorrectResponse:
    """Execute spelling corrections using the 5-filter pruned Damerau-Levenshtein engine."""
    q = req.query.strip().lower()
    if not q:
        return CorrectResponse(query="", language=req.language, is_exact_match=False, corrections=[], latency_ms=0.0)

    t0 = time.perf_counter_ns()
    raw_lang = req.language.lower()
    from data_loader import LANGUAGE_CODE_MAP, load_dictionary_and_trie
    lang = LANGUAGE_CODE_MAP.get(raw_lang, raw_lang)
    if lang not in state["language_data"]:
        loaded_t, loaded_d = load_dictionary_and_trie(lang)
        if loaded_t:
            state["language_data"][lang] = (loaded_t, loaded_d)
        else:
            lang = "english"

    trie, dict_by_len = state["language_data"][lang]
    is_exact = False
    if trie:
        node = trie.search(q)
        if node and node.is_end:
            is_exact = True

    corrections = correct_query(
        dictionary_by_length=dict_by_len,
        trie=trie,
        query=q,
        threshold=req.threshold,
        max_results=req.max_results
    )
    t1 = time.perf_counter_ns()
    elapsed_ms = round((t1 - t0) / 1_000_000.0, 3)

    return CorrectResponse(
        query=req.query,
        language=lang,
        is_exact_match=is_exact,
        corrections=corrections,
        latency_ms=elapsed_ms
    )


@app.get("/api/v1/autocomplete", response_model=AutocompleteResponse, tags=["Autocomplete"])
def get_autocomplete(
    q: str = Query(..., description="Query prefix to complete"),
    language: str = Query("english", description="Target language ('english', 'tamil', 'spanish', 'german', 'french', 'italian', 'portuguese', 'dutch')"),
    k: int = Query(5, ge=1, le=20, description="Max completions")
) -> AutocompleteResponse:
    """Retrieve prefix completions directly from in-memory Trie."""
    prefix = q.strip().lower()
    if not prefix:
        return AutocompleteResponse(prefix="", language=language, candidates=[], latency_ms=0.0)

    t0 = time.perf_counter_ns()
    raw_lang = language.lower()
    from data_loader import LANGUAGE_CODE_MAP, load_dictionary_and_trie
    lang = LANGUAGE_CODE_MAP.get(raw_lang, raw_lang)
    if lang not in state["language_data"]:
        loaded_t, loaded_d = load_dictionary_and_trie(lang)
        if loaded_t:
            state["language_data"][lang] = (loaded_t, loaded_d)
        else:
            lang = "english"

    trie, _ = state["language_data"][lang]
    cands = autocomplete(trie, prefix, k=k) if trie else []
    t1 = time.perf_counter_ns()
    elapsed_ms = round((t1 - t0) / 1_000_000.0, 3)

    return AutocompleteResponse(
        prefix=prefix,
        language=lang,
        candidates=cands,
        latency_ms=elapsed_ms
    )



@app.post("/api/v1/feedback", response_model=FeedbackResponse, tags=["Feedback"])
def post_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Record explicit selection and implicit rejection feedback for online learning and personalization."""
    chosen = req.chosen_word.strip()
    if not chosen:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chosen_word cannot be empty")

    # 1. Feature 12: Global feedback learner
    learner = state["learner"]
    learner.record_interaction(
        query=req.query,
        chosen_word=chosen,
        rejected_words=req.rejected_words,
        interaction_type=req.interaction_type
    )

    # 2. Feature 14: Temporal query tracking
    state["temporal_ranker"].record_selection(chosen, query=req.query)

    # 3. Feature 15: Per-user personalized profile update
    pers_ranker = state["personalized_ranker"]
    pers_ranker.record_user_interaction(
        user_id=req.user_id,
        query=req.query,
        chosen_word=chosen,
        rejected_words=req.rejected_words,
        intent=req.intent,
        language=req.language
    )

    summary = learner.get_feedback_summary(chosen)
    return FeedbackResponse(
        status="success",
        learned_word=chosen,
        user_id=req.user_id,
        summary=summary
    )


@app.get("/api/v1/metrics", tags=["Evaluation"])
def get_metrics() -> Dict[str, Any]:
    """Return empirical evaluation benchmarks across robustness, latency, memory, and ablations."""
    metrics_data = {
        "status": "available",
        "timestamp": time.time(),
        "robustness_and_latency": None,
        "ablation_and_baselines": None,
    }

    if os.path.exists(ROBUSTNESS_METRICS_FILE):
        try:
            with open(ROBUSTNESS_METRICS_FILE, "r", encoding="utf-8") as f:
                metrics_data["robustness_and_latency"] = json.load(f)
        except Exception as e:
            metrics_data["robustness_and_latency_error"] = str(e)
    else:
        metrics_data["robustness_and_latency_error"] = "Run benchmark_evaluation.py to generate results."

    if os.path.exists(ABLATION_METRICS_FILE):
        try:
            with open(ABLATION_METRICS_FILE, "r", encoding="utf-8") as f:
                metrics_data["ablation_and_baselines"] = json.load(f)
        except Exception as e:
            metrics_data["ablation_and_baselines_error"] = str(e)
    else:
        metrics_data["ablation_and_baselines_error"] = "Run ablation_and_comparison.py to generate results."

    return metrics_data


@app.post("/api/v1/compare", response_model=CompareResponse, tags=["Comparison"])
def post_compare(req: CompareRequest) -> CompareResponse:
    """Compare suggestions, candidate orderings, and latency across baseline algorithms."""
    query = req.query.strip().lower()
    trie_en, dict_en = state["language_data"]["english"]
    results = []

    # 1. Standard Trie Autocomplete
    if "trie" in req.methods:
        t0 = time.perf_counter_ns()
        trie_cands = autocomplete(trie_en, query, k=req.top_k) if trie_en else []
        t1 = time.perf_counter_ns()
        results.append(AlgorithmResult(
            algorithm_name="Standard Prefix Trie",
            suggestions=trie_cands,
            latency_ms=round((t1 - t0) / 1_000_000.0, 3),
            notes="Exact prefix matches ordered by unigram corpus frequency. Zero typo tolerance."
        ))

    # 2. Pruned D-L Spelling Corrector
    if "corrector" in req.methods:
        t0 = time.perf_counter_ns()
        corr_cands = correct_query(dict_en, trie_en, query, threshold=2, max_results=req.top_k)
        t1 = time.perf_counter_ns()
        results.append(AlgorithmResult(
            algorithm_name="Pruned Damerau-Levenshtein",
            suggestions=corr_cands,
            latency_ms=round((t1 - t0) / 1_000_000.0, 3),
            notes="5 pruning filters (boundary anchor, length proximity, set overlap, row-min early break)."
        ))

    # 3. Fixed-Weight Hybrid Ranking
    if "hybrid" in req.methods:
        t0 = time.perf_counter_ns()
        res_hybrid = multilingual_search(
            query=query,
            language_data=state["language_data"],
            ml_model=None,  # No ML
            enable_dynamic_weights=False,  # Static default weights
            top_k=req.top_k
        )
        t1 = time.perf_counter_ns()
        results.append(AlgorithmResult(
            algorithm_name="Fixed-Weight Hybrid (F9)",
            suggestions=res_hybrid.get("ranked_results", [])[:req.top_k],
            latency_ms=round((t1 - t0) / 1_000_000.0, 3),
            top_score=round(float(res_hybrid.get("candidate_pool", [{}])[0].get("hybrid_score", 0.0)), 4) if res_hybrid.get("candidate_pool") else None,
            notes="7-signal scoring (lexical, edit, frequency, semantics, context, intent, feedback) with static weights."
        ))

    # 4. Machine Learning Reranker (Feature 10)
    if "ml" in req.methods:
        t0 = time.perf_counter_ns()
        res_ml = multilingual_search(
            query=query,
            language_data=state["language_data"],
            ml_model=state["ml_model"],
            enable_dynamic_weights=False,
            top_k=req.top_k
        )
        t1 = time.perf_counter_ns()
        results.append(AlgorithmResult(
            algorithm_name="Hybrid + ML Reranker (F10)",
            suggestions=res_ml.get("ml_reranked_results", [])[:req.top_k],
            latency_ms=round((t1 - t0) / 1_000_000.0, 3),
            top_score=round(float(res_ml.get("candidate_pool", [{}])[0].get("final_score", 0.0)), 4) if res_ml.get("candidate_pool") else None,
            notes="28 extracted features reranked by 35-tree Random Forest Regressor."
        ))

    # 5. Full Proposed System (Features 1–15)
    if "proposed" in req.methods:
        t0 = time.perf_counter_ns()
        res_prop = multilingual_search(
            query=query,
            language_data=state["language_data"],
            ml_model=state["ml_model"],
            enable_dynamic_weights=True,
            top_k=req.top_k
        )
        t1 = time.perf_counter_ns()
        results.append(AlgorithmResult(
            algorithm_name="Proposed Adaptive System (Full Pipeline)",
            suggestions=res_prop.get("top_k_suggestions", [])[:req.top_k],
            latency_ms=round((t1 - t0) / 1_000_000.0, 3),
            top_score=round(float(res_prop.get("personalized_candidates", [{}])[0].get("final_personalized_score", 0.0)), 4) if res_prop.get("personalized_candidates") else None,
            notes="End-to-end: Dynamic weights + ML reranking + Temporal burst adaptation + Personalization."
        ))

    return CompareResponse(query=req.query, results=results)


@app.post("/api/v1/explain", tags=["Explainability"])
def post_explain(req: ExplainRequest) -> Dict[str, Any]:
    """Expose granular 7-signal score components, dynamic weights reallocation, and blend factors."""
    query = req.query.strip()
    if not query:
        return {"query": "", "signals": [], "dynamic_weights": {}}

    res = multilingual_search(
        query=query,
        language_data=state["language_data"],
        ml_model=state["ml_model"],
        user_id=req.user_id,
        top_k=5
    )

    pers_cands = res.get("personalized_candidates", [])
    explained_cands = []

    for c in pers_cands[:5]:
        explained_cands.append({
            "word": c.get("word"),
            "source": c.get("source"),
            "edit_distance": c.get("edit_distance", 0),
            "signals": {
                "lexical_prefix": round(float(c.get("lexical_score", c.get("prefix_score", 0.0))), 4),
                "edit_similarity": round(float(c.get("edit_score", 0.0)), 4),
                "log_frequency": round(float(c.get("frequency_score", 0.0)), 4),
                "semantic_similarity": round(float(c.get("semantic_score", 0.0)), 4),
                "session_context": round(float(c.get("context_score", 0.0)), 4),
                "intent_compatibility": round(float(c.get("intent_score", 0.5)), 4),
                "user_feedback": round(float(c.get("user_score", 0.0)), 4),
            },
            "hybrid_score": round(float(c.get("hybrid_score", 0.0)), 4),
            "ml_score": round(float(c.get("ml_score", 0.0)), 4) if "ml_score" in c else None,
            "temporal_boost": round(float(c.get("temporal_boost", 0.0)), 4),
            "personalization_score": round(float(c.get("personalization_score", 0.0)), 4),
            "personalization_boost": round(float(c.get("personalization_boost", 0.0)), 4),
            "final_score": round(float(c.get("final_personalized_score", c.get("final_score", 0.0))), 4),
            "reasons": {
                "dynamic_weights": res.get("weight_adaptation_reason", []),
                "confidence": res.get("confidence_reasons", []),
                "temporal": c.get("temporal_reason"),
                "personalization": c.get("personalization_reason"),
            }
        })

    return {
        "query": query,
        "detected_language": res.get("language"),
        "detected_intent": res.get("intent"),
        "confidence_level": res.get("confidence_level"),
        "confidence_score": res.get("confidence_score"),
        "dynamic_weights_used": res.get("dynamic_weights"),
        "default_weights": DEFAULT_HYBRID_WEIGHTS,
        "candidates": explained_cands,
    }


@app.get("/api/v1/trending", tags=["Temporal"])
def get_trending() -> Dict[str, Any]:
    """Retrieve active trending query spikes discovered across users."""
    tr = state.get("temporal_ranker")
    trending = tr.get_top_trending(limit=10) if tr else []
    return {"trending": trending}


@app.get("/api/v1/users", tags=["Personalization"])
def get_users() -> Dict[str, Any]:
    """List all registered user profiles."""
    pr = state.get("personalized_ranker")
    users = pr.list_users() if pr else []
    return {"users": users}


@app.get("/api/v1/ablation", tags=["Evaluation"])
def get_ablation() -> Dict[str, Any]:
    """Retrieve component ablation matrix."""
    if os.path.exists(ABLATION_METRICS_FILE):
        try:
            with open(ABLATION_METRICS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {"status": "success", "ablation": data.get("ablation", {}), "comparison": data.get("comparison", {})}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "not_evaluated", "message": "Run ablation_and_comparison.py first."}
