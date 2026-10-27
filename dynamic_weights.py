"""Feature 11: Dynamic Ranking Weight Adaptation.

Dynamically computes normalized ranking weights for Feature 9's 7 signals:
- lexical
- edit
- frequency
- semantic
- context
- intent
- user

Adapts based on query characteristics:
1. Intent (Feature 8)
2. Language / Code-switched status (Features 1 & 2)
3. Typo vs prefix / autocomplete behavior
4. OOV / Entity protection (Feature 7)
5. Session context availability (Feature 3)
6. Query length and phrase structure


Entity & OOV Protection (Feature 7):
If a token is a protected brand name or rare entity, lexical (+0.15) and semantic (+0.05) are boosted while edit (-0.15) is penalized to prevent spell-checkers from distorting named entities.
Intent-Driven Adaptations (Feature 8):
Question Intent: Boosts semantic (+0.18) & intent (+0.10).
Transactional Intent: Boosts intent (+0.15) & semantic (+0.10).
Local Intent: Boosts intent (+0.15) & context (+0.10).
Navigational Intent: Boosts lexical (+0.15) & frequency (+0.10).
Typo vs. Autocomplete Behavior (Features 4, 5, 6):
Pure Typos: Boosts edit (+0.20) to focus on finding the correct spelling.
Prefix Autocomplete (≤6 letters): Boosts lexical (+0.18) and frequency (+0.05) to rapidly expand prefixes.
Code-Switched / Multilingual Queries (Features 1 & 2):
Boosts semantic (+0.10) and context (+0.05) when users mix English and Romanized Tamil script.
Session Context Availability (Feature 3):
Boosts context (+0.12) and user (+0.05) if active user history exists in recent.json.
Multi-Word Queries (≥3 tokens):
Boosts semantic (+0.08) to prioritize overall phrase meaning over individual word edit distances.
"""

from ranker import DEFAULT_HYBRID_WEIGHTS


def normalize_weights(weights):
    """Ensure all weights are non-negative and strictly sum to 1.0."""
    total = sum(max(0.01, w) for w in weights.values())
    if total <= 0:
        return dict(DEFAULT_HYBRID_WEIGHTS)

    normalized = {k: round(max(0.01, w) / total, 4) for k, w in weights.items()}
    diff = round(1.0 - sum(normalized.values()), 4)
    if diff != 0:
        max_key = max(normalized, key=normalized.get)
        normalized[max_key] = round(normalized[max_key] + diff, 4)
    return normalized


def get_dynamic_weights(
    query,
    intent="unknown",
    language="english",
    language_type="native_script",
    candidates=None,
    context=None,
    token_results=None,
    base_weights=None
):
    """Determines dynamic ranking weights based on query properties and candidates.

    Returns:
        dict with:
            - "weights": normalized {lexical, edit, frequency, semantic, context, intent, user}
            - "reasons": list of explainable adaptation reasons
            - "dynamic_weighting_used": True
            - "query_profile": dictionary of detected query signals
    """
    raw_query = str(query or "").strip()
    if not raw_query:
        return {
            "weights": dict(DEFAULT_HYBRID_WEIGHTS),
            "reasons": ["empty query: default weights applied"],
            "dynamic_weighting_used": False,
            "query_profile": {"is_empty": True}
        }

    tokens = raw_query.split()
    token_count = len(tokens)
    intent_clean = str(intent or "unknown").lower()
    lang_clean = str(language or "english").lower()
    lang_type_clean = str(language_type or "native_script").lower()

    # Base starting weights
    w = dict(base_weights if base_weights is not None else DEFAULT_HYBRID_WEIGHTS)
    reasons = []

    # 1. Candidate Pool Inspection
    candidates = candidates or []
    has_autocomplete = any(c.get("source") == "trie_autocomplete" for c in candidates)
    has_corrections = any(c.get("source") in {"damerau_levenshtein", "correction"} for c in candidates)
    has_romanized = any(c.get("source") == "romanized_lexicon" for c in candidates)

    # 2. Protection Inspection (Feature 7)
    should_protect = any(c.get("should_protect") for c in candidates)
    is_entity_like = any(c.get("is_entity_like") for c in candidates)
    if token_results:
        for t in token_results:
            prot = t.get("protection", {})
            if prot.get("should_protect"):
                should_protect = True
            if prot.get("is_entity_like"):
                is_entity_like = True

    # 3. Context Inspection (Feature 3)
    has_context = bool(context and (context.get("recent_words") or context.get("query_history")))

    # 4. Code-Switched Inspection (Features 1 & 2)
    is_code_switched = (
        lang_clean == "mixed"
        or lang_type_clean == "code_switched"
        or (token_results and any(t.get("type") == "romanized" for t in token_results))
        or has_romanized
    )

    query_profile = {
        "raw_query": raw_query,
        "token_count": token_count,
        "intent": intent_clean,
        "language": lang_clean,
        "language_type": lang_type_clean,
        "should_protect": should_protect,
        "is_entity_like": is_entity_like,
        "has_context": has_context,
        "is_code_switched": is_code_switched,
        "has_autocomplete": has_autocomplete,
        "has_corrections": has_corrections,
    }

    # =========================================================================
    # RULE SET 1: Protected Entity / OOV Preservation (Feature 7)
    # =========================================================================
    if should_protect or is_entity_like:
        w["lexical"] += 0.15
        w["semantic"] += 0.05
        w["edit"] = max(0.02, w["edit"] - 0.15)
        reasons.append("protected entity/OOV: lexical match prioritized and spelling correction minimized")

    # =========================================================================
    # RULE SET 2: Intent-Driven Adaptation (Feature 8)
    # =========================================================================
    if intent_clean == "question":
        w["semantic"] += 0.18
        w["intent"] += 0.10
        w["edit"] = max(0.02, w["edit"] - 0.12)
        w["lexical"] = max(0.05, w["lexical"] - 0.05)
        reasons.append("informational/question intent: semantic similarity and intent compatibility emphasized")

    elif intent_clean == "transactional":
        w["intent"] += 0.15
        w["semantic"] += 0.10
        w["frequency"] += 0.05
        w["edit"] = max(0.02, w["edit"] - 0.10)
        reasons.append("transactional intent: commercial intent and semantic relevance emphasized")

    elif intent_clean == "local":
        w["intent"] += 0.15
        w["context"] += 0.10
        w["semantic"] += 0.08
        w["edit"] = max(0.02, w["edit"] - 0.10)
        reasons.append("local intent: proximity intent and context signals emphasized")

    elif intent_clean == "navigational":
        w["lexical"] += 0.15
        w["frequency"] += 0.10
        w["intent"] += 0.08
        w["edit"] = max(0.02, w["edit"] - 0.10)
        w["semantic"] = max(0.02, w["semantic"] - 0.05)
        reasons.append("navigational intent: exact brand/platform lexical match and frequency emphasized")

    # =========================================================================
    # RULE SET 3: Typo Query vs Prefix Autocomplete (Features 4, 5, 6)
    # =========================================================================
    has_exact_match = any(c.get("word", "").lower() == raw_query.lower() for c in candidates)
    has_prefix_completions = any(
        c.get("source") == "trie_autocomplete" and c.get("word", "").lower().startswith(raw_query.lower())
        for c in candidates
    )
    is_pure_typo = (not has_prefix_completions and not has_exact_match and has_corrections)

    if not should_protect and is_pure_typo:
        # Misspelled query with no prefix completions needing error correction
        w["edit"] += 0.20
        w["frequency"] += 0.05
        w["lexical"] = max(0.05, w["lexical"] - 0.12)
        reasons.append("spelling correction required: Damerau-Levenshtein edit distance emphasized")

    elif has_prefix_completions and len(raw_query) <= 6:
        # Active prefix autocomplete query with strong prefix completions
        w["lexical"] += 0.18
        w["frequency"] += 0.05
        w["edit"] = max(0.02, w["edit"] - 0.12)
        reasons.append("prefix autocomplete query: lexical prefix match and Trie frequency emphasized")

    # =========================================================================
    # RULE SET 4: Code-Switched / Multilingual Query (Features 1 & 2)
    # =========================================================================
    if is_code_switched:
        w["semantic"] += 0.10
        w["context"] += 0.05
        reasons.append("code-switched / multilingual query: cross-language semantic and contextual signals boosted")

    # =========================================================================
    # RULE SET 5: Active Session Context Availability (Feature 3)
    # =========================================================================
    if has_context:
        w["context"] += 0.12
        w["user"] += 0.05
        w["frequency"] = max(0.05, w["frequency"] - 0.05)
        reasons.append("active session context available: recency and user history bonus emphasized")

    # =========================================================================
    # RULE SET 6: Multi-Word Phrase
    # =========================================================================
    if token_count >= 3 and intent_clean not in {"navigational"}:
        w["semantic"] += 0.08
        w["context"] += 0.04
        w["edit"] = max(0.02, w["edit"] - 0.05)
        reasons.append("multi-word phrase: semantic composition prioritized over token-level edits")

    normalized = normalize_weights(w)
    return {
        "weights": normalized,
        "reasons": reasons if reasons else ["standard query: balanced hybrid weights"],
        "dynamic_weighting_used": True,
        "query_profile": query_profile,
    }
