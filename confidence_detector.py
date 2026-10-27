"""Feature 13: Confidence and Ambiguity Detection Module.

Calculates query and suggestion confidence scores in [0.0, 1.0] and detects
ambiguity across candidate scores, spelling corrections, intent, language,
and out-of-vocabulary entities.

Signals evaluated:
- Top candidate score quality (hybrid & ML)
- Score margin / gap between top-1 and runner-up candidates
- Lexical and edit-distance fidelity
- Language identification confidence
- Intent classification clarity
- OOV / Entity protection status
- Candidate pool diversity & distribution
"""


def evaluate_confidence_and_ambiguity(
    query,
    ranked_candidates=None,
    raw_candidate_pool=None,
    language_info=None,
    intent_info=None,
    token_results=None,
    dynamic_weights_info=None
):
    """Evaluates search result confidence and detects candidate/query ambiguities.

    Returns:
        dict:
            - confidence_score: float in [0.0, 1.0]
            - confidence_level: "high" | "medium" | "low"
            - is_ambiguous: bool
            - ambiguity_score: float in [0.0, 1.0]
            - ambiguity_type: list of ambiguity category strings
            - confidence_reasons: list of explainable reason strings
            - signal_breakdown: dict of intermediate component scores
    """
    raw_query = str(query or "").strip()
    if not raw_query:
        return {
            "confidence_score": 0.0,
            "confidence_level": "low",
            "is_ambiguous": False,
            "ambiguity_score": 0.0,
            "ambiguity_type": [],
            "confidence_reasons": ["empty query: zero confidence"],
            "signal_breakdown": {
                "top_candidate_quality": 0.0,
                "score_margin": 0.0,
                "lexical_fidelity": 0.0,
                "language_clarity": 0.0,
                "intent_clarity": 0.0,
            }
        }

    ranked_candidates = ranked_candidates or []
    language_info = language_info or {}
    intent_info = intent_info or {}
    token_results = token_results or []

    # 1. No-candidate Edge Case
    if not ranked_candidates:
        return {
            "confidence_score": 0.0,
            "confidence_level": "low",
            "is_ambiguous": False,
            "ambiguity_score": 0.0,
            "ambiguity_type": ["no_candidates_found"],
            "confidence_reasons": ["no valid autocomplete or correction candidates found in dictionary"],
            "signal_breakdown": {
                "top_candidate_quality": 0.0,
                "score_margin": 0.0,
                "lexical_fidelity": 0.0,
                "language_clarity": float(language_info.get("confidence", 0.0)),
                "intent_clarity": 0.0,
            }
        }

    top_cand = ranked_candidates[0]
    top_score = float(top_cand.get("final_score", top_cand.get("hybrid_score", 0.0)))
    top_word = str(top_cand.get("word", "")).lower()
    top_phrase = str(top_cand.get("full_query") or top_cand.get("display_suggestion") or top_word).lower()
    q_lower = raw_query.lower()

    # Check protection
    should_protect = any(c.get("should_protect") for c in ranked_candidates)
    if token_results:
        for t in token_results:
            if t.get("protection", {}).get("should_protect"):
                should_protect = True

    # -------------------------------------------------------------------------
    # SIGNAL 1: Top Candidate Quality (S_top)
    # -------------------------------------------------------------------------
    is_exact_match = (top_word == q_lower or top_phrase == q_lower)
    is_prefix_match = (top_word.startswith(q_lower) or top_phrase.startswith(q_lower)) and len(q_lower) >= 2

    if is_exact_match:
        s_top = 1.0
    elif is_prefix_match:
        s_top = max(0.85, min(1.0, top_score + 0.15))
    else:
        # If candidate is a reconstructed phrase with high hybrid score, reward it
        if " " in q_lower and " " in top_phrase:
            s_top = max(0.80, min(1.0, top_score + 0.10))
        else:
            s_top = max(0.0, min(1.0, top_score))

    # -------------------------------------------------------------------------
    # SIGNAL 2: Candidate Score Margin / Gap (S_margin)
    # -------------------------------------------------------------------------
    if len(ranked_candidates) >= 2:
        runner_up = ranked_candidates[1]
        runner_up_score = float(runner_up.get("final_score", runner_up.get("hybrid_score", 0.0)))
        score_gap = max(0.0, top_score - runner_up_score)
        # A gap of >= 0.10 gives maximum margin confidence
        s_margin = min(1.0, score_gap * 8.0)
    else:
        score_gap = 1.0
        s_margin = 1.0

    # -------------------------------------------------------------------------
    # SIGNAL 3: Lexical & Edit Fidelity (S_fidelity)
    # -------------------------------------------------------------------------
    edit_dist = top_cand.get("edit_distance")
    eval_target = top_phrase if " " in q_lower else top_word
    if edit_dist is None:
        if is_exact_match or is_prefix_match:
            edit_dist = 0
        else:
            from corrector import damerau_levenshtein
            edit_dist = damerau_levenshtein(q_lower, eval_target, max_dist=3)

    max_len = max(len(q_lower), len(eval_target), 1)
    if is_exact_match or is_prefix_match:
        s_fidelity = 1.0
    else:
        s_fidelity = max(0.0, 1.0 - (float(edit_dist) / float(max_len)))

    # -------------------------------------------------------------------------
    # SIGNAL 4: Language & Intent Clarity
    # -------------------------------------------------------------------------
    s_lang = float(language_info.get("confidence", 0.8))
    intent_val = str(intent_info.get("intent", "unknown")).lower()
    s_intent = 0.90 if intent_val not in {"unknown", ""} else 0.50

    # -------------------------------------------------------------------------
    # COMPOSITE CONFIDENCE SCORE
    # -------------------------------------------------------------------------
    # Formula: 0.35 * top_quality + 0.25 * fidelity + 0.20 * margin + 0.10 * lang + 0.10 * intent
    raw_confidence = (
        0.35 * s_top
        + 0.25 * s_fidelity
        + 0.20 * s_margin
        + 0.10 * s_lang
        + 0.10 * s_intent
    )

    if should_protect and (is_exact_match or is_prefix_match):
        raw_confidence = max(raw_confidence, 0.90)

    # Severe typo dampening: if edit distance > 2 on short word, cap confidence
    if edit_dist is not None and edit_dist >= 3 and len(q_lower) <= 6:
        raw_confidence = min(raw_confidence, 0.40)

    confidence_score = round(max(0.0, min(1.0, raw_confidence)), 4)

    # Confidence Levels
    if confidence_score >= 0.75:
        confidence_level = "high"
    elif confidence_score >= 0.45:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    # -------------------------------------------------------------------------
    # AMBIGUITY DETECTION
    # -------------------------------------------------------------------------
    ambiguity_types = []
    ambiguity_signals = []
    confidence_reasons = []

    # 1. Candidate Score Competition (Nearly equal scores at the top)
    if len(ranked_candidates) >= 2:
        runner_up_score = float(ranked_candidates[1].get("final_score", ranked_candidates[1].get("hybrid_score", 0.0)))
        gap = top_score - runner_up_score
        if gap < 0.025 and top_score > 0.3:
            ambiguity_types.append("candidate_score_competition")
            amb_comp = round(1.0 - (gap / 0.025), 4)
            ambiguity_signals.append(amb_comp * 0.45)
            confidence_reasons.append(f"close score competition: top candidates differ by only {gap:.4f}")
        else:
            confidence_reasons.append(f"clear score separation: margin gap of {gap:.4f} over runner-up")

    # 2. Uncertain Spelling Correction
    if not (is_exact_match or is_prefix_match):
        if edit_dist is not None and edit_dist >= 2:
            ambiguity_types.append("uncertain_spelling_correction")
            ambiguity_signals.append(0.35)
            confidence_reasons.append(f"significant edit distance ({edit_dist}) required for correction")
        elif top_cand.get("source") in {"damerau_levenshtein", "correction"}:
            confidence_reasons.append(f"spelling correction applied: '{top_word}' (edit distance {edit_dist})")

    # 3. Ambiguous Intent
    if intent_val == "unknown":
        ambiguity_types.append("ambiguous_intent")
        ambiguity_signals.append(0.20)
        confidence_reasons.append("intent is unclassified / general search")
    else:
        confidence_reasons.append(f"intent identified with clarity: {intent_val}")

    # 4. Ambiguous Language / Low Language Confidence
    if s_lang < 0.60 or language_info.get("type") == "mixed":
        ambiguity_types.append("ambiguous_language")
        ambiguity_signals.append(0.25)
        confidence_reasons.append(f"multilingual or mixed-language query (language confidence: {s_lang:.2f})")

    # 5. OOV / Entity Uncertainty
    if should_protect and not is_exact_match:
        ambiguity_types.append("oov_entity_uncertainty")
        ambiguity_signals.append(0.25)
        confidence_reasons.append("protected entity token detected with potential partial match")

    # Prefix/Exact match positive reasons
    if is_exact_match:
        confidence_reasons.insert(0, f"exact match confirmed in dictionary: '{top_word}'")
    elif is_prefix_match:
        confidence_reasons.insert(0, f"strong prefix match in dictionary: '{top_word}'")

    ambiguity_score = round(min(1.0, sum(ambiguity_signals)), 4)
    is_ambiguous = bool(ambiguity_score >= 0.35 or len(ambiguity_types) >= 2)

    return {
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
        "is_ambiguous": is_ambiguous,
        "ambiguity_score": ambiguity_score,
        "ambiguity_type": ambiguity_types,
        "confidence_reasons": confidence_reasons,
        "signal_breakdown": {
            "top_candidate_quality": round(s_top, 4),
            "score_margin": round(s_margin, 4),
            "lexical_fidelity": round(s_fidelity, 4),
            "language_clarity": round(s_lang, 4),
            "intent_clarity": round(s_intent, 4),
        }
    }
