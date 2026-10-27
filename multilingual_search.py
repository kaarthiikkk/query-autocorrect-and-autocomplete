from romanized_lexicon import lookup_romanized_candidates
from unified_language_detector import analyze_language, analyze_token
from suggestor import autocomplete
from corrector import correct_query, damerau_levenshtein
from context_predictor import apply_context
from error_detector import detect_error_type
from oov_protector import assess_token
from intent_detector import detect_intent
from ranker import rank_multilingual_candidates, DEFAULT_HYBRID_WEIGHTS
from ml_ranker import rerank_candidates, load_model
from dynamic_weights import get_dynamic_weights
from feedback_learner import get_feedback_learner
from confidence_detector import evaluate_confidence_and_ambiguity
from temporal_ranker import detect_temporal_query, get_temporal_ranker
from personalized_ranker import get_personalized_ranker


_CACHED_ML_MODEL = None
_ML_MODEL_LOADED = False
_CACHED_ROMANIZED_RESOURCES = None


def _get_romanized_resources(language_data=None, language="tamil"):
    """Retrieve or lazily initialize the Romanized trie and dictionary for tamil, hindi, or telugu."""
    rom_key = f"{language}_romanized" if not language.endswith("_romanized") else language
    if language_data and rom_key in language_data:
        return language_data[rom_key]
    try:
        from data_loader import load_dictionary_and_trie
        return load_dictionary_and_trie(rom_key)
    except Exception:
        return (None, None)


def _get_default_ml_model():
    global _CACHED_ML_MODEL, _ML_MODEL_LOADED
    if not _ML_MODEL_LOADED:
        _CACHED_ML_MODEL = load_model("models/ml_ranker.pkl")
        _ML_MODEL_LOADED = True
    return _CACHED_ML_MODEL


def multilingual_search(
    query,
    language_data,
    user_freq=None,
    autocomplete_k=5,
    correction_k=5,
    context=None,
    ml_model=None,
    weights=None,
    enable_dynamic_weights=True,
    user_id="default",
    top_k=5,
    target_language=None,
    **kwargs
):
    """Generate language-aware candidate pools and rank them globally.

    Architecture:
        Candidate Pool → Dynamic Weight Adaptation (Feature 11)
        → Hybrid Ranking with User Feedback (Features 9, 12)
        → ML Reranker (Feature 10) → Confidence & Ambiguity Detection (Feature 13)
        → Temporal / Trending Boost (Feature 14)
        → Top-k Personalised Suggestions (Feature 15)
    """

    raw_query = query.strip()
    query = raw_query.lower()
    active_user_freq = user_freq if user_freq is not None else get_feedback_learner().get_user_frequencies()

    if not query:
        return {
            "language": "unknown",
            "type": "unknown",
            "confidence": 0.0,
            "intent": "unknown",
            "tokens": [],
            "token_results": [],
            "autocomplete_candidates": [],
            "correction_candidates": [],
            "candidate_pool": [],
            "ranked_candidates": [],
            "ranked_results": [],
            "ml_reranked_candidates": [],
            "ml_reranked_results": [],
            "ml_reranking_used": False,
            "dynamic_weights": dict(DEFAULT_HYBRID_WEIGHTS),
            "weight_adaptation_reason": ["empty query: default weights applied"],
            "dynamic_weighting_used": False,
            "confidence_score": 0.0,
            "confidence_level": "low",
            "is_ambiguous": False,
            "ambiguity_score": 0.0,
            "ambiguity_type": [],
            "confidence_reasons": ["empty query: zero confidence"],
            "temporal_detected": False,
            "temporal_type": None,
            "temporal_signals": [],
            "temporal_confidence": 0.0,
            "temporal_candidates": [],
            "temporal_results": [],
            "temporal_ranking_used": False,
            "personalized_candidates": [],
            "personalized_results": [],
            "top_k_suggestions": [],
            "user_id": user_id,
            "autocomplete": [],
            "corrections": []
        }

    analysis = analyze_language(query, target_language=target_language)
    intent_analysis = detect_intent(raw_query)
    tokens = query.split()
    raw_tokens = raw_query.split()
    token_results = [
        _generate_token_candidates(
            token,
            raw_token,
            analyze_token(token, target_language=target_language),
            language_data,
            active_user_freq,
            autocomplete_k,
            correction_k,
            context
        )
        for token, raw_token in zip(tokens, raw_tokens)
    ]

    if len(tokens) > 1:
        phrase_candidates = _generate_phrase_candidates(
            tokens,
            raw_tokens,
            token_results,
            language_data,
            active_user_freq,
            autocomplete_k,
            correction_k,
            analysis,
            context
        )
        autocomplete_candidates = [
            c for c in phrase_candidates
            if "autocomplete" in c.get("source", "") or c.get("source") in {"hybrid_phrase", "romanized_lexicon"}
        ]
        correction_candidates = [
            c for c in phrase_candidates
            if "correction" in c.get("source", "")
        ]
        if not correction_candidates:
            correction_candidates = phrase_candidates[:correction_k]
        if not autocomplete_candidates:
            autocomplete_candidates = phrase_candidates[:autocomplete_k]
        raw_candidate_pool = phrase_candidates
    else:
        autocomplete_candidates = apply_context(
            _merge_candidates(token_results, "autocomplete_candidates"),
            context
        )
        correction_candidates = apply_context(
            _merge_candidates(token_results, "correction_candidates"),
            context
        )
        seen_words = set()
        deduped_pool = []
        for c in autocomplete_candidates + correction_candidates:
            if c["word"] not in seen_words:
                seen_words.add(c["word"])
                deduped_pool.append(c)
        raw_candidate_pool = deduped_pool

    # Feature 11: Dynamic Ranking Weight Adaptation
    if weights is not None or not enable_dynamic_weights:
        active_weights = weights if weights is not None else dict(DEFAULT_HYBRID_WEIGHTS)
        dynamic_weights_info = {
            "weights": active_weights,
            "reasons": ["explicit or fixed weights supplied"],
            "dynamic_weighting_used": False
        }
    else:
        dynamic_weights_info = get_dynamic_weights(
            raw_query,
            intent=intent_analysis["intent"],
            language=analysis["language"],
            language_type=analysis["type"],
            candidates=raw_candidate_pool,
            context=context,
            token_results=token_results
        )

    ranked_candidates = rank_multilingual_candidates(
        raw_query,
        raw_candidate_pool,
        language_data,
        active_user_freq,
        intent=intent_analysis["intent"],
        weights=dynamic_weights_info["weights"]
    )

    active_model = ml_model if ml_model is not None else _get_default_ml_model()
    ml_reranked_candidates = rerank_candidates(
        raw_query,
        ranked_candidates,
        model=active_model
    )
    ml_used = active_model is not None

    # Feature 13: Confidence and Ambiguity Detection
    confidence_info = evaluate_confidence_and_ambiguity(
        raw_query,
        ranked_candidates=ml_reranked_candidates,
        raw_candidate_pool=raw_candidate_pool,
        language_info=analysis,
        intent_info=intent_analysis,
        token_results=token_results,
        dynamic_weights_info=dynamic_weights_info
    )

    # Feature 14: Temporal / Trending Query Adaptation
    temporal_info = detect_temporal_query(raw_query)
    temporal_ranker = get_temporal_ranker()
    temporal_candidates = temporal_ranker.apply_temporal_boost(
        ml_reranked_candidates,
        raw_query,
        temporal_info
    )

    # Feature 15: Top-k Personalised Suggestions
    personalized_ranker = get_personalized_ranker()
    personalized_candidates = personalized_ranker.apply_personalization(
        temporal_candidates,
        user_id=user_id,
        raw_query=raw_query,
        intent=intent_analysis["intent"],
        language=analysis["language"],
        top_k=top_k
    )

    return {
        "language": analysis["language"],
        "type": analysis["type"],
        "confidence": analysis["confidence"],
        "intent": intent_analysis["intent"],
        "tokens": tokens,
        "token_results": token_results,
        "autocomplete_candidates": autocomplete_candidates,
        "correction_candidates": correction_candidates,
        "candidate_pool": temporal_candidates,
        "ranked_candidates": ranked_candidates,
        "ranked_results": [item["word"] for item in ranked_candidates],
        "ml_reranked_candidates": ml_reranked_candidates,
        "ml_reranked_results": [item["word"] for item in ml_reranked_candidates],
        "ml_reranking_used": ml_used,
        "dynamic_weights": dynamic_weights_info["weights"],
        "weight_adaptation_reason": dynamic_weights_info["reasons"],
        "dynamic_weighting_used": dynamic_weights_info["dynamic_weighting_used"],
        "confidence_score": confidence_info["confidence_score"],
        "confidence_level": confidence_info["confidence_level"],
        "is_ambiguous": confidence_info["is_ambiguous"],
        "ambiguity_score": confidence_info["ambiguity_score"],
        "ambiguity_type": confidence_info["ambiguity_type"],
        "confidence_reasons": confidence_info["confidence_reasons"],
        "confidence_analysis": confidence_info,
        "temporal_detected": temporal_info["temporal_detected"],
        "temporal_type": temporal_info["temporal_type"],
        "temporal_signals": temporal_info["temporal_signals"],
        "temporal_confidence": temporal_info["temporal_confidence"],
        "temporal_candidates": temporal_candidates,
        "temporal_results": [item["word"] for item in temporal_candidates],
        "temporal_ranking_used": True,
        # Feature 15: Personalized Suggestions
        "personalized_candidates": personalized_candidates,
        "personalized_results": [item["word"] for item in personalized_candidates],
        "top_k_suggestions": [item["word"] for item in personalized_candidates],
        "user_id": user_id,
        # Keep the original public fields for existing callers.
        "autocomplete": [item["word"] for item in autocomplete_candidates],
        "corrections": [item["word"] for item in correction_candidates]
    }


def _generate_token_candidates(
    token,
    raw_token,
    token_analysis,
    language_data,
    user_freq,
    autocomplete_k,
    correction_k,
    context
):
    """Dispatch one token to a language-specific candidate provider."""

    language = token_analysis["language"]
    result = {
        "token": token,
        "language": language,
        "type": token_analysis["type"],
        "confidence": token_analysis["confidence"],
        "resource_available": language in language_data,
        "autocomplete_candidates": [],
        "correction_candidates": []
    }

    if token_analysis["type"] == "romanized":
        return _generate_romanized_candidates(
            result,
            token,
            raw_token,
            language_data,
            user_freq,
            autocomplete_k,
            correction_k,
            context
        )

    # Native script candidate generation for any supported language
    if token_analysis["type"] == "native_script":
        resources = language_data.get(language)
        if not resources:
            try:
                from data_loader import load_dictionary_and_trie
                resources = load_dictionary_and_trie(language)
                if resources and resources[0]:
                    language_data[language] = resources
            except Exception:
                resources = None

        if resources and resources[0]:
            native_res = _generate_native_candidates(
                result,
                token,
                raw_token,
                resources,
                user_freq,
                autocomplete_k,
                correction_k,
                context
            )
            # Cross-lingual recovery: if an English-classified token is OOV with no good completions,
            # check Romanized Tamil corpus as fallback (e.g. vemum, venum, briyani)
            if language == "english" and not native_res.get("autocomplete_candidates"):
                rom_trie, rom_dict = _get_romanized_resources(language_data)
                if rom_trie and (rom_trie.search(token) or lookup_romanized_candidates("tamil", token)):
                    rom_res = _generate_romanized_candidates(
                        dict(result),
                        token,
                        raw_token,
                        language_data,
                        user_freq,
                        autocomplete_k,
                        correction_k,
                        context
                    )
                    if rom_res.get("autocomplete_candidates") or rom_res.get("correction_candidates"):
                        return rom_res
            return native_res

    result["resource_status"] = "unsupported_token_or_language"
    return result



def _generate_native_candidates(
    result,
    token,
    raw_token,
    resources,
    user_freq,
    autocomplete_k,
    correction_k,
    context
):
    """Generate ordinary Trie and correction candidates for native input."""

    trie, dictionary_by_length = resources
    language = result["language"]
    protection = assess_token(
        token, trie, language, result["type"], context, raw_token
    )
    result["protection"] = protection
    result["resource_status"] = "available"
    result["autocomplete_candidates"] = [
        _with_protection(
            _candidate(word, token, language, result["type"], "trie_autocomplete"),
            protection
        )
        for word in autocomplete(trie, token, k=autocomplete_k)
    ]
    if correction_k > 0 and not protection["should_protect"]:
        result["correction_candidates"] = [
            _with_protection(
                _correction_candidate(word, token, language, result["type"]),
                protection
            )
            for word in correct_query(
                dictionary_by_length,
                trie,
                token,
                user_freq,
                threshold=2,
                max_results=correction_k
            )
        ]
    return result


def _generate_romanized_candidates(
    result,
    token,
    raw_token,
    language_data,
    user_freq=None,
    autocomplete_k=5,
    correction_k=5,
    context=None
):
    """Bridge data-driven Romanized forms (from ta_rom) and curated lexicons to Romanized corrections and native resources."""

    language = result["language"]
    native_trie = language_data[language][0] if language in language_data else None
    if not native_trie:
        try:
            from data_loader import load_dictionary_and_trie
            res = load_dictionary_and_trie(language)
            native_trie = res[0] if res else None
        except Exception:
            native_trie = None
    protection = assess_token(token, native_trie, language, "romanized", context, raw_token)
    result["protection"] = protection

    rom_trie, rom_dict = _get_romanized_resources(language_data, language)

    corr_words = []
    # 1. Romanized Spelling Correction (Feature 2 with ta_rom dataset)
    has_hard_entity_signal = any(
        s in protection.get("entity_signals", [])
        for s in ("digits_or_symbols", "acronym", "capitalization_or_mixed_case")
    )
    if rom_dict and rom_trie and correction_k > 0 and not has_hard_entity_signal:
        corr_words = correct_query(
            rom_dict,
            rom_trie,
            token,
            user_freq=user_freq,
            threshold=2,
            max_results=correction_k
        )
        for cw in corr_words:
            cand = _correction_candidate(cw, token, language, "romanized")
            cand["source"] = "romanized_correction"
            if rom_trie:
                rnode = rom_trie.search(cw)
                if rnode and rnode.is_end:
                    cand["frequency"] = rnode.freq
            result["correction_candidates"].append(
                _with_protection(cand, protection)
            )

    # 2. Romanized Autocomplete (prefix completions in Romanized script)
    ac_words = []
    if rom_trie and autocomplete_k > 0:
        node = rom_trie.search(token)
        if node and node.is_end:
            ac_words.append(token)
        for w in autocomplete(rom_trie, token, k=autocomplete_k):
            if w not in ac_words:
                ac_words.append(w)
        for aw in ac_words:
            result["autocomplete_candidates"].append(
                _with_protection(
                    _candidate(aw, token, language, "romanized", "trie_autocomplete"),
                    protection
                )
            )

    # 3. Bridge to Native Script via Romanized Lexicon
    words_to_bridge = [token]
    for w in corr_words[:2]:
        if w not in words_to_bridge:
            words_to_bridge.append(w)
    for w in ac_words[:2]:
        if w not in words_to_bridge:
            words_to_bridge.append(w)

    seen_native = set()
    bridged_any = False

    for w in words_to_bridge:
        mapped_words = lookup_romanized_candidates(language, w)
        if not mapped_words and language != "tamil":
            mapped_words = lookup_romanized_candidates("tamil", w)
        for mapped_word in mapped_words:
            if mapped_word in seen_native:
                continue
            seen_native.add(mapped_word)
            bridged_any = True

            result["autocomplete_candidates"].append(
                _with_protection(
                    _candidate(mapped_word, token, language, "romanized", "romanized_lexicon"),
                    protection
                )
            )
            if native_trie:
                for n_ac in autocomplete(native_trie, mapped_word, k=autocomplete_k):
                    if n_ac not in seen_native:
                        seen_native.add(n_ac)
                        result["autocomplete_candidates"].append(
                            _with_protection(
                                _candidate(n_ac, token, language, "romanized", "romanized_lexicon"),
                                protection
                            )
                        )

    if bridged_any:
        result["resource_status"] = "available_via_romanized_lexicon"
    elif corr_words or ac_words:
        result["resource_status"] = "available_via_romanized_corpus"
    else:
        result["resource_status"] = "romanized_lexicon_no_match"

    result["resource_available"] = True
    return result


def _candidate(word, token, language, script_type, source):
    return {
        "word": word,
        "token": token,
        "language": language,
        "type": script_type,
        "source": source
    }


def _correction_candidate(word, token, language, script_type):
    candidate = _candidate(
        word,
        token,
        language,
        script_type,
        "damerau_levenshtein"
    )
    candidate.update(detect_error_type(token, word))
    return candidate


def _with_protection(candidate, protection):
    candidate.update({
        key: protection[key]
        for key in (
            "oov_status", "rare_status", "is_entity_like", "should_protect",
            "protection_status", "protection_reason", "protection_score"
        )
    })
    return candidate


def _merge_candidates(token_results, candidate_key):
    """Deduplicate candidates while retaining every token/source provenance."""

    merged = {}
    for token_result in token_results:
        for candidate in token_result[candidate_key]:
            key = (candidate["word"], candidate["language"])
            if key not in merged:
                merged[key] = {**candidate, "provenance": [candidate["token"]]}
            elif candidate["token"] not in merged[key]["provenance"]:
                merged[key]["provenance"].append(candidate["token"])
    return list(merged.values())


def _generate_phrase_candidates(
    tokens,
    raw_tokens,
    token_results,
    language_data,
    active_user_freq,
    autocomplete_k,
    correction_k,
    analysis,
    context
):
    """Reconstruct multi-token candidate query expressions preserving script and order."""
    import itertools

    prefix_tokens = tokens[:-1]
    last_token = tokens[-1]
    last_idx = len(tokens) - 1

    prefix_choices = []
    merged_protections = {}

    for j, tok in enumerate(prefix_tokens):
        tok_res = token_results[j]
        lang = tok_res.get("language", "english")
        prot = tok_res.get("protection", {})
        merged_protections.update(prot)

        trie, dict_by_len = language_data.get(lang, (None, None))
        is_exact = False
        tok_freq = 0
        if trie:
            node = trie.search(tok)
            if node and node.is_end:
                is_exact = True
                tok_freq = node.freq

        tok_opts = []
        # 1. Romanized lexicon (checks token language and fallback to tamil)
        rom = lookup_romanized_candidates(lang, tok)
        if not rom and lang != "tamil":
            rom = lookup_romanized_candidates("tamil", tok)
        if rom:
            tok_opts.extend(rom)

        # 2. Autocomplete completions (e.g. pytho -> python)
        ac_words = [c["word"] for c in tok_res.get("autocomplete_candidates", [])]
        if not ac_words and trie:
            ac_words = autocomplete(trie, tok, k=3)

        if ac_words and trie:
            top_ac = ac_words[0]
            ac_node = trie.search(top_ac)
            ac_freq = ac_node.freq if ac_node and ac_node.is_end else 1
            if ac_freq > tok_freq * 10 or not is_exact:
                if top_ac not in tok_opts:
                    tok_opts.append(top_ac)

        # 3. Spelling corrections (e.g. pythno -> python)
        corr_words = [c["word"] for c in tok_res.get("correction_candidates", [])]
        if not corr_words and dict_by_len and trie:
            corr_words = correct_query(dict_by_len, trie, tok, user_freq=active_user_freq, max_results=3)
        if corr_words and trie:
            top_node = trie.search(corr_words[0])
            top_freq = top_node.freq if top_node and top_node.is_end else 1
            if top_freq > 100000:
                corr_words = [
                    cw for cw in corr_words
                    if (trie.search(cw).freq if trie.search(cw) and trie.search(cw).is_end else 1) * 50 >= top_freq
                ]
        for cw in corr_words:
            if cw not in tok_opts:
                tok_opts.append(cw)

        # 4. Exact / original token
        has_dominant_expansion = ac_words and trie and ((trie.search(ac_words[0]).freq if trie.search(ac_words[0]) else 1) > max(1, tok_freq) * 50)
        if has_dominant_expansion:
            tok_opts = [w for w in tok_opts if w != tok]
        elif is_exact and (tok_freq > 100000 or not tok_opts):
            if tok not in tok_opts:
                tok_opts.insert(0, tok)
        elif not is_exact and tok_opts and not prot.get("should_protect"):
            # Obvious typo/non-word with valid correction options available - omit the typo
            pass
        elif tok not in tok_opts:
            tok_opts.append(tok)

        # If input query was Latin script, place Latin script options first
        is_q_latin = all(ord(c) < 128 or ('\u00C0' <= c <= '\u024F') for c in " ".join(tokens))
        if is_q_latin:
            tok_opts.sort(key=lambda w: 0 if all(ord(c) < 128 or ('\u00C0' <= c <= '\u024F') for c in w) else 1)

        prefix_choices.append(tok_opts[:3])

    last_res = token_results[last_idx]
    last_lang = last_res.get("language", "english")
    last_trie, last_dict = language_data.get(last_lang, (None, None))
    last_prot = last_res.get("protection", {})
    merged_protections.update(last_prot)

    last_opts = []
    # 1. Romanized
    rom_last = lookup_romanized_candidates(last_lang, last_token)
    if not rom_last and last_lang != "tamil":
        rom_last = lookup_romanized_candidates("tamil", last_token)
    if rom_last:
        for rw in rom_last:
            last_opts.append((rw, "romanized_lexicon"))

    # Also bridge top Romanized corrections of last token to native script (e.g. vemum -> venum -> வேண்டும்)
    last_corrs = [c["word"] for c in last_res.get("correction_candidates", [])]
    for cw in last_corrs[:2]:
        for rw in lookup_romanized_candidates("tamil", cw):
            if not any(rw == o[0] for o in last_opts):
                last_opts.append((rw, "romanized_lexicon"))

    # 2. Autocomplete candidates
    rom_trie, _ = _get_romanized_resources(language_data)
    is_rom_context = (
        analysis.get("language") == "tamil"
        or analysis.get("type") == "romanized"
        or any(t.get("type") == "romanized" for t in token_results)
    )

    last_ac = [c["word"] for c in last_res.get("autocomplete_candidates", [])]
    if is_rom_context and rom_trie and all(ord(c) < 128 for c in last_token):
        rom_completions = autocomplete(rom_trie, last_token, k=autocomplete_k)
        last_ac = rom_completions + [w for w in last_ac if w not in rom_completions]
    elif not last_ac and last_trie:
        last_ac = autocomplete(last_trie, last_token, k=autocomplete_k)
    for ac_w in last_ac:
        if not any(ac_w == o[0] for o in last_opts):
            last_opts.append((ac_w, "autocomplete"))

    # 3. Correction candidates
    if not last_corrs and last_dict and last_trie:
        last_corrs = correct_query(last_dict, last_trie, last_token, user_freq=active_user_freq, max_results=correction_k)
    for cr_w in last_corrs:
        if not any(cr_w == o[0] for o in last_opts):
            last_opts.append((cr_w, "correction"))

    # 4. Fallback original token
    last_node = last_trie.search(last_token) if last_trie else None
    last_is_exact = last_node is not None and last_node.is_end
    last_has_dominant = last_ac and last_trie and ((last_trie.search(last_ac[0]).freq if last_trie.search(last_ac[0]) else 1) > max(1, (last_node.freq if last_is_exact else 0)) * 50)
    if last_has_dominant:
        last_opts = [o for o in last_opts if o[0] != last_token]
    elif not last_is_exact and last_opts and not last_prot.get("should_protect"):
        # Typo at the end with valid expansions/corrections
        pass
    elif not any(last_token == o[0] for o in last_opts):
        last_opts.append((last_token, "original"))

    is_query_latin = all(ord(c) < 128 or ('\u00C0' <= c <= '\u024F') for c in " ".join(tokens))
    if is_query_latin:
        last_opts.sort(key=lambda opt: 0 if all(ord(c) < 128 or ('\u00C0' <= c <= '\u024F') for c in opt[0]) else 1)


    prefix_combos = list(itertools.product(*prefix_choices)) if prefix_choices else [()]

    query_str = " ".join(tokens)
    phrases = []
    seen = set()

    for p_combo in prefix_combos:
        for last_w, src in last_opts:
            full_parts = list(p_combo) + [last_w]
            full_phrase = " ".join(full_parts)
            if full_phrase in seen:
                continue
            seen.add(full_phrase)

            # Compute constituent frequency across words using geometric mean (checking English, Tamil, and Romanized Tamil tries)
            import math
            rom_trie, _ = _get_romanized_resources(language_data)
            log_freqs = []
            is_query_romanized = analysis.get("language") == "tamil" or analysis.get("type") == "romanized"
            for w in full_parts:
                w_lang = "tamil" if any("\u0B80" <= c <= "\u0BFF" for c in w) else "english"
                t, _ = language_data.get(w_lang, (None, None))
                found_freq = 0
                if is_query_romanized and rom_trie and all(ord(c) < 128 for c in w):
                    rnode = rom_trie.search(w)
                    if rnode and rnode.is_end:
                        found_freq = rnode.freq
                if not found_freq and t:
                    wnode = t.search(w)
                    if wnode and wnode.is_end:
                        found_freq = wnode.freq
                if not found_freq and rom_trie:
                    rnode = rom_trie.search(w)
                    if rnode and rnode.is_end:
                        found_freq = rnode.freq
                log_freqs.append(math.log(max(1, found_freq)))
            avg_log_freq = sum(log_freqs) / max(len(log_freqs), 1)
            phrase_freq = max(1, int(math.exp(avg_log_freq)))

            is_native_phrase = any("\u0B80" <= c <= "\u0BFF" for c in full_phrase)

            if is_native_phrase and is_query_latin:
                edit_dist = 0
            else:
                # Token-aligned edit distance (prefix completions count as 0 distance)
                token_dists = []
                for tok, w in zip(tokens, full_parts):
                    if w.startswith(tok) or w == tok or (any("\u0B80" <= c <= "\u0BFF" for c in w) and not any("\u0B80" <= c <= "\u0BFF" for c in tok)):
                        token_dists.append(0)
                    else:
                        token_dists.append(damerau_levenshtein(tok, w))
                edit_dist = sum(token_dists)

            p_type = "native_script" if is_native_phrase else ("romanized" if analysis.get("language") == "tamil" else analysis.get("type", "native_script"))
            p_lang = "tamil" if (analysis.get("language") == "tamil" or is_native_phrase) else analysis.get("language", "english")
            is_full_latin = all(ord(c) < 128 for c in full_phrase.replace(" ", ""))

            phrase_src = src
            if is_full_latin and (is_query_romanized or analysis.get("language") == "tamil"):
                phrase_src = "romanized_correction"

            cand = {
                "word": full_phrase,
                "display_suggestion": full_phrase,
                "full_query": full_phrase,
                "token": query_str,
                "language": p_lang,
                "type": p_type,
                "source": phrase_src,
                "edit_distance": edit_dist,
                "frequency": phrase_freq,
                "is_phrase": True,
                "provenance": tokens,
            }
            if merged_protections:
                cand = _with_protection(cand, merged_protections)
            phrases.append(cand)

    return apply_context(phrases, context)
