"""hybrid.py — Multi-token, multilingual candidate generator with Cross-Language Fallback.

Supports:
  • Single-word typos and autocomplete
  • Multi-word query composition and phrase reconstruction (e.g. 'buy ipohne 15 pro' -> 'buy iphone 15 pro')
  • Romanized Tamil and Loanword cross-language fallback (e.g. 'briyani venum' -> 'biryani venum')
  • Native/mixed script preservation (e.g. 'python கற்றல்')
  • Strict deduplication on full_query.lower()
"""

import re
from suggestor import autocomplete
from corrector import correct_query
from error_detector import detect_error_type

_ROMANIZED_TAMIL_MARKERS = frozenset({
    "iruku", "irruku", "irukka", "irukku", "sapda", "saapadu", "saapduven",
    "venum", "vendum", "venaam", "venaum", "kammal", "enga", "inga", "yenna",
    "enna", "paaru", "paarunga", "vaanga", "sollnga", "sollu", "thevai",
    "nalla", "kadai", "kadasiku", "kettu", "ketka", "kedukka", "sapada",
    "sapadavum", "venom", "venm", "briyani", "biryani",
})

_SKIP_CORRECTION_RE = re.compile(r"^\d+$|^[^\w]$")


def _detect_token_language(token):
    tamil_chars = sum(1 for c in token if '\u0B80' <= c <= '\u0BFF')
    if tamil_chars > 0:
        return 'tamil'
    if token.lower() in _ROMANIZED_TAMIL_MARKERS:
        return 'tamil'
    return 'english'


def _get_language_resources(lang, eng_trie, eng_dict, language_data):
    if lang != "english" and language_data:
        trie_dict = language_data.get(lang)
        if trie_dict:
            return trie_dict
    return eng_trie, eng_dict


def _get_corrections_with_fallback(
    token, primary_trie, primary_dict, user_freq,
    threshold=2, max_results=5, language_data=None, lang="english"
):
    """Attempt primary dictionary correction; if empty or loanword, fall back to English dictionary."""
    corrections = correct_query(
        primary_dict, primary_trie, token, user_freq,
        threshold=threshold, max_results=max_results
    )
    if not corrections and lang == "tamil" and language_data and "english" in language_data:
        eng_trie, eng_dict = language_data["english"]
        corrections = correct_query(
            eng_dict, eng_trie, token, user_freq,
            threshold=threshold, max_results=max_results
        )
    return corrections


def _best_correction(token, trie, dictionary_by_length, user_freq, threshold=2, language_data=None, lang="english"):
    corrections = _get_corrections_with_fallback(
        token, trie, dictionary_by_length, user_freq,
        threshold=threshold, max_results=1, language_data=language_data, lang=lang
    )
    return corrections[0] if corrections else token


def _autocomplete_last(token, trie, k=5, language_data=None, lang="english"):
    results = autocomplete(trie, token.lower(), k=k)
    if not results and lang == "tamil" and language_data and "english" in language_data:
        eng_trie, _ = language_data["english"]
        results = autocomplete(eng_trie, token.lower(), k=k)
    return results


def generate_hybrid_candidates(
    trie,
    dictionary_by_length,
    query,
    user_freq=None,
    autocomplete_k=5,
    correction_k=5,
    threshold=2,
    language_data=None
):
    raw_query = query.strip()
    if not raw_query:
        return []

    tokens = raw_query.split()

    if len(tokens) == 1:
        return _single_token_candidates(
            tokens[0], trie, dictionary_by_length,
            user_freq, autocomplete_k, correction_k, threshold, language_data
        )

    return _multi_token_candidates(
        tokens, trie, dictionary_by_length,
        user_freq, autocomplete_k, correction_k, threshold, language_data
    )


def _single_token_candidates(
    token, eng_trie, eng_dict, user_freq,
    autocomplete_k, correction_k, threshold, language_data
):
    token_lower = token.strip().lower()
    lang = _detect_token_language(token_lower)
    active_trie, active_dict = _get_language_resources(
        lang, eng_trie, eng_dict, language_data
    )

    candidates = []
    seen = set()

    ac_words = _autocomplete_last(
        token_lower, active_trie, k=autocomplete_k,
        language_data=language_data, lang=lang
    )
    for w in ac_words:
        key = w.lower()
        if key not in seen:
            seen.add(key)
            candidates.append({
                "word": w,
                "source": "autocomplete",
                "display_suggestion": w,
                "full_query": w,
                "language": lang,
                "token": token_lower,
            })

    if not _SKIP_CORRECTION_RE.match(token_lower):
        corr_words = _get_corrections_with_fallback(
            token_lower, active_trie, active_dict, user_freq,
            threshold=threshold, max_results=correction_k,
            language_data=language_data, lang=lang
        )
        for w in corr_words:
            key = w.lower()
            if key not in seen:
                seen.add(key)
                err_meta = detect_error_type(token_lower, w)
                candidates.append({
                    "word": w,
                    "source": "correction",
                    "display_suggestion": w,
                    "full_query": w,
                    "language": lang,
                    "token": token_lower,
                    **err_meta
                })
            else:
                for c in candidates:
                    if c["word"].lower() == key and c["source"] == "autocomplete":
                        c["source"] = "hybrid"

    return candidates


def _multi_token_candidates(
    tokens, eng_trie, eng_dict, user_freq,
    autocomplete_k, correction_k, threshold, language_data
):
    prefix_tokens = tokens[:-1]
    last_token = tokens[-1]

    corrected_prefix_parts = []
    phrase_language = "english"

    for tok in prefix_tokens:
        tok_lower = tok.strip().lower()
        if _SKIP_CORRECTION_RE.match(tok_lower) or not tok_lower:
            corrected_prefix_parts.append(tok_lower)
            continue
        lang = _detect_token_language(tok_lower)
        if lang == 'tamil':
            phrase_language = 'tamil'

        active_trie, active_dict = _get_language_resources(
            lang, eng_trie, eng_dict, language_data
        )

        # Exact match verification in active trie OR fallback English trie
        node = active_trie.search(tok_lower)
        is_exact = node and node.is_end
        if not is_exact and language_data and "english" in language_data:
            eng_trie_ref = language_data["english"][0]
            eng_node = eng_trie_ref.search(tok_lower)
            is_exact = eng_node and eng_node.is_end

        if is_exact:
            corrected_prefix_parts.append(tok_lower)
        else:
            best = _best_correction(
                tok_lower, active_trie, active_dict, user_freq,
                threshold=threshold, language_data=language_data, lang=lang
            )
            corrected_prefix_parts.append(best)

    last_lower = last_token.strip().lower()
    last_lang = _detect_token_language(last_lower)
    last_trie, last_dict = _get_language_resources(
        last_lang, eng_trie, eng_dict, language_data
    )

    candidates = []
    seen_phrases = set()

    def _add_phrase(suffix_word, source, err_meta=None):
        phrase_parts = corrected_prefix_parts + [suffix_word]
        full_q = " ".join(phrase_parts)
        key = full_q.lower()
        if key in seen_phrases:
            return
        seen_phrases.add(key)
        entry = {
            "word": suffix_word,
            "source": source,
            "display_suggestion": full_q,
            "full_query": full_q,
            "language": last_lang if last_lang == "tamil" else phrase_language,
            "token": last_lower,
        }
        if err_meta:
            entry.update(err_meta)
        candidates.append(entry)

    # Autocomplete expansions of the last token
    if not _SKIP_CORRECTION_RE.match(last_lower):
        ac_words = _autocomplete_last(
            last_lower, last_trie, k=autocomplete_k,
            language_data=language_data, lang=last_lang
        )
        for w in ac_words:
            _add_phrase(w, "autocomplete")

    # Correction candidates for the last token (only needed if not an exact word)
    if not _SKIP_CORRECTION_RE.match(last_lower):
        last_is_exact = False
        node = last_trie.search(last_lower)
        if node and node.is_end:
            last_is_exact = True
        elif last_lang == "tamil" and language_data and "english" in language_data:
            eng_node = language_data["english"][0].search(last_lower)
            if eng_node and eng_node.is_end:
                last_is_exact = True

        if not last_is_exact:
            corr_words = _get_corrections_with_fallback(
                last_lower, last_trie, last_dict, user_freq,
                threshold=threshold, max_results=correction_k,
                language_data=language_data, lang=last_lang
            )
            for w in corr_words:
                err_meta = detect_error_type(last_lower, w)
                key = (" ".join(corrected_prefix_parts + [w])).lower()
                if key in seen_phrases:
                    for c in candidates:
                        if c["full_query"].lower() == key and c["source"] == "autocomplete":
                            c["source"] = "hybrid"
                else:
                    _add_phrase(w, "correction", err_meta)
    else:
        _add_phrase(last_lower, "autocomplete")

    if not candidates:
        fallback = " ".join(corrected_prefix_parts + [last_lower])
        candidates.append({
            "word": last_lower,
            "source": "autocomplete",
            "display_suggestion": fallback,
            "full_query": fallback,
            "language": phrase_language,
            "token": last_lower,
        })

    return candidates