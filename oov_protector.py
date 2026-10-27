"""Explainable OOV, rare-word, and entity-like token protection."""

import unicodedata

RARE_FREQUENCY_THRESHOLD = 1_000
ENTITY_PROTECTION_THRESHOLD = 0.45


def assess_token(token, trie=None, language="unknown", script_type="unknown",
                 context=None, raw_token=None):
    """Return a lightweight protection decision without scanning a dictionary.

    Known rare terms are preserved.  Unknown terms are protected only with a
    strong lexical signal (capitalization, acronym/mixed case, digits/symbols,
    a recognized Romanized language form, or recent user context).  Plain
    lowercase unknown words remain eligible for Damerau-Levenshtein correction.
    """

    raw_token = raw_token if raw_token is not None else token
    normalized = token.strip().lower()
    node = trie.search(normalized) if trie else None
    is_known = bool(node and node.is_end)
    frequency = node.freq if is_known else None
    is_rare = bool(is_known and frequency <= RARE_FREQUENCY_THRESHOLD)

    signals = []
    score = 0.0
    if script_type == "romanized":
        score += 0.60
        signals.append("recognized_romanized_form")
    if _has_digits_or_symbols(raw_token):
        score += 0.60
        signals.append("digits_or_symbols")
    if _is_acronym(raw_token):
        score += 0.60
        signals.append("acronym")
    elif _has_name_or_mixed_case(raw_token):
        score += 0.45
        signals.append("capitalization_or_mixed_case")
    if len(raw_token) >= 14:
        score += 0.15
        signals.append("unusually_long")
    if _in_recent_context(normalized, context):
        score += 0.45
        signals.append("recent_user_context")

    is_entity_like = not is_known and score >= ENTITY_PROTECTION_THRESHOLD
    should_protect = is_rare or is_entity_like
    if is_rare:
        reason = "known_rare_word"
    elif is_entity_like:
        reason = "entity_like_oov"
    elif is_known:
        reason = "known_word"
    else:
        reason = "ordinary_oov_allow_correction"

    return {
        "token": token,
        "is_oov": not is_known,
        "oov_status": "oov" if not is_known else "known",
        "frequency": frequency,
        "is_rare": is_rare,
        "rare_status": "rare" if is_rare else "not_rare",
        "is_entity_like": is_entity_like,
        "entity_signals": signals,
        "protection_score": round(score, 3),
        "should_protect": should_protect,
        "protection_status": "protected" if should_protect else "not_protected",
        "protection_reason": reason,
        "language": language,
    }


def _has_digits_or_symbols(token):
    return any(
        char.isdigit() or unicodedata.category(char)[0] in {"P", "S"}
        for char in token
    )


def _is_acronym(token):
    letters = [char for char in token if char.isalpha()]
    return len(letters) >= 2 and all(char.isupper() for char in letters)


def _has_name_or_mixed_case(token):
    return any(char.isupper() for char in token[1:]) or (token[:1].isupper() and token[1:].islower())


def _in_recent_context(token, context):
    if not context:
        return False
    entries = context.get("recent_words", []) + context.get("query_history", [])
    return any(token in str(entry).lower().split() for entry in entries)
