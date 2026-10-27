"""Lightweight, explainable context signals for predictive search.

The context state is intentionally simple and JSON-serializable so it can be
stored in ``recent.json``.  It does not infer semantic relationships or train
a model; it only rewards a candidate that was recently selected or appeared
as a token in a recent query.
"""

DEFAULT_HISTORY_SIZE = 10
RECENT_SELECTION_WEIGHT = 0.60
RECENT_QUERY_WEIGHT = 0.12


def load_context_state(data=None, max_history=DEFAULT_HISTORY_SIZE):
    """Normalize legacy and current ``recent.json`` data into context state."""

    data = data or {}
    return {
        "recent_words": _normalize_entries(data.get("recent_words", []), max_history),
        "query_history": _normalize_entries(data.get("query_history", []), max_history),
    }


def record_query(context_state, query, max_history=DEFAULT_HISTORY_SIZE):
    """Add a query after it has been scored, ready for the next query."""

    _record_entry(context_state["query_history"], query, max_history)


def record_selection(context_state, word, max_history=DEFAULT_HISTORY_SIZE):
    """Add a user-selected candidate as the strongest lightweight signal."""

    _record_entry(context_state["recent_words"], word, max_history)


def apply_context(candidates, context_state=None):
    """Annotate and rerank candidates using recency-decayed context signals.

    ``context_score = selection_bonus + query_bonus`` where each matching
    entry contributes ``weight / (recency_position + 1)``.  The adjusted score
    preserves the original candidate order as a base score and adds context:

    ``adjusted_score = base_rank_score + context_score``.
    """

    state = load_context_state(context_state)
    candidate_count = len(candidates)
    enriched = []

    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        word = item.get("word", "").lower()
        selection_bonus, selection_matches = _selection_bonus(
            word,
            state["recent_words"]
        )
        query_bonus, query_matches = _query_bonus(
            word,
            state["query_history"]
        )
        base_rank_score = (candidate_count - index) / max(candidate_count, 1)
        context_score = selection_bonus + query_bonus

        item.update({
            "base_rank_score": round(base_rank_score, 4),
            "context_score": round(context_score, 4),
            "context_reasons": selection_matches + query_matches,
            "context_adjusted_score": round(base_rank_score + context_score, 4),
        })
        enriched.append(item)

    return sorted(
        enriched,
        key=lambda item: item["context_adjusted_score"],
        reverse=True
    )


def _normalize_entries(entries, max_history):
    if not isinstance(entries, list):
        return []

    normalized = []
    for entry in entries:
        if not isinstance(entry, str):
            continue
        entry = entry.strip().lower()
        if entry and entry not in normalized:
            normalized.append(entry)
    return normalized[:max_history]


def _record_entry(history, entry, max_history):
    entry = entry.strip().lower()
    if not entry:
        return
    if entry in history:
        history.remove(entry)
    history.insert(0, entry)
    del history[max_history:]


def _selection_bonus(word, recent_words):
    bonus = 0.0
    reasons = []
    for index, selected_word in enumerate(recent_words):
        if word == selected_word:
            value = RECENT_SELECTION_WEIGHT / (index + 1)
            bonus += value
            reasons.append({
                "signal": "recent_selection",
                "matched_context": selected_word,
                "bonus": round(value, 4),
            })
    return bonus, reasons


def _query_bonus(word, query_history):
    bonus = 0.0
    reasons = []
    for index, previous_query in enumerate(query_history):
        if word in previous_query.split():
            value = RECENT_QUERY_WEIGHT / (index + 1)
            bonus += value
            reasons.append({
                "signal": "recent_query_token",
                "matched_context": previous_query,
                "bonus": round(value, 4),
            })
    return bonus, reasons
