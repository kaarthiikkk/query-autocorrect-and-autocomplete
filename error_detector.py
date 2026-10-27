"""Explain the edit relationship between a query and correction candidate."""

from corrector import damerau_levenshtein


ERROR_TYPE_PRIORITY = {
    "insertion": 0,
    "deletion": 1,
    "substitution": 2,
    "transposition": 3,
    "multiple/combined": 4,
    "none": 5,
}


def detect_error_type(query, candidate):
    """Return structured, deterministic error metadata for one candidate.

    Error labels describe the user's query relative to the correction.  For
    example, ``pythonn`` has an insertion because removing one character from
    the query yields ``python``.  Any non-single-edit relationship is labelled
    ``multiple/combined``.
    """

    query = query.strip().lower()
    candidate = candidate.strip().lower() if candidate else ""

    if not candidate:
        return {
            "error_type": "none",
            "edit_distance": None,
            "query": query,
            "candidate": candidate or None,
        }

    distance = damerau_levenshtein(query, candidate)
    error_type = _single_edit_type(query, candidate)

    if distance == 0:
        error_type = "none"
    elif error_type is None:
        error_type = "multiple/combined"

    return {
        "error_type": error_type,
        "edit_distance": distance,
        "query": query,
        "candidate": candidate,
    }


def select_best_error(query, candidates):
    """Summarize a correction list without performing another dictionary scan.

    Ties use the documented deterministic order: edit distance first, then the
    error-type order in ``ERROR_TYPE_PRIORITY``, then candidate spelling.
    """

    analyses = [
        detect_error_type(query, _candidate_word(candidate))
        for candidate in candidates or []
        if _candidate_word(candidate)
    ]

    if not analyses:
        return detect_error_type(query, None)

    return min(
        analyses,
        key=lambda item: (
            item["edit_distance"],
            ERROR_TYPE_PRIORITY[item["error_type"]],
            item["candidate"],
        )
    )


def _candidate_word(candidate):
    if isinstance(candidate, dict):
        return candidate.get("word", "")
    return candidate


def _single_edit_type(query, candidate):
    if len(query) == len(candidate) + 1 and _removing_one_character(query, candidate):
        return "insertion"

    if len(candidate) == len(query) + 1 and _removing_one_character(candidate, query):
        return "deletion"

    if len(query) == len(candidate):
        mismatches = [
            index
            for index, (query_char, candidate_char) in enumerate(zip(query, candidate))
            if query_char != candidate_char
        ]
        if len(mismatches) == 1:
            return "substitution"
        if (
            len(mismatches) == 2
            and mismatches[1] == mismatches[0] + 1
            and query[mismatches[0]] == candidate[mismatches[1]]
            and query[mismatches[1]] == candidate[mismatches[0]]
        ):
            return "transposition"

    return None


def _removing_one_character(longer_word, shorter_word):
    return any(
        longer_word[:index] + longer_word[index + 1:] == shorter_word
        for index in range(len(longer_word))
    )
