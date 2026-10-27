from error_detector import select_best_error


def analyze_query(query, trie, dictionary_by_length, correction_candidates=None):
    """
    Analyzes the query and determines whether it is:
    - a valid dictionary word
    - a valid prefix
    - a possible spelling error
    """

    query = query.strip().lower()

    if not query:
        return {
            "query": query,
            "is_valid_word": False,
            "is_valid_prefix": False,
            "query_type": "empty",
            "error_type": "none",
            "edit_distance": None,
            "candidate": None,
        }

    # Check whether the exact query is a dictionary word
    node = trie.search(query)

    is_valid_word = (
        node is not None
        and node.is_end
    )

    # Check whether the query exists as a prefix
    is_valid_prefix = (
        node is not None
    )

    # Determine query type
    if is_valid_word:
        query_type = "valid_word"

    elif is_valid_prefix:
        query_type = "valid_prefix"

    else:
        query_type = "possible_typo"

    error_analysis = (
        select_best_error(query, correction_candidates)
        if query_type == "possible_typo"
        else select_best_error(query, None)
    )

    return {
        "query": query,
        "is_valid_word": is_valid_word,
        "is_valid_prefix": is_valid_prefix,
        "query_type": query_type,
        "error_type": error_analysis["error_type"],
        "edit_distance": error_analysis["edit_distance"],
        "candidate": error_analysis["candidate"],
    }


def get_adaptive_weights(query_type):
    """
    Selects ranking weights according to query behavior.
    """

    if query_type == "valid_word":

        return {
            "prefix": 0.45,
            "edit": 0.20,
            "frequency": 0.25,
            "user": 0.10
        }

    elif query_type == "valid_prefix":

        return {
            "prefix": 0.50,
            "edit": 0.15,
            "frequency": 0.20,
            "user": 0.15
        }

    elif query_type == "possible_typo":

        return {
            "prefix": 0.15,
            "edit": 0.50,
            "frequency": 0.20,
            "user": 0.15
        }

    else:

        return {
            "prefix": 0.25,
            "edit": 0.40,
            "frequency": 0.20,
            "user": 0.15
        }
