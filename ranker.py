import math

from corrector import damerau_levenshtein


DEFAULT_HYBRID_WEIGHTS = {
    "lexical": 0.25,
    "edit": 0.20,
    "frequency": 0.15,
    "semantic": 0.15,
    "context": 0.10,
    "intent": 0.05,
    "user": 0.10,
}


class PurePythonSemanticModel:
    """Lightweight, deterministic multilingual subword embedding model.

    Projects text into a subword character n-gram and semantic cluster
    vector space, computing normalized cosine similarity without requiring
    external machine-learning dependencies.
    """

    def __init__(self, dim=128):
        self.dim = dim
        self.semantic_clusters = {
            "tech": {
                "python", "machine", "learning", "code", "programming",
                "software", "developer", "algorithm", "data", "computer",
                "ai", "pythno", "pytho", "phython", "பைதான்", "நிரலாக்கம்"
            },
            "media": {
                "youtube", "video", "watch", "music", "song", "stream",
                "audio", "media", "பாடல்", "படம்"
            },
            "commerce": {
                "buy", "price", "pricing", "order", "purchase", "shop",
                "sale", "iphone", "ipohne", "laptop", "cheap", "cost", "deal",
                "discount", "coupon", "வாங்க", "விலை", "விற்பனை"
            },
            "food": {
                "restaurant", "restaurants", "food", "biryani", "briyani", "cafe",
                "hotel", "eat", "dinner", "lunch", "சாப்பாடு", "உணவகம்",
                "பிரியாணி", "venum", "vendum", "வேண்டும்", "vemum"
            },
            "location": {
                "near", "nearby", "location", "map", "directions",
                "place", "area", "அருகில்", "பக்கத்தில்"
            },
        }

    def _hash_token(self, text):
        h = 0
        for char in text:
            h = (h * 31 + ord(char)) & 0xFFFFFFFF
        return h

    def get_vector(self, text):
        text = text.strip().lower()
        if not text:
            return [0.0] * self.dim

        vec = [0.0] * self.dim
        words = text.split()

        for word in words:
            # 1. Subword character n-grams
            padded = f"<{word}>"
            for n in (2, 3, 4):
                if len(padded) >= n:
                    for i in range(len(padded) - n + 1):
                        ngram = padded[i:i + n]
                        idx = self._hash_token(ngram) % self.dim
                        vec[idx] += 1.0

            # 2. Semantic cluster anchor projections
            for c_idx, (cluster_name, cluster_words) in enumerate(self.semantic_clusters.items()):
                if word in cluster_words:
                    cluster_dim_start = (c_idx * 16) % self.dim
                    for k in range(16):
                        vec[(cluster_dim_start + k) % self.dim] += 2.0

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def compute_similarity(self, text1, text2):
        if not text1 or not text2:
            return 0.0
        v1 = self.get_vector(text1)
        v2 = self.get_vector(text2)
        dot = sum(a * b for a, b in zip(v1, v2))
        return max(0.0, min(1.0, dot))


_GLOBAL_SEMANTIC_MODEL = PurePythonSemanticModel()


def calculate_semantic_similarity(query, candidate_word):
    """Return normalized semantic similarity in [0.0, 1.0]."""
    return _GLOBAL_SEMANTIC_MODEL.compute_similarity(query, candidate_word)


def calculate_intent_compatibility(intent, candidate):
    """Return explainable intent score in [0.0, 1.0] for a candidate."""
    if not intent or intent == "unknown":
        return 0.0

    word = candidate.get("word", "").lower()
    source = candidate.get("source", "")

    if intent == "navigational":
        if word in PurePythonSemanticModel().semantic_clusters["media"] or source == "trie_autocomplete":
            return 1.0
        return 0.60

    if intent == "transactional":
        if word in PurePythonSemanticModel().semantic_clusters["commerce"]:
            return 1.0
        return 0.70

    if intent == "local":
        if word in PurePythonSemanticModel().semantic_clusters["food"] or word in PurePythonSemanticModel().semantic_clusters["location"]:
            return 1.0
        return 0.70

    if intent == "question":
        return 0.80

    if intent == "general_search":
        return 0.50

    return 0.50


def calculate_prefix_similarity(query, word):
    """
    Measures how closely the candidate matches the query prefix.
    Returns a value between 0 and 1.
    """

    if not query or not word:
        return 0.0

    query = query.lower()
    word = word.lower()

    matching_characters = 0

    max_length = min(len(query), len(word))

    for i in range(max_length):
        if query[i] == word[i]:
            matching_characters += 1
        else:
            break

    return matching_characters / len(query)


def calculate_frequency_score(frequency):
    """
    Converts word frequency into a normalized logarithmic score.
    """

    if frequency <= 0:
        return 0.0

    return math.log(frequency + 1)


def calculate_candidate_score(
    query,
    candidate,
    trie,
    user_freq,
    weights=None
):
    """
    Calculates the single-language hybrid ranking score using:

    - Prefix similarity
    - Edit-distance similarity
    - Word frequency
    - User preference
    """

    if weights is None:
        weights = {
            "prefix": 0.35,
            "edit": 0.30,
            "frequency": 0.20,
            "user": 0.15
        }

    word = candidate["word"]

    # 1. Prefix similarity
    prefix_score = calculate_prefix_similarity(
        query,
        word
    )

    # 2. Edit-distance similarity
    distance = damerau_levenshtein(
        query,
        word
    )

    max_length = max(
        len(query),
        len(word),
        1
    )

    edit_score = 1 - (
        distance / max_length
    )

    edit_score = max(
        0.0,
        edit_score
    )

    # 3. Word frequency
    node = trie.search(word) if trie else None

    if node and node.is_end:
        frequency = node.freq
    else:
        frequency = 1

    frequency_score = calculate_frequency_score(
        frequency
    )

    # Normalize roughly into 0–1
    frequency_score = frequency_score / (
        frequency_score + 10
    )

    # 4. User preference
    user_count = user_freq.get(
        word,
        0
    ) if user_freq else 0

    user_score = user_count / (
        user_count + 10
    )

    # 5. Final hybrid score
    final_score = (
        weights["prefix"] * prefix_score
        + weights["edit"] * edit_score
        + weights["frequency"] * frequency_score
        + weights["user"] * user_score
    )

    return {
        "word": word,
        "source": candidate["source"],
        "prefix_score": prefix_score,
        "edit_score": edit_score,
        "frequency_score": frequency_score,
        "user_score": user_score,
        "final_score": final_score
    }


def calculate_hybrid_candidate_score(
    query,
    candidate,
    language_data,
    user_freq,
    intent="unknown",
    weights=None
):
    """
    Calculates Feature 9 Semantic + Lexical Hybrid Score across languages.
    """
    if weights is None:
        weights = DEFAULT_HYBRID_WEIGHTS

    word = candidate["word"]
    token = candidate.get("token") or query
    language = candidate.get("language", "english")
    # 2. Edit-distance similarity
    if candidate.get("edit_distance") is not None:
        distance = candidate["edit_distance"]
    elif token == word or query == word or word.startswith(token) or candidate.get("source") == "romanized_lexicon":
        distance = 0
    else:
        distance = damerau_levenshtein(token, word)

    max_len = max(len(token), len(word), 1)
    edit_score = max(0.0, min(1.0, 1.0 - (distance / max_len)))

    # 1. Lexical / Prefix similarity
    prefix_sim = calculate_prefix_similarity(token, word)
    if query != token:
        prefix_sim = max(prefix_sim, calculate_prefix_similarity(query, word))

    # Multi-token token-aligned lexical similarity
    if " " in query and " " in word:
        q_tokens = query.split()
        w_tokens = word.split()
        if len(q_tokens) == len(w_tokens) and q_tokens:
            token_scores = []
            for qt, wt in zip(q_tokens, w_tokens):
                p_sim = calculate_prefix_similarity(qt, wt)
                e_dist = damerau_levenshtein(qt, wt)
                e_sim = max(0.0, 1.0 - e_dist / max(len(qt), len(wt), 1))
                token_scores.append(max(p_sim, e_sim))
            phrase_lexical = sum(token_scores) / len(token_scores)
            prefix_sim = max(prefix_sim, phrase_lexical)

    if distance <= 1 or (" " in query and " " in word):
        lexical_score = max(prefix_sim, edit_score)
    else:
        lexical_score = max(0.0, min(1.0, prefix_sim))

    # 3. Word frequency
    trie = language_data.get(language, (None, None))[0] if language_data else None
    rom_trie = language_data.get("tamil_romanized", (None, None))[0] if language_data else None
    if trie:
        node = trie.search(word)
        if node and node.is_end:
            frequency = node.freq
        elif rom_trie and rom_trie.search(word) and rom_trie.search(word).is_end:
            frequency = rom_trie.search(word).freq
        else:
            frequency = candidate.get("frequency", 1)
    elif rom_trie and rom_trie.search(word) and rom_trie.search(word).is_end:
        frequency = rom_trie.search(word).freq
    else:
        frequency = candidate.get("frequency", 1)

    raw_freq = calculate_frequency_score(frequency)
    frequency_score = raw_freq / (raw_freq + 10)

    # 4. Semantic similarity
    sem_sim = calculate_semantic_similarity(query, word)
    if token != query:
        sem_sim = max(sem_sim, calculate_semantic_similarity(token, word))
    semantic_score = max(0.0, min(1.0, sem_sim))

    # 5. Context score (from Feature 3)
    context_score = min(1.0, candidate.get("context_score", 0.0))

    # 6. Intent compatibility (from Feature 8)
    intent_score = calculate_intent_compatibility(intent, candidate)

    # 7. User preference
    user_count = user_freq.get(word, 0) if user_freq else 0
    user_score = user_count / (user_count + 10)

    # Combined hybrid score
    hybrid_score = (
        weights["lexical"] * lexical_score
        + weights["edit"] * edit_score
        + weights["frequency"] * frequency_score
        + weights["semantic"] * semantic_score
        + weights["context"] * context_score
        + weights["intent"] * intent_score
        + weights["user"] * user_score
    )

    # Multi-token provenance bonus
    provenance = candidate.get("provenance", [])
    if len(provenance) > 1:
        hybrid_score += 0.05 * (len(provenance) - 1)

    # Preserving Feature 7 protection relevance
    if candidate.get("should_protect"):
        hybrid_score += 0.05

    # Script consistency bonus: prioritize same-script completion/correction for Latin/Romanized inputs
    is_q_latin = all(ord(c) < 128 or ('\u00C0' <= c <= '\u024F') for c in query.replace(" ", ""))
    is_w_latin = all(ord(c) < 128 or ('\u00C0' <= c <= '\u024F') for c in word.replace(" ", ""))
    is_w_tamil = any("\u0B80" <= c <= "\u0BFF" for c in word)
    is_q_tamil = any("\u0B80" <= c <= "\u0BFF" for c in query)


    if (is_q_latin and is_w_latin) or (is_q_tamil and is_w_tamil):
        hybrid_score += 0.05
    elif is_w_latin and is_w_tamil:
        hybrid_score -= 0.05

    if candidate.get("source") == "romanized_correction":
        hybrid_score += 0.08

    hybrid_score = round(hybrid_score, 4)

    result = dict(candidate)
    result.update({
        "intent": intent,
        "lexical_score": round(lexical_score, 4),
        "prefix_score": round(lexical_score, 4),
        "edit_score": round(edit_score, 4),
        "frequency_score": round(frequency_score, 4),
        "semantic_score": round(semantic_score, 4),
        "context_score": round(context_score, 4),
        "intent_score": round(intent_score, 4),
        "user_score": round(user_score, 4),
        "hybrid_score": hybrid_score,
        "global_score": hybrid_score,
        "ranking_components": {
            "lexical": round(lexical_score, 4),
            "edit": round(edit_score, 4),
            "frequency": round(frequency_score, 4),
            "semantic": round(semantic_score, 4),
            "context": round(context_score, 4),
            "intent": round(intent_score, 4),
            "user": round(user_score, 4),
        }
    })
    return result


def rank_multilingual_candidates(
    query,
    candidates,
    language_data,
    user_freq,
    intent="unknown",
    top_k=None,
    weights=None
):
    """
    Ranks the multilingual candidate pool using Semantic + Lexical Hybrid Scoring.

    Deterministic tie-breaking:
    1. hybrid_score descending
    2. context_adjusted_score descending
    3. edit_distance ascending (0 for exact/autocomplete)
    4. word alphabetically ascending
    """
    scored_candidates = [
        calculate_hybrid_candidate_score(
            query,
            candidate,
            language_data,
            user_freq,
            intent=intent,
            weights=weights
        )
        for candidate in candidates
    ]

    def _tie_break_key(item):
        h_score = item.get("hybrid_score", 0.0)
        ctx_adj = item.get("context_adjusted_score", item.get("context_score", 0.0))
        edit_dist = item.get("edit_distance")
        if edit_dist is None:
            edit_dist = 0 if item.get("source") in {"trie_autocomplete", "romanized_lexicon"} else 99
        word = item.get("word", "")
        return (-h_score, -ctx_adj, edit_dist, word)

    scored_candidates.sort(key=_tie_break_key)

    if top_k is not None:
        return scored_candidates[:top_k]
    return scored_candidates


def rank_candidates(
    query,
    candidates,
    trie,
    user_freq,
    top_k=5,
    weights=None
):
    """
    Ranks the hybrid candidate pool and
    returns the top-k candidates.
    """

    ranked_candidates = []

    for candidate in candidates:

        scored_candidate = calculate_candidate_score(
            query,
            candidate,
            trie,
            user_freq,
            weights
        )

        ranked_candidates.append(
            scored_candidate
        )

    # Highest score first
    ranked_candidates.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return ranked_candidates[:top_k]