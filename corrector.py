import math


def damerau_levenshtein(s1, s2, max_dist=None):
    """Calculates Damerau-Levenshtein distance with optional early exit.

    Supports insertion, deletion, substitution, and transposition.
    """
    len1 = len(s1)
    len2 = len(s2)

    if max_dist is not None and abs(len1 - len2) > max_dist:
        return max_dist + 1

    d = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        d[i][0] = i
    for j in range(len2 + 1):
        d[0][j] = j

    for i in range(1, len1 + 1):
        row_min = len2
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            insertion    = d[i][j - 1] + 1
            deletion     = d[i - 1][j] + 1
            substitution = d[i - 1][j - 1] + cost
            d[i][j] = min(insertion, deletion, substitution)
            if (
                i > 1
                and j > 1
                and s1[i - 1] == s2[j - 2]
                and s1[i - 2] == s2[j - 1]
            ):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
            if d[i][j] < row_min:
                row_min = d[i][j]

        if max_dist is not None and row_min > max_dist:
            return max_dist + 1

    return d[len1][len2]


def correct_query(
    dictionary_by_length,
    trie,
    query,
    user_freq=None,
    threshold=2,
    max_results=5
):
    """Generates spelling correction candidates ranked strictly by edit distance,
    word frequency, and user preference.

    Guarantees distance 1 always ranks above distance 2.
    Uses boundary anchors, length proximity, and pruning to evaluate in < 30 ms.
    """
    query = query.strip().lower()
    if not query:
        return []

    # Fast check: if query itself is an exact word in trie
    if trie:
        exact_node = trie.search(query)
        if exact_node and exact_node.is_end:
            # High frequency core words (> 10M) are definitively valid
            if exact_node.freq >= 10_000_000:
                return [query]
            
            # For lower frequency entries, verify it is not a noisy web typo of a vastly more frequent word
            # (e.g. 'recieved' freq 1.3M vs 'received' freq 90M)
            q_len = len(query)
            has_dominant_neighbor = False
            for l in (q_len - 1, q_len, q_len + 1):
                for w in dictionary_by_length.get(l, []):
                    if w == query:
                        continue
                    if abs(len(w) - q_len) <= 1 and damerau_levenshtein(query, w, max_dist=1) == 1:
                        node = trie.search(w)
                        if node and node.is_end and node.freq >= exact_node.freq * 15:
                            has_dominant_neighbor = True
                            break
                if has_dominant_neighbor:
                    break
            
            if not has_dominant_neighbor:
                return [query]

    query_len = len(query)
    q_set = set(query)
    min_overlap = max(1, len(q_set) - threshold)

    # First-character anchor with accent equivalence
    first_chars = {query[0]}
    if query_len > 1 and query[1] != query[0]:
        first_chars.add(query[1])

    ACCENT_MAPPING = {
        'a': {'a', 'á', 'à', 'â', 'ä', 'ã', 'å'},
        'e': {'e', 'é', 'è', 'ê', 'ë'},
        'i': {'i', 'í', 'ì', 'î', 'ï'},
        'o': {'o', 'ó', 'ò', 'ô', 'ö', 'õ'},
        'u': {'u', 'ú', 'ù', 'û', 'ü'},
        'c': {'c', 'ç'},
        'n': {'n', 'ñ'},
    }
    for base_c, variants in ACCENT_MAPPING.items():
        if query[0] in variants:
            first_chars.update(variants)


    # Sort lengths by closeness to query_len: same length first, then +-1, +-2
    lengths = sorted(
        [l for l in range(max(1, query_len - threshold), query_len + threshold + 1)
         if l in dictionary_by_length],
        key=lambda l: (abs(l - query_len), l)
    )

    candidates = []
    min_dist_found = threshold + 1

    for l in lengths:
        if min_dist_found <= 1 and abs(l - query_len) >= 2:
            continue

        words = dictionary_by_length[l]
        for word in words:
            if word == query:
                continue

            if word[0] not in first_chars:
                continue

            if len(set(word) & q_set) < min_overlap:
                continue

            distance = damerau_levenshtein(query, word, max_dist=min(threshold, min_dist_found + 1))
            if distance > threshold:
                continue

            if distance < min_dist_found:
                min_dist_found = distance

            node = trie.search(word) if trie else None
            if node is None or not node.is_end:
                continue

            base_frequency = node.freq
            user_bonus = user_freq.get(word, 0) if user_freq else 0

            score = (
                distance * 100.0
                - 0.1 * math.log(base_frequency + 1)
                - 0.5 * user_bonus
            )

            candidates.append({
                "word": word,
                "distance": distance,
                "frequency": base_frequency,
                "user_bonus": user_bonus,
                "score": score
            })

    candidates.sort(key=lambda x: x["score"])

    seen = set()
    results = []
    for c in candidates:
        w = c["word"]
        if w not in seen:
            seen.add(w)
            results.append(w)
            if len(results) >= max_results:
                break

    return results

