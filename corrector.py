# File: corrector.py

import math

def damerau_levenshtein(s1, s2):
    # This function is unchanged
    d = {}
    len1, len2 = len(s1), len(s2)
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1
    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i-1, j)] + 1,
                d[(i, j-1)] + 1,
                d[(i-1, j-1)] + cost
            )
            if i > 0 and j > 0 and s1[i] == s2[j-1] and s1[i-1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i-2, j-2)] + cost)
    return d[(len1-1, len2-1)]

# --- UPDATED FUNCTION ---
# It now accepts 'user_freq'
def correct_query(dictionary_by_length, trie, query, user_freq, threshold=2, max_results=5):
    """
    Finds spelling corrections, balancing edit distance, base frequency,
    and user's learned preferences.
    """
    
    query_len = len(query)
    words_to_check = []
    for i in range(query_len - threshold, query_len + threshold + 1):
        if i in dictionary_by_length:
            words_to_check.extend(dictionary_by_length[i])
    
    if not words_to_check:
        return []

    candidates = []
    
    for word in words_to_check:
        distance = damerau_levenshtein(query, word)
        
        if distance <= threshold:
            node = trie.search(word)
            if node and node.is_end:
                base_freq = node.freq
                
                # --- NEW: ADAPTIVE LEARNING ---
                # Get the user's personal preference count for this word
                user_bonus_count = user_freq.get(word, 0)
                
                # Smart Score:
                # 1. Base score from distance and log of base frequency
                # 2. Adaptive Bonus: Give a large bonus for each time the user has picked this word.
                #    We subtract because lower scores are better. This bonus is very strong.
                score = distance - (0.1 * math.log(base_freq + 1)) - (user_bonus_count * 0.5)
                
                candidates.append((score, word))

    candidates.sort()
    return [word for score, word in candidates[:max_results]]