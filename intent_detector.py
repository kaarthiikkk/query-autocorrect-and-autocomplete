"""Lightweight, explainable rule-based intent detection.

Supports multilingual queries (English, Tamil native script, Romanized, and
code-switched input) without external machine-learning models or APIs.
"""

import re


INTENT_CONFIDENCE = {
    "navigational": 0.90,
    "question": 0.90,
    "transactional": 0.85,
    "local": 0.85,
    "general_search": 0.70,
    "unknown": 0.0,
}

NAVIGATIONAL_TOKENS = {
    "youtube", "google", "gmail", "facebook", "instagram", "linkedin",
    "github", "amazon", "netflix", "wikipedia", "reddit", "twitter",
    "spotify", "yahoo", "bing", "whatsapp", "chatgpt", "quora",
    "tiktok", "pinterest", "apple", "microsoft",
}

NAVIGATIONAL_PHRASES = [
    "x.com", "sign in", "log in", "login", "portal", "official site",
]

NAVIGATIONAL_PATTERNS = [
    r"^https?://",
    r"^www\.",
    r"\.(com|org|net|edu|gov|io|co|in|ai|app|dev)(/.*)?$",
]

LOCAL_TOKENS = {
    "nearby", "nearest", "restaurant", "restaurants", "cafe", "cafes",
    "hotel", "hotels", "hospital", "hospitals", "pharmacy", "pharmacies",
    "supermarket", "supermarkets", "atm", "bakery",
    # Tamil local entities and indicators
    "மருத்துவமனை", "உணவகம்", "விடுதி", "அருகில்", "பக்கத்தில்",
}

LOCAL_PHRASES = [
    "near me", "near by", "close to me", "around me", "in my area",
    "directions to", "route to", "gas station", "petrol bunk",
]

TRANSACTIONAL_TOKENS = {
    "buy", "price", "pricing", "order", "purchase", "shop", "shopping",
    "sale", "discount", "coupon", "coupons", "deal", "deals", "cheap",
    "cheapest", "cost", "subscribe", "subscription", "hire", "rent",
    "booking",
    # Tamil transactional terms
    "வாங்க", "விலை", "விற்பனை", "ஆர்டர்",
}

TRANSACTIONAL_PHRASES = [
    "price of", "cost of", "book ticket", "book flights", "flight tickets",
]

QUESTION_STARTERS = {
    "what", "why", "when", "where", "who", "which", "how", "whose", "whom",
    "can", "could", "is", "are", "does", "do", "did", "will", "would",
    "should", "was", "were",
}

QUESTION_TOKENS = {
    # Tamil question words (can appear anywhere in query)
    "என்ன", "எப்படி", "ஏன்", "எப்போது", "எங்கே", "யார்", "எது", "எவை",
    "எவ்வளவு", "எத்தனை", "எதற்கு", "விளக்கம்", "பொருள்",
    # Romanized / Indian language question words
    "enna", "epdi", "eppadi", "ethuku", "edhuku", "yen", "eppo",
    "eppodhu", "enga", "enge", "yaar", "kaise", "kyun", "kya",
    "kahan", "kab", "ela", "elaa", "enti", "enduku",
}

QUESTION_PHRASES = [
    "how to", "how do", "how does", "how can", "what is", "what are",
    "why is", "why do", "why does", "where is", "where can", "when is",
    "who is", "which is", "meaning of", "definition of",
    "difference between", "tutorial", "guide", "explain",
]


def detect_intent(query):
    """
    Detect the primary intent of a search query.

    Returns:
        {
            "intent": str,
            "confidence": float,
            "matched_signals": list
        }

    This module is intentionally annotation-only.
    It does not modify candidate generation or ranking.
    """

    raw_query = query.strip()
    query = raw_query.lower()

    if not query:
        return {
            "intent": "unknown",
            "confidence": INTENT_CONFIDENCE["unknown"],
            "matched_signals": [],
        }

    tokens = [t.strip('.,!?:;"\'()[]{}') for t in query.split()]
    tokens = [t for t in tokens if t]
    padded_query = f" {query} "

    # --------------------------------------------------
    # 1. Navigational intent
    # --------------------------------------------------
    nav_matches = []
    for token in tokens:
        if token in NAVIGATIONAL_TOKENS:
            nav_matches.append(token)
    for phrase in NAVIGATIONAL_PHRASES:
        if f" {phrase} " in padded_query:
            nav_matches.append(phrase)
    for pat in NAVIGATIONAL_PATTERNS:
        if re.search(pat, query):
            nav_matches.append(pat)

    if nav_matches:
        return {
            "intent": "navigational",
            "confidence": INTENT_CONFIDENCE["navigational"],
            "matched_signals": nav_matches,
        }

    # --------------------------------------------------
    # 2. Local intent
    # --------------------------------------------------
    local_matches = []
    for phrase in LOCAL_PHRASES:
        if f" {phrase} " in padded_query:
            local_matches.append(phrase)
    for token in tokens:
        if token in LOCAL_TOKENS:
            local_matches.append(token)

    if local_matches:
        return {
            "intent": "local",
            "confidence": INTENT_CONFIDENCE["local"],
            "matched_signals": local_matches,
        }

    # --------------------------------------------------
    # 3. Transactional intent
    # --------------------------------------------------
    tx_matches = []
    for phrase in TRANSACTIONAL_PHRASES:
        if f" {phrase} " in padded_query:
            tx_matches.append(phrase)
    for token in tokens:
        if token in TRANSACTIONAL_TOKENS:
            tx_matches.append(token)

    if tx_matches:
        return {
            "intent": "transactional",
            "confidence": INTENT_CONFIDENCE["transactional"],
            "matched_signals": tx_matches,
        }

    # --------------------------------------------------
    # 4. Question intent
    # --------------------------------------------------
    q_matches = []
    if "?" in raw_query:
        q_matches.append("question_mark")
    if tokens and tokens[0] in QUESTION_STARTERS:
        q_matches.append(tokens[0])
    for phrase in QUESTION_PHRASES:
        if f" {phrase} " in padded_query:
            q_matches.append(phrase)
    for token in tokens:
        if token in QUESTION_TOKENS:
            q_matches.append(token)

    if q_matches:
        return {
            "intent": "question",
            "confidence": INTENT_CONFIDENCE["question"],
            "matched_signals": q_matches,
        }

    # --------------------------------------------------
    # 5. Default search
    # --------------------------------------------------
    return {
        "intent": "general_search",
        "confidence": INTENT_CONFIDENCE["general_search"],
        "matched_signals": [],
    }