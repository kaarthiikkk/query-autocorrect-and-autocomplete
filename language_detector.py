def detect_language(query):
    """
    Detects whether a query contains English, Tamil, Hindi, or Telugu characters.

    Returns:
        'english'
        'tamil'
        'hindi'
        'telugu'
        'mixed'
        'unknown'
    """

    has_english = False
    has_tamil = False
    has_hindi = False
    has_telugu = False

    for char in query:
        # Latin alphabet including accented characters
        if ('a' <= char.lower() <= 'z') or ('\u00C0' <= char <= '\u024F'):
            has_english = True

        # Tamil Unicode range
        elif '\u0B80' <= char <= '\u0BFF':
            has_tamil = True

        # Devanagari (Hindi) Unicode range
        elif '\u0900' <= char <= '\u097F':
            has_hindi = True

        # Telugu Unicode range
        elif '\u0C00' <= char <= '\u0C7F':
            has_telugu = True

    detected_scripts = sum([bool(has_tamil), bool(has_hindi), bool(has_telugu)])
    if has_english and detected_scripts > 0:
        return "mixed"
    if detected_scripts > 1:
        return "mixed"

    if has_hindi:
        return "hindi"
    elif has_telugu:
        return "telugu"
    elif has_tamil:
        return "tamil"
    elif has_english:
        return "english"
    else:
        return "unknown"