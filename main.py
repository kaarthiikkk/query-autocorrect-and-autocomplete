import csv
import os
import sys
import time

from trie import Trie
from query_analyzer import analyze_query
from hybrid import generate_hybrid_candidates
from ranker import rank_multilingual_candidates
from dynamic_weights import get_dynamic_weights
from feedback_learner import get_feedback_learner

ENGLISH_DICT_PATH = "data/english.csv"
TAMIL_DICT_PATH = "data/tamil_small.csv"


def load_dictionary_from_csv(filepath):
    trie = Trie()
    dictionary_by_length = {}
    total_words = 0

    if not os.path.exists(filepath):
        print(f"Error: Data file not found at '{filepath}'")
        return trie, dictionary_by_length

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                word = row[0].strip().lower()
                freq = int(row[1]) if len(row) > 1 and row[1].isdigit() else 1

                if word:
                    trie.insert(word, frequency=freq)
                    length = len(word)
                    dictionary_by_length.setdefault(length, []).append(word)
                    total_words += 1

        print(f"Dictionary for '{os.path.basename(filepath)}' loaded successfully!")
        print(f"Total words loaded: {total_words}\n")
    except Exception as e:
        print(f"Error loading dictionary file '{filepath}': {e}")

    return trie, dictionary_by_length


def detect_language(query):
    query = query.strip()
    if not query:
        return "english", "native_script"

    tamil_chars = sum(1 for char in query if '\u0B80' <= char <= '\u0BFF')

    if tamil_chars > 0 and len(query.replace(" ", "")) == tamil_chars:
        return "tamil", "native_script"
    elif tamil_chars > 0:
        return "mixed", "code_switched"

    romanized_tamil_markers = {"iruku", "irruku", "irukka", "sapda", "saapadu", "venum", "kammal", "enga"}
    words = set(query.lower().split())
    if words.intersection(romanized_tamil_markers):
        return "tamil", "romanized"

    return "english", "native_script"


def detect_intent(query):
    q = query.lower().strip()
    if any(k in q for k in ["buy", "price", "cheap", "cost", "order", "discount"]):
        return "transactional"
    if any(k in q for k in ["how", "what", "why", "where", "who", "code"]):
        return "question"
    if any(k in q for k in ["near", "nearby", "location", "hotel", "restaurant", "map"]):
        return "local"
    if any(k in q for k in ["youtube", "facebook", "login", "portal", "yt"]):
        return "navigational"
    return "general_search"


def calculate_confidence(ranked_suggestions):
    if not ranked_suggestions:
        return "low", 0.0

    top_score = ranked_suggestions[0].get("hybrid_score", ranked_suggestions[0].get("final_score", 0.0))
    if top_score >= 0.80:
        return "high", round(top_score, 3)
    elif top_score >= 0.50:
        return "medium", round(top_score, 3)
    return "low", round(top_score, 3)


def main():
    print("\nLoading multilingual resources...\n")

    print(f"Loading dictionary for 'english' from {ENGLISH_DICT_PATH}...")
    eng_trie, eng_dict_by_length = load_dictionary_from_csv(ENGLISH_DICT_PATH)

    print(f"Loading dictionary for 'tamil' from {TAMIL_DICT_PATH}...")
    tam_trie, tam_dict_by_length = load_dictionary_from_csv(TAMIL_DICT_PATH)

    language_data = {
        "english": (eng_trie, eng_dict_by_length),
        "tamil": (tam_trie, tam_dict_by_length),
    }

    learner = get_feedback_learner()
    user_freq = learner.get_user_frequencies()
    print(f"Loaded {len(user_freq)} user preferences.\n")
    print("Adaptive Multilingual Autocorrect System\n")

    while True:
        try:
            raw_input_query = input("Enter your search query (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting system...")
            sys.exit(0)

        if raw_input_query.lower() in {"exit", "quit"}:
            print("Exiting system...")
            break

        if not raw_input_query:
            continue

        start_time = time.time()

        lang, lang_type = detect_language(raw_input_query)
        intent = detect_intent(raw_input_query)

        # Choose primary trie for the detected language
        active_trie, active_dict_by_length = language_data.get(lang, (eng_trie, eng_dict_by_length))

        print(f"\nDetected Language: {lang}")
        print(f"Query Type: {lang_type}")
        print(f"Intent: {intent}")

        candidates = generate_hybrid_candidates(
            trie=active_trie,
            dictionary_by_length=active_dict_by_length,
            query=raw_input_query,
            user_freq=learner.get_user_frequencies(),
            threshold=2,
            language_data=language_data
        )

        dynamic_w = get_dynamic_weights(
            query=raw_input_query,
            intent=intent,
            language=lang,
            language_type=lang_type,
            candidates=candidates
        )

        final_ranked = rank_multilingual_candidates(
            query=raw_input_query,
            candidates=candidates,
            language_data=language_data,
            user_freq=learner.get_user_frequencies(),
            intent=intent,
            top_k=5,
            weights=dynamic_w["weights"]
        )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        # ----------------------------------------------------------------
        # Display — show full phrase (display_suggestion) when available
        # ----------------------------------------------------------------

        def _display(item):
            return item.get("display_suggestion") or item.get("full_query") or item["word"]

        autocomplete_items = [c for c in candidates if c.get("source") in {"autocomplete", "hybrid"}]
        print(f"\nAutocomplete ({elapsed_ms} ms):")
        if autocomplete_items:
            for i, item in enumerate(autocomplete_items[:5], 1):
                print(f" {i}. {_display(item)}")
        else:
            print(" No autocomplete suggestions found.")

        correction_items = [c for c in candidates if c.get("source") in {"correction", "damerau_levenshtein"}]
        print("\nSpell Corrections:")
        if correction_items:
            for i, item in enumerate(correction_items[:5], 1):
                print(f" {i}. {_display(item)}")
        else:
            print(" No spelling corrections needed.")

        print("\nFinal Ranked Suggestions:")
        if final_ranked:
            for i, item in enumerate(final_ranked, 1):
                print(f" {i}. {_display(item)}")
        else:
            print(" No ranked suggestions available.")

        conf_level, conf_score = calculate_confidence(final_ranked)
        print(f"\nConfidence: {conf_level}")
        print(f"Confidence Score: {conf_score}\n")

        # ----------------------------------------------------------------
        # Feedback handler — accepts index number, word string, or Enter
        # ----------------------------------------------------------------

        feedback_input = input("Select a suggestion (number or word, Enter to skip): ").strip()

        if not feedback_input:
            print()
            continue

        # Determine chosen word
        chosen_word = None
        if feedback_input.isdigit():
            idx = int(feedback_input) - 1
            if 0 <= idx < len(final_ranked):
                chosen_word = _display(final_ranked[idx])
            else:
                print(f" Invalid index. Please enter 1–{len(final_ranked)}.\n")
                continue
        else:
            chosen_word = feedback_input

        rejected = [_display(item) for item in final_ranked if _display(item) != chosen_word]
        learner.record_interaction(
            query=raw_input_query,
            chosen_word=chosen_word,
            rejected_words=rejected,
            interaction_type="select"
        )
        from temporal_ranker import get_temporal_ranker
        get_temporal_ranker().record_selection(chosen_word, query=raw_input_query)
        from personalized_ranker import get_personalized_ranker
        get_personalized_ranker().record_user_interaction(
            user_id="default",
            query=raw_input_query,
            chosen_word=chosen_word,
            rejected_words=rejected,
            intent=intent,
            language=lang
        )
        print(f"Recorded preference for: '{chosen_word}'\n")


if __name__ == "__main__":
    main()