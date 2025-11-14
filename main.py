# File: main.py

import csv
import time
import collections
import json
import os
# NEW (Feature 5): Import deque for a fixed-size list
from collections import deque

from trie import Trie
from suggestor import autocomplete
from corrector import correct_query

# --- File Paths ---
DATA_DIR = "data"
# NEW (Feature 5): Add a path for the recent suggestions cache
RECENT_PATH = os.path.join(DATA_DIR, "recent.json")
# (Feature 2) Path for adaptive learning
USER_FREQ_PATH = os.path.join(DATA_DIR, "user_freq.json")

# --- Helper functions for JSON (Unchanged) ---
def load_json_data(filepath):
    """Loads a JSON file. If it doesn't exist, returns an empty dictionary."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_json_data(data, filepath):
    """Saves a dictionary to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# --- Global Data Loading (Using English as default for now) ---
def load_dictionary_and_trie(language="english"):
    # We are defaulting to 'english.csv' as you requested.
    filepath = os.path.join(DATA_DIR, f"{language}.csv")
    print(f"\nLoading dictionary for '{language}' from {filepath}...")
    
    trie = Trie()
    dictionary_by_length = collections.defaultdict(list)

    try:
        # Renaming 'unigram_freq.csv' to 'english.csv'
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row.get("word", "").strip().lower()
                if not word: continue
                try:
                    freq = int(row.get("count", 1))
                except (ValueError, KeyError):
                    freq = 1
                trie.insert(word, freq)
                dictionary_by_length[len(word)].append(word)
    except FileNotFoundError:
        print(f"FATAL ERROR: Language file '{filepath}' not found.")
        print("Please make sure you have a file named 'english.csv' in your 'data' folder.")
        return None, None
    
    print(f"Dictionary for '{language}' loaded successfully!")
    return trie, dictionary_by_length

# --- Main Program Logic ---
def main():
    # --- Load Data (Defaulting to English) ---
    trie, dictionary_by_length = load_dictionary_and_trie("english")
    if trie is None:
        exit()

    # --- Load Adaptive Learning Data (Feature 2) ---
    user_frequencies = load_json_data(USER_FREQ_PATH)
    print(f"Loaded {len(user_frequencies)} user-learned words.")

    # --- NEW: Load Recent Suggestions Cache (Feature 5) ---
    # We load the list of words from the JSON
    recent_list = load_json_data(RECENT_PATH).get("recent_words", [])
    # We put it in a deque (a fixed-size list). We'll keep the 10 most recent.
    recent_suggestions = deque(recent_list, maxlen=10)
    
    if recent_suggestions:
        print("\n🌟 Recent Suggestions:")
        for item in recent_suggestions:
            print(f"  - {item}")
    # --- END NEW (Feature 5) ---

    while True:
        query = input("\nEnter your search query (or 'exit'): ").strip().lower()
        if query == "exit":
            # --- NEW: Save Recent Suggestions on exit (Feature 5) ---
            save_json_data({"recent_words": list(recent_suggestions)}, RECENT_PATH)
            # --- END NEW (Feature 5) ---
            
            # Save Adaptive Learning data (Feature 2)
            save_json_data(user_frequencies, USER_FREQ_PATH)
            print(f"\nYour preferences and recent history have been saved. Goodbye!")
            break
        
        if not query:
            continue

        # --- Autocomplete (Feature 1: Tracker) ---
        start_time_auto = time.time()
        auto_suggestions = autocomplete(trie, query, k=5)
        end_time_auto = time.time()
        auto_time_ms = (end_time_auto - start_time_auto) * 1000
        print(f"\n🔍 Autocomplete Suggestions (took {auto_time_ms:.2f} ms):")
        if auto_suggestions:
            for i, word in enumerate(auto_suggestions, 1):
                print(f" {i}. {word}")
        else:
            print(" No autocomplete suggestions found.")

        # --- Spell Correction (Features 1 & 2) ---
        start_time_correct = time.time()
        # Pass user_frequencies (Feature 2)
        corrections = correct_query(dictionary_by_length, trie, query, user_frequencies, threshold=2, max_results=5)
        end_time_correct = time.time()
        correct_time_ms = (end_time_correct - start_time_correct) * 1000
        print(f"\n🛠 Spell Correction Suggestions (took {correct_time_ms:.2f} ms):")
        
        if not corrections:
            print(" No corrections found.")
        else:
            for i, word in enumerate(corrections, 1):
                print(f" {i}. {word}")

            # --- Learning Interaction (Features 2 & 5) ---
            try:
                choice_str = input("\nType the number of the correct word to learn it (or press Enter to skip): ")
                if not choice_str: continue
                choice_idx = int(choice_str) - 1
                
                if 0 <= choice_idx < len(corrections):
                    chosen_word = corrections[choice_idx]
                    
                    # 1. Add to adaptive learning (Feature 2)
                    user_frequencies[chosen_word] = user_frequencies.get(chosen_word, 0) + 1
                    print(f"✅ Learned! '{chosen_word}' now has a preference count of {user_frequencies[chosen_word]}.")

                    # --- NEW: Add to Recent Suggestions (Feature 5) ---
                    # If word is already in the list, remove it first
                    if chosen_word in recent_suggestions:
                        recent_suggestions.remove(chosen_word)
                    # Add it to the front of the list
                    recent_suggestions.appendleft(chosen_word)
                    # --- END NEW (Feature 5) ---
                else:
                    print("Invalid number. Skipping learning.")
            except (ValueError):
                print("Invalid input. Skipping learning.")

if __name__ == "__main__":
    main()