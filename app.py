# File: app.py

from flask import Flask, render_template, request, jsonify
import csv
import collections
import time
import os

# Import all your backend logic
from trie import Trie
from suggestor import autocomplete
from corrector import correct_query

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Global Data Loading ---
# We load the dictionary and trie once when the server starts.
DATA_DIR = "data"

def load_dictionary_and_trie(language="english"):
    """Loads the dictionary and trie into memory."""
    filepath = os.path.join(DATA_DIR, f"{language}.csv")
    print(f"\nLoading dictionary for '{language}' from {filepath}...")
    
    trie = Trie()
    dictionary_by_length = collections.defaultdict(list) 

    try:
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
        return None, None
    
    print(f"Dictionary for '{language}' loaded successfully!")
    return trie, dictionary_by_length

# Load the data globally
trie, dictionary_by_length = load_dictionary_and_trie("english")

# --- Web Page Route ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    # This tells Flask to find 'index.html' in the 'templates' folder
    return render_template('index.html')

# --- API Endpoint ---
@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    """API endpoint to get suggestions for a query."""
    data = request.get_json()
    query = data.get('query', '').strip().lower()

    if not query:
        return jsonify({'autocomplete': [], 'corrections': []})

    # --- Run Your Backend Code ---
    
    # 1. Get Autocomplete
    auto_suggestions = autocomplete(trie, query, k=5)
    
    # 2. Get Corrections
    # For a web app, we don't have user sessions, so we pass an
    # empty user_freq dict. This is a simplification for the demo.
    user_freq = {} 
    corrections = correct_query(dictionary_by_length, trie, query, user_freq, threshold=2, max_results=5)
    
    # Return both lists as JSON
    return jsonify({
        'autocomplete': auto_suggestions,
        'corrections': corrections
    })

# --- Run the App ---

