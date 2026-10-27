# File: evaluate.py

import csv
import time
import os
import collections
import json  # <-- For loading user_freq.json
import matplotlib.pyplot as plt # <-- For plotting

# Import all your backend logic
from trie import Trie
from corrector import correct_query

# --- File Paths ---
DATA_DIR = "data"
USER_FREQ_PATH = os.path.join(DATA_DIR, "user_freq.json")

# --- Helper functions (Copied from main.py) ---
def load_json_data(filepath):
    """Loads a JSON file. If it doesn't exist, returns an empty dictionary."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

from data_loader import load_dictionary_and_trie

def run_test(test_cases, trie, dictionary_by_length, user_frequencies):
    """
    Helper function to run a single test pass.
    This version includes the detailed PASS/FAIL report.
    """
    print("--- DETAILED TEST REPORT ---")
    correct_count = 0
    total_time_ms = 0
    test_case_count = len(test_cases)
    
    if test_case_count == 0:
        return 0, 0

    for misspelled, correct in test_cases:
        start_time = time.time()
        # Get the Top-1 suggestion
        predictions = correct_query(dictionary_by_length, trie, misspelled, user_frequencies, threshold=2, max_results=1)
        end_time = time.time()
        total_time_ms += (end_time - start_time) * 1000
        
        # --- DETAILED PRINTING ---
        if predictions and predictions[0] == correct:
            correct_count += 1
            print(f"  [PASS] Query: '{misspelled}', Correct: '{correct}', Got: '{predictions[0]}'")
        else:
            # This will show you *why* it failed (e.g., "Got: 'helo'")
            prediction_word = predictions[0] if predictions else 'NONE'
            print(f"  [FAIL] Query: '__{misspelled}__', Correct: '{correct}', Got: '__{prediction_word}__'")
        # --- END ---
            
    print("----------------------------")
    accuracy = (correct_count / test_case_count) * 100
    avg_time_ms = total_time_ms / test_case_count
    return accuracy, avg_time_ms

def run_evaluation():
    """
    Runs the evaluation, compares baseline vs. adaptive, and generates a plot.
    """
    
    # 1. Load dictionary and test cases
    trie, dictionary_by_length = load_dictionary_and_trie("english")
    if trie is None: return

    test_cases = []
    test_file = os.path.join(DATA_DIR, "test_cases.csv")
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_cases.append((row['misspelled'], row['correct']))
    except (FileNotFoundError, KeyError) as e:
        print(f"FATAL ERROR: Test file '{test_file}' not found or is malformed. Error: {e}")
        return

    print(f"\nRunning evaluation on {len(test_cases)} test cases...")

    # --- 2. Run Test 1: Baseline System ---
    print("\nRunning baseline test (without learning)...")
    # We pass an empty dict {} for user_frequencies to guarantee a "dumb" test
    baseline_accuracy, baseline_time = run_test(test_cases, trie, dictionary_by_length, {})
    print(f"  ✅ Baseline Accuracy: {baseline_accuracy:.2f}%")
    print(f"  ⏱️ Baseline Avg. Time: {baseline_time:.2f} ms")

    # --- 3. Run Test 2: Adaptive System ---
    print("\nRunning adaptive test (with user learning)...")
    # Here, we load the *real* file from disk
    user_frequencies = load_json_data(USER_FREQ_PATH)
    if not user_frequencies:
        print("  WARNING: 'user_freq.json' is empty or not found.")
        print("  The adaptive accuracy will be the same as the baseline.")
        print("  To get the 100% result, run 'main.py' and train the system first.")
    
    adaptive_accuracy, adaptive_time = run_test(test_cases, trie, dictionary_by_length, user_frequencies)
    print(f"  ✅ Adaptive Accuracy: {adaptive_accuracy:.2f}%")
    print(f"  ⏱️ Adaptive Avg. Time: {adaptive_time:.2f} ms")

    # --- 4. Generate the Comparison Graph ---
    labels = ['Baseline Accuracy', 'Adaptive Accuracy']
    values = [baseline_accuracy, adaptive_accuracy]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=['#FF6B6B', '#6BFFB8']) # Red / Green
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=12)

    plt.title('Impact of Adaptive Learning on Accuracy', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 110) # Set Y-axis from 0 to 110
    
    output_filename = "accuracy_comparison.png"
    plt.savefig(output_filename)
    
    print(f"\n📊 Comparison graph saved as '{output_filename}'")
    # plt.show() # You can uncomment this to have the graph pop up

if __name__ == "__main__":
    run_evaluation()