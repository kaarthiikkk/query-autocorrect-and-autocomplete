# scripts/process_hf_datasets.py
# Downloads sk-community/romanized_hindi and indiehackers/tenglish_dataset
# Splits into separate native and romanized CSV files with frequencies.

import csv
import re
from collections import Counter
from datasets import load_dataset
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Regex patterns for scripts
RE_DEVANAGARI = re.compile(r'[\u0900-\u097F]+')
RE_TELUGU = re.compile(r'[\u0C00-\u0C7F]+')
RE_LATIN = re.compile(r'[a-zA-Z]+')

def clean_and_count_tokens(text, regex_pattern, min_len=2):
    tokens = []
    for match in regex_pattern.finditer(text):
        token = match.group().strip().lower()
        if len(token) >= min_len:
            tokens.append(token)
    return tokens

def process_hindi():
    print(">>> Processing sk-community/romanized_hindi...")
    ds = load_dataset("sk-community/romanized_hindi", split="train", streaming=True)
    
    devanagari_counts = Counter()
    romanized_counts = Counter()
    
    count = 0
    MAX_SAMPLES = 100000
    
    for row in ds:
        hindi_text = row.get("Hindi", "") or ""
        translit_text = row.get("Transliterated Hindi", "") or ""
        
        # Native Devanagari tokens
        for token in clean_and_count_tokens(hindi_text, RE_DEVANAGARI, min_len=2):
            devanagari_counts[token] += 1
            
        # Romanized Hinglish tokens
        for token in clean_and_count_tokens(translit_text, RE_LATIN, min_len=2):
            romanized_counts[token] += 1
            
        count += 1
        if count >= MAX_SAMPLES:
            break
            
    print(f"Extracted {len(devanagari_counts):,} unique Hindi native words.")
    print(f"Extracted {len(romanized_counts):,} unique Hindi romanized words.")
    
    # Save hindi.csv (native)
    hindi_file = os.path.join(DATA_DIR, "hindi.csv")
    with open(hindi_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        for word, freq in devanagari_counts.most_common(20000):
            writer.writerow([word, freq * 100 + 500]) # scale to comparable corpus counts
            
    # Save hindi_romanized.csv (romanized)
    hindi_rom_file = os.path.join(DATA_DIR, "hindi_romanized.csv")
    with open(hindi_rom_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        for word, freq in romanized_counts.most_common(20000):
            writer.writerow([word, freq * 100 + 500])
            
    print(f"✓ Saved {hindi_file} and {hindi_rom_file}")

def process_telugu():
    print("\n>>> Processing indiehackers/tenglish_dataset (telugu_asr)...")
    ds = load_dataset("indiehackers/tenglish_dataset", "telugu_asr", split="train", streaming=True)
    
    telugu_counts = Counter()
    tenglish_counts = Counter()
    
    count = 0
    MAX_SAMPLES = 100000
    
    for row in ds:
        native_text = row.get("text", "") or ""
        translit_text = row.get("translit", "") or ""
        
        # Native Telugu tokens
        for token in clean_and_count_tokens(native_text, RE_TELUGU, min_len=2):
            telugu_counts[token] += 1
            
        # Romanized Tenglish tokens
        for token in clean_and_count_tokens(translit_text, RE_LATIN, min_len=2):
            tenglish_counts[token] += 1
            
        count += 1
        if count >= MAX_SAMPLES:
            break
            
    print(f"Extracted {len(telugu_counts):,} unique Telugu native words.")
    print(f"Extracted {len(tenglish_counts):,} unique Telugu romanized words.")
    
    # Save telugu.csv (native)
    telugu_file = os.path.join(DATA_DIR, "telugu.csv")
    with open(telugu_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        for word, freq in telugu_counts.most_common(20000):
            writer.writerow([word, freq * 100 + 500])
            
    # Save telugu_romanized.csv (romanized)
    telugu_rom_file = os.path.join(DATA_DIR, "telugu_romanized.csv")
    with open(telugu_rom_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        for word, freq in tenglish_counts.most_common(20000):
            writer.writerow([word, freq * 100 + 500])
            
    print(f"✓ Saved {telugu_file} and {telugu_rom_file}")

if __name__ == "__main__":
    process_hindi()
    process_telugu()
    print("\nAll datasets split and exported successfully!")
