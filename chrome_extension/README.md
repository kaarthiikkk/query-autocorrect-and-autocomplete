# 🚀 Multilingual Search Intelligence — Chrome Extension

A real-time predictive search, typo-correction, and multilingual autocomplete browser extension powered by your FastAPI backend.

---

## ⚡ Quick 60-Second Setup Guide

### Step 1: Ensure the FastAPI Backend is Running
The extension connects directly to your local FastAPI server at `http://127.0.0.1:8000`.

In your project directory terminal, start the API:
```bash
.venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
*(Verify by visiting `http://localhost:8000/docs` in your browser).*

---

### Step 2: Load the Extension into Google Chrome / Brave / Edge

1. Open Google Chrome (or Brave / Microsoft Edge).
2. In the address bar, type:
   ```text
   chrome://extensions/
   ```
   *(or `edge://extensions/` for Microsoft Edge, `brave://extensions/` for Brave).*
3. In the top right corner, toggle on **"Developer mode"**.
4. In the top left corner, click the **"Load unpacked"** button.
5. In the file picker, select the `chrome_extension` folder located at:
   ```text
   /Users/karthikreddy/Desktop/search_engine_project/chrome_extension
   ```
6. The extension card **"Multilingual Search Intelligence" (v1.0.0)** will appear in your extensions list!

---

### Step 3: Pin and Test Live

1. Click the **Puzzle Piece 🧩 (Extensions)** icon in your browser toolbar (next to the address bar).
2. Click the **Pin 📌** icon next to **"Multilingual Search Intelligence"**.
3. Click the extension icon in your toolbar to open the search popup!

---

## 🎯 Test Scenarios for Review Demonstration

| Query | What Happens | Core Engineering Feature Demonstrated |
| :--- | :--- | :--- |
| **`pythno`** | Instantly corrects to **`python`** in `< 4ms` | 5-Filter Pruned Damerau–Levenshtein & ML Reranking |
| **`pyth`** | Autocompletes to `python`, `pytho`, `pythons` | Prefix Trie Autocomplete |
| **`sapda venum`** | Resolves to **`சாப்பிட வேண்டும்`** | Romanized Tamil (Tanglish) Transliteration Lexicon |
| **`informacion`** | Autocompletes & recovers Spanish `información` | Cross-Lingual European Language Module (ES) |
| **`informazion`** | Reconstructs German `information` | Multilingual Diacritic & Phonetic Correction (DE) |

---

## 🌟 Key Features for Review Committee Demo

* **Live Keystroke Latency Badge:** Displays real-time API roundtrip latency (typically **2 to 5 ms**).
* **Click to Reinforce Feedback:** Clicking any candidate copies it to your clipboard and fires an asynchronous event to `/api/v1/feedback`, actively training the model.
* **1-Click Google Search:** Clicking the **🌐** globe icon searches for the corrected expression on Google in a new tab.
* **User Personalization:** Toggle between profiles (`developer_alex`, `student_priya`, etc.) to see suggestion ranking adapt in real time.
