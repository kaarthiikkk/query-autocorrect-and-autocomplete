# Multilingual Search Intelligence Lab

> **A Hybrid Adaptive Ranking Framework for Multilingual Predictive Search, Autocomplete, and Query Correction**

Research-oriented multilingual search engine, predictive autocorrect, and intelligence lab featuring a **FastAPI REST API**, an interactive **Streamlit Research Laboratory Dashboard**, and automated evaluation suites.

---

## 1. Supported Languages & Input Modes

The engine supports **10 languages** across **11 selectable modes** in the user interface and REST API:

| # | Language | ISO Code | Mode Identifier | Corpus Size | Dataset Source & Provenance |
|---|---|---|---|---|---|
| **1** | English | `en` | `english` | 333,333 words | Google Web Trillion Word Corpus (`data/english.csv`) |
| **2** | Tamil (Native Script) | `ta` | `tamil` | 50,000 words | Tamil Wikipedia & Articles Frequency Corpus (`data/tamil_small.csv`) |
| **3** | Romanised Tamil (Tanglish) | `ta` | `tamil_romanized` | 146,129 words | AI4Bharat / IndicCorp Romanised Tamil Corpus (`data/tamil_romanized.csv`) |
| **4** | Spanish | `es` | `spanish` | 20,000 words | `wordfreq` (Luminoso / Exquisite Corpus, `data/spanish.csv`) |
| **5** | German | `de` | `german` | 20,000 words | `wordfreq` (Luminoso / Exquisite Corpus, `data/german.csv`) |
| **6** | French | `fr` | `french` | 20,000 words | `wordfreq` (Luminoso / Exquisite Corpus, `data/french.csv`) |
| **7** | Italian | `it` | `italian` | 20,000 words | `wordfreq` (Luminoso / Exquisite Corpus, `data/italian.csv`) |
| **8** | Portuguese | `pt` | `portuguese` | 20,000 words | `wordfreq` (Luminoso / Exquisite Corpus, `data/portuguese.csv`) |
| **9** | Dutch | `nl` | `dutch` | 20,000 words | `wordfreq` (Luminoso / Exquisite Corpus, `data/dutch.csv`) |
| **10** | Hindi | `hi` | `hindi` | Tagged / Lexicon | Transliteration bridge & Romanized lexicon |
| **11** | Telugu | `te` | `telugu` | Tagged / Lexicon | Transliteration bridge & Romanized lexicon |

### Distinction Between Tamil Script and Romanised Tamil Input
- **Tamil Native Script (`tamil`)**: Queries entered in the native Tamil Brahmi-derived Unicode script (block `\u0B80`–`\u0BFF`), e.g., `வணக்கம்` or `பிரியாணி வேண்டும்`.
- **Romanised Tamil (`tamil_romanized` / Tanglish)**: Tamil phonetically rendered into Latin characters, e.g., `sapda venum`, `briyani vemum`. The system bridges Romanised typos (`vemum` $\to$ `venum`) and provides dual Latin-script and native-script transliterations (`வேண்டும்`).

---

## 2. Dataset Sources, Representation & Licensing

- **Vocabulary Datasets**: Extracted using the Python package `wordfreq` (v3.1.1, by Robyn Speer / Luminoso Technologies).
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) and MIT License.
- **Frequency Representation**:
  - `wordfreq` provides word probability $P$ and logarithmic Zipf scores ($1.0$–$8.0$).
  - To match the existing English and Tamil integer frequency scale in `ranker.py`, frequencies are scaled deterministically as $\text{count} = \max(1, \text{round}(P \times 10^9))$.
  - Top words have counts in the tens of millions (e.g. Spanish `de`: $64,600,000$, German `die`: $30,200,000$), preserving the calibrated log-frequency contribution without dominating hybrid ranking.
- **Manifest**: Complete dataset metadata, word counts, min/max Zipf scores, and licenses are preserved in `data/multilingual_datasets_manifest.json`.

---

## 3. Installation & Setup

### Prerequisites
- Python 3.9+
- macOS / Linux / Windows

### Setup Environment
```bash
# 1. Clone repository & enter workspace
cd /path/to/search_engine_project

# 2. Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Regenerating Datasets (Offline Reproducibility)
To regenerate or inspect all six foreign language vocabulary files:
```bash
python scripts/build_multilingual_datasets.py
```

---

## 4. Running the Applications

### A. Run the REST API Server (Terminal 1)
```bash
source .venv/bin/activate
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger documentation:
👉 **`http://127.0.0.1:8000/docs`**

### B. Run the Streamlit Research Laboratory (Terminal 2)
```bash
source .venv/bin/activate
streamlit run dashboard.py --server.port 8501
```
Open in browser:
👉 **`http://localhost:8501`**

### C. Language Selection in the UI & API
- **Streamlit Dashboard**: In the sidebar under **Language Routing**, select any of the 11 options or leave on **Auto-Detect (Multilingual)**.
- **REST API**: Pass the optional `language` field in POST `/api/v1/suggest`, POST `/api/v1/correct`, or GET `/api/v1/autocomplete`.

---

## 5. Verified Examples Across All Languages

| Language | Task | Query | Top Generated Suggestions | Measured Latency |
|---|---|---|---|---|
| **English** | Autocomplete | `"pyth"` | `['python', 'pythons', 'pythagoras']` | 35.7 ms |
| **English** | Autocorrect | `"pythno"` | `['python', 'pytho', 'pythons']` | 33.3 ms |
| **Tamil Native** | Autocorrect | `"வணக்கம"` | `['வணக்கம்', 'வணக்க', 'வைக்க']` | 12.4 ms |
| **Tamil Romanised** | Autocorrect | `"briyani vemum"` | `['biryani venum', 'biriyani venum', 'briyaani venum']` | 56.7 ms |
| **Spanish** | Autocomplete | `"infor"` | `['informa', 'informó', 'informe']` | 3.4 ms |
| **Spanish** | Autocorrect | `"informacion"` | `['informaciones', 'información', 'information']` | 5.1 ms |
| **German** | Autocomplete | `"informa"` | `['information', 'informationen', 'informations']` | 3.3 ms |
| **German** | Autocorrect | `"informazion"` | `['information', 'informations']` | 5.2 ms |
| **French** | Autocomplete | `"inform"` | `['informé', 'informe', 'informer']` | 3.7 ms |
| **French** | Autocorrect | `"informasion"` | `['information', 'informations']` | 2.8 ms |
| **Italian** | Autocomplete | `"infor"` | `['informazioni', 'informato', 'informatico']` | 4.6 ms |
| **Italian** | Autocorrect | `"informazzione"` | `['informazione', 'informazioni']` | 2.2 ms |
| **Portuguese** | Autocomplete | `"infor"` | `['informar', 'informa', 'informou']` | 4.6 ms |
| **Portuguese** | Autocorrect | `"informasao"` | `['informado', 'informada', 'informação']` | 5.2 ms |
| **Dutch** | Autocomplete | `"inform"` | `['informatie', 'informeren', 'informele']` | 4.7 ms |
| **Dutch** | Autocorrect | `"informasie"` | `['informatie', 'informatief']` | 2.0 ms |

---

## 6. Automated Verification & Test Results

Run the full automated test suite using `pytest`:
```bash
pytest tests/ -v
```
**Results: 37 / 37 passed in 6.79s**
- `tests/test_api_and_features.py`: 18/18 passed
- `tests/test_multilingual_extensions.py`: 19/19 passed

Run the end-to-end regression suite:
```bash
python test_comprehensive_project.py
```
**Results: 14 / 14 passed in 1.07s (OK)**

---

## 7. Known Limitations & Scope

1. **Automatic Detection for Shared Short Words**: Words shared across European languages (e.g., `in`, `no`, `information`) without distinctive diacritics default to English when automatic detection confidence is below threshold; manual language selection is available in both the UI and REST API.
2. **ML Reranker Feature Generalization**: The 28-dimensional Random Forest reranker operates on language-agnostic signals (edit distance, prefix overlap, frequency ratio, intent compatibility); it does not compute language-specific semantic embeddings.
3. **Hindi & Telugu Vocabulary**: Transliteration rules and romanized markers exist for Hindi and Telugu, but full 20,000-word native corpora are loaded for English, Tamil, and the six European languages.
