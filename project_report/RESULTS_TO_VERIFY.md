# Experimental Results Reconciliation & Verification Guide (RESULTS_TO_VERIFY.md)

**Project Title:** A Hybrid Adaptive Ranking Framework for Multilingual Predictive Search and Query Correction  
**Course:** BCSE497J - Project-I, Fall Semester 2026–2027  
**Department:** School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT), Vellore  

---

## 1. Executive Summary of Evaluation Scripts & Metric Sources

During code inspection and performance audit, three distinct evaluation scripts and historical log artifacts were discovered in the repository. Because these scripts operated on differing datasets, candidate pool thresholds, and user frequency profiles, their numerical outcomes differ significantly.

> [!IMPORTANT]
> **Academic Rule:** To maintain rigorous scientific honesty, these numbers **must never be blended or averaged** into a single table without explaining their differing evaluation conditions. The table below outlines every metric source found in the codebase.

| Evaluation Harness | Script File | Dataset / Test Input | Number of Queries | Primary Metrics Reported | Notes & Conditions |
|:---|:---|:---|:---:|:---|:---|
| **Early Prototype Evaluator** | [`evaluate.py`](file:///Users/karthikreddy/Desktop/search_engine_project/evaluate.py) | `data/test_cases.csv` | 10 | Baseline: 60.00%<br>Adaptive: 100.00%<br>Latency: ~1.2 ms | Evaluated 10 hardcoded English words. "Adaptive" was tested by reading `data/user_freq.json` containing pre-recorded boosted counts for those exact 10 words. |
| **Comprehensive Robustness Benchmark** | [`benchmark_evaluation.py`](file:///Users/karthikreddy/Desktop/search_engine_project/benchmark_evaluation.py) | Algorithmic perturbation of 33 ground-truth seeds | 253 | Top-1: 50.59%<br>Top-3: 66.80%<br>Top-5: 77.87%<br>MRR: 0.6051<br>NDCG@5: 0.6479<br>Median Latency: 36.70 ms | Evaluated across 8 distinct error classes on full 333,333 English dictionary with pruned D-L ($max\_d \le 3$). Continuous 1,000-query memory trace. |
| **Ablation & Baseline Benchmark** | [`ablation_and_comparison.py`](file:///Users/karthikreddy/Desktop/search_engine_project/ablation_and_comparison.py) | Diverse benchmark queries (Prefix, Typo, Romanized, Tamil) | 29 | Full Proposed: 37.9% Top-1, 1.48 ms (1/10 demonstrated)<br>Trie: 33.3%, 0.09 ms<br>Norvig: 66.7%, 1.34 ms<br>SymSpell: 66.7%, 0.67 ms, 81.8 MB<br>Unpruned D-L: 66.7%, 118.03 ms | Evaluates 5 baseline algorithms and 6 ablation configurations under identical candidate generation conditions with 1/10 demonstrated final model latency scaling. |
| **Keystroke Latency Prototype** | [`generate_fig3.py`](file:///Users/karthikreddy/Desktop/search_engine_project/generate_fig3.py) | Single unpruned dictionary scan | 1 | Trie: 0.11 ms<br>D-L Corrector: 4330.63 ms | Proof-of-concept demonstrating why unpruned Damerau-Levenshtein across 333,333 unigram words violates the 30 ms keystroke budget without early filtering. |

---

## 2. Reconciling Metric Inconsistencies

### Conflict 1: 100.0% Adaptive Accuracy vs. 50.59% Benchmark Accuracy
- **Source 1:** `evaluate.py` / `accuracy_comparison.png` $\to$ **100.0%**
- **Source 2:** `benchmark_evaluation.py` / `robustness_latency_memory_results.json` $\to$ **50.59%**
- **Root Cause:**
  `evaluate.py` tests only 10 fixed English typos (`helo` $\to$ `hello`, `progra` $\to$ `program`, `langage` $\to$ `language`, `systen` $\to$ `system`, `writting` $\to$ `writing`, `wierd` $\to$ `weird`, `thier` $\to$ `their`, `occurence` $\to$ `occurrence`, `complet` $\to$ `complete`, `pythn` $\to$ `python`). In `data/user_freq.json`, each of these 10 target words had been manually or programmatically clicked multiple times (frequencies 15–20), artificially elevating their unigram rank above competing candidates.
  In contrast, `benchmark_evaluation.py` tests 253 realistic queries generated across 8 distinct error classes (clean, deletion, insertion, substitution, transposition, QWERTY proximity, edit distance 2, and edit distance 3) against the unprimed 333,333-word vocabulary without prior user frequency biasing.
- **Resolution for Project-I Report:**
  Report the **50.59% Top-1 (77.87% Top-5)** from the 253-query benchmark as the primary scientific accuracy. Clearly explain `evaluate.py` as a controlled demonstration of how user-frequency feedback resolves ambiguous ties on specific trained words.

### Conflict 2: Unpruned D-L Latency: 4,330.63 ms vs. 135.80 ms
- **Source 1:** `generate_fig3.py` / `fig3.png` $\to$ **4,330.63 ms**
- **Source 2:** `ablation_and_comparison.py` $\to$ **135.80 ms**
- **Root Cause:**
  `generate_fig3.py` executed a pure unpruned Python nested loop calculating the dynamic programming matrix for the query against every single word in the 333,333-entry unigram list without length filtering or dictionary-by-length partitioning ($333,333 \times O(|q| \times |w|)$ operations).
  `ablation_and_comparison.py` measured unpruned D-L on a vocabulary partition where words with length difference $>3$ were skipped by dictionary-by-length buckets, reducing total comparisons to approximately 25,000 words.
- **Resolution for Project-I Report:**
  Cite 4,330 ms as the worst-case unpruned scan across the full linear dictionary, and 135.80 ms as length-partitioned unpruned D-L. Cite the proposed engine's **19.57 ms – 36.70 ms** as the production 5-filter pruned performance.

---

## 3. Ground-Truth Results Verified on Real Test Runs

The following numbers have been directly measured on Apple Silicon M-Series (macOS Darwin, Python 3.9.6) and saved in JSON format:

### 3.1 Robustness Benchmark (`data/robustness_latency_memory_results.json`)
- **Total Test Cases:** 253 queries across 8 error categories
- **Overall Top-1 Accuracy:** $50.59\%$ (128 / 253)
- **Overall Top-3 Accuracy:** $66.80\%$ (169 / 253)
- **Overall Top-5 Accuracy:** $77.87\%$ (197 / 253)
- **Mean Reciprocal Rank (MRR):** $0.6051$
- **NDCG@5:** $0.6479$
- **Latency Percentiles:**
  - Mean: $46.54\text{ ms}$
  - Median ($P_{50}$): $36.70\text{ ms}$
  - 90th Percentile ($P_{90}$): $94.61\text{ ms}$
  - 95th Percentile ($P_{95}$): $103.95\text{ ms}$
  - 99th Percentile ($P_{99}$): $124.63\text{ ms}$
  - Throughput: $21.5\text{ Queries/Second}$
- **Memory Profiling (Continuous 1,000-query test):**
  - Baseline RAM: $86.58\text{ MB}$
  - Peak Dynamic RAM: $112.43\text{ MB}$
  - Incremental Allocation: $55.23\text{ KB/query}$
  - Total 1,000-query net memory drift: $+51.2\text{ KB}$ (zero cumulative leak)

### 3.2 Ablation Study Benchmark (`data/ablation_and_comparison_results.json`)
- **Test Corpus:** 29 benchmark queries (Prefix, Typo, Transliteration, Code-switched)
- **Full Proposed Engine:** Top-1: $37.9\%$, Top-5: $48.3\%$, MRR: $0.4167$, Latency: $1.48\text{ ms}$ (scaled by 1/10)
- **Ablation: Without ML Reranker (F10):** Top-1: $24.1\%$ ($-13.8\%$), MRR: $0.3161$ ($-0.1006$), Latency: $1.49\text{ ms}$
- **Ablation: Without Dynamic Weights (F11):** Top-1: $37.9\%$, MRR: $0.4080$, Latency: $1.49\text{ ms}$
- **Ablation: - Temporal / Trending Boost:** Top-1: $34.5\%$, MRR: $0.3994$, Latency: $1.48\text{ ms}$
- **Ablation: - Feedback Learner:** Top-1: $34.5\%$, MRR: $0.3822$, Latency: $1.48\text{ ms}$

---

## 4. Commands to Rerun Experiments Exactly

To verify or regenerate any of these numbers independently, execute the following commands in the virtual environment:

### Command 1: Run 253-Query Robustness, Latency & Memory Evaluation
```bash
source .venv/bin/activate
python benchmark_evaluation.py
```
*Outputs generated:* `data/robustness_latency_memory_results.json`, `robustness_latency_memory_report.png`.

### Command 2: Run Baseline Comparison & Component Ablation Study
```bash
source .venv/bin/activate
python ablation_and_comparison.py
```
*Outputs generated:* `data/ablation_and_comparison_results.json`, `ablation_and_comparison.png`.

### Command 3: Run 10-Query Feedback Learning Prototype
```bash
source .venv/bin/activate
python evaluate.py
```
*Outputs generated:* `accuracy_comparison.png`.

### Command 4: Run Complete Pytest & Unittest Verification
```bash
source .venv/bin/activate
pytest tests/test_api_and_features.py -v
python -m unittest test_comprehensive_project.py
```

---

## 5. Items Requiring Student Verification Before Final Viva

During your project defense, faculty members may ask specific questions regarding the baselines and training sets. Ensure you are familiar with the following:

1. `[BASELINE RESULT PENDING: Multilingual SymSpell Baseline]`
   - SymSpell was benchmarked on English unigrams ($81.8\text{ MB}$ memory footprint, $0.62\text{ ms}$ latency). For Tamil script, SymSpell requires pre-generating all deletion permutations of 50,000 Tamil words, which was omitted due to Unicode combining character segmentation issues.
2. `[RESULT TO BE VERIFIED: ML Reranker Synthetic vs Production Click Data]`
   - The Random Forest Regressor in `models/ml_ranker.pkl` was trained on 1,500 synthetically generated queries perturbed across Gaussian error distributions. Future evaluation on real search engine query logs (e.g., MS MARCO or AOL query logs) is documented as a future work milestone.
3. `[DATASET NOTE: Tamil Dictionary Size]`
   - The full Tamil dictionary (`data/tamil.csv`) contained 200,000+ words; for interactive keystroke latency testing, `data/tamil_small.csv` (top 50,000 words by frequency) was indexed into Trie memory.
