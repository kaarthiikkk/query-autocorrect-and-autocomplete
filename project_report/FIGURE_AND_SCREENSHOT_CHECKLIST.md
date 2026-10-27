# Figure & Screenshot Inventory Checklist (FIGURE_AND_SCREENSHOT_CHECKLIST.md)

**Project Title:** A Hybrid Adaptive Ranking Framework for Multilingual Predictive Search and Query Correction  
**Course:** BCSE497J - Project-I, Fall Semester 2026–2027  
**Department:** School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT), Vellore  

---

## 1. Inventory of Figures Embedded in `Project_I_Report.docx`

The following figures have been generated at 300 DPI publication resolution and inserted directly into the formal Word report document:

| Figure # | Official Title in Report | Caption / Description | Source File / Asset Path |
|:---:|:---|:---|:---|
| **Fig. 4.1** | Overall System Architecture | Four-tier architectural diagram depicting Client/Presentation tier, FastAPI orchestration tier, Query Intelligence tier, and Multi-Stage Ranking tier. | [`project_report/report_assets/diagrams/system_architecture.png`](file:///Users/karthikreddy/Desktop/search_engine_project/project_report/report_assets/diagrams/system_architecture.png) |
| **Fig. 4.2** | Data Flow Diagram (DFD Level 1) | Detailed functional flow showing data transforms from Raw Query input through Language Analysis, Candidate Pools, Dynamic Weighting, ML Ranking, to Feedback storage. | [`project_report/report_assets/diagrams/data_flow_diagram.png`](file:///Users/karthikreddy/Desktop/search_engine_project/project_report/report_assets/diagrams/data_flow_diagram.png) |
| **Fig. 4.3** | UML Use Case Diagram | Use case interactions between End User/Searcher, Research Evaluator, and System Administrator across query, explainability, and benchmarking functions. | [`project_report/report_assets/diagrams/use_case_diagram.png`](file:///Users/karthikreddy/Desktop/search_engine_project/project_report/report_assets/diagrams/use_case_diagram.png) |
| **Fig. 4.4** | System Class & Module Architecture | Class structure showing `Trie`, `Corrector`, `DynamicWeights`, `MultilingualSearchEngine`, `MLRanker`, and `PersonalizedRanker`. | [`project_report/report_assets/diagrams/class_diagram.png`](file:///Users/karthikreddy/Desktop/search_engine_project/project_report/report_assets/diagrams/class_diagram.png) |
| **Fig. 4.5** | End-to-End Query Execution Sequence | UML sequence diagram tracing asynchronous HTTP query handling through internal algorithmic stages to final 200 OK response and feedback event. | [`project_report/report_assets/diagrams/sequence_diagram.png`](file:///Users/karthikreddy/Desktop/search_engine_project/project_report/report_assets/diagrams/sequence_diagram.png) |
| **Fig. 5.1** | Latency Comparison: Trie vs. Unpruned D-L | Keystroke latency benchmarking comparing sub-millisecond Trie prefix lookup (0.11 ms) with unpruned brute-force Damerau-Levenshtein (4,330.63 ms, log scale). | [`project_report/report_assets/benchmark_plots/fig3.png`](file:///Users/karthikreddy/Desktop/search_engine_project/project_report/report_assets/benchmark_plots/fig3.png) |
| **Fig. 7.1** | Robustness, Latency & Memory Profiling | 4-quadrant benchmark report: (a) Accuracy across 8 error classes, (b) Latency percentile distribution curve, (c) Continuous 1,000-query memory trace, (d) Summary card table. | [`project_report/report_assets/benchmark_plots/robustness_latency_memory_report.png`](file:///Users/karthikreddy/Desktop/search_engine_project/project_report/report_assets/benchmark_plots/robustness_latency_memory_report.png) |
| **Fig. 7.2** | Baseline Comparison & Component Ablation | 4-quadrant comparative analysis: (a) Accuracy-latency Pareto frontier, (b) Memory footprint comparison (SymSpell vs. Trie vs. Proposed), (c) Component degradation ablation, (d) Metric table. | [`project_report/report_assets/benchmark_plots/ablation_and_comparison.png`](file:///Users/karthikreddy/Desktop/search_engine_project/project_report/report_assets/benchmark_plots/ablation_and_comparison.png) |
| **Fig. 7.3** | Impact of Adaptive Feedback Learning | Comparative bar chart demonstrating accuracy improvement from baseline unprimed state to adaptive user feedback state on target test words. | [`project_report/report_assets/benchmark_plots/accuracy_comparison.png`](file:///Users/karthikreddy/Desktop/search_engine_project/project_report/report_assets/benchmark_plots/accuracy_comparison.png) |

---

## 2. Inventory of Tables in `Project_I_Report.docx`

| Table # | Title | Description / Content |
|:---:|:---|:---|
| **Table 2.1** | Project Execution Timeline & Milestones | Gantt / Sprint breakdown from literature survey through architecture, evaluation, and documentation. |
| **Table 3.1** | Hardware Specification | Minimum and development environment hardware specs (Apple M-Series / Intel Core i7, 16GB RAM). |
| **Table 3.2** | Software Specification | Operating system, Python 3.9 runtime, FastAPI, Streamlit, scikit-learn, and pytest toolchains. |
| **Table 5.1** | Unit & Integration Test Suite Summary | Pytest (16 tests) and Unittest (14 tests) execution matrices with pass/fail status and runtimes. |
| **Table 7.1** | Robustness Evaluation Across 8 Error Categories | Detailed accuracy (Top-1, Top-3, Top-5), MRR, and NDCG@5 across clean, deletion, insertion, transposition, qwerty, and edit distance 2/3 queries. |
| **Table 7.2** | Query Latency Percentile Benchmarks | Latency distribution percentiles ($P_{50}, P_{90}, P_{95}, P_{99}$), throughput (QPS), and incremental memory consumption. |
| **Table 7.3** | Algorithmic Baseline Comparison | Direct comparison of Standard Prefix Trie, Peter Norvig Corrector, SymSpell, Unpruned D-L, and Proposed Engine on accuracy, speed, and RAM. |
| **Table 7.4** | Component Ablation Study Results | Quantitative evaluation of model performance when individual architectural components (ML Reranker, Dynamic Weights, Autocomplete) are isolated. |
| **Table 7.5** | Computational Cost & Memory Footprint Analysis | Resource utilization assessment establishing computational feasibility on resource-constrained servers. |

---

## 3. Checklist for Live Review / Viva Screenshots

For your final B.Tech project review and laboratory viva presentation, start both servers locally and capture high-resolution screenshots of the following interactive screens:

- [ ] **Screenshot 1: Live Query Playground — Clean Query Completion**
  - **URL:** `http://localhost:8501` (Navigate to Page 1: Live Query Playground)
  - **Action:** Enter prefix query `pyth` with language set to `English`.
  - **What to highlight:** Trie autocomplete candidates (`python`, `pythonic`), high confidence score ($0.95+$), and sub-5 ms latency.
- [ ] **Screenshot 2: Live Query Playground — Typo Correction with Damerau-Levenshtein**
  - **URL:** `http://localhost:8501` (Page 1)
  - **Action:** Enter misspelled query `pythn` and click *Execute Search Pipeline*.
  - **What to highlight:** Automatic typo classification, candidate ranked list with `python` ranked #1, confidence gap ($>0.25$), and source tagged as `damerau_levenshtein`.
- [ ] **Screenshot 3: Live Query Playground — Romanized Tamil Transliteration**
  - **URL:** `http://localhost:8501` (Page 1)
  - **Action:** Enter query `vanakam` or `chenai`.
  - **What to highlight:** Detection of Romanized Tamil, phonetic bridge to native script `வணக்கம்` or `சென்னை`, and source tagging.
- [ ] **Screenshot 4: Ranking Comparison Page**
  - **URL:** `http://localhost:8501` (Navigate to Page 2: Ranking Comparison)
  - **Action:** Enter test query `pythn`.
  - **What to highlight:** Side-by-side execution showing Standard Trie returning empty (`[]`), while Pruned D-L, Hybrid, ML, and Proposed Engine successfully recover `python`. Highlight latency differences.
- [ ] **Screenshot 5: 7-Signal Radar Explainability Page**
  - **URL:** `http://localhost:8501` (Navigate to Page 3: Ranking Explainability)
  - **Action:** Inspect the top candidate `python`.
  - **What to highlight:** Interactive Plotly radar chart displaying `lexical_prefix`, `edit_similarity`, `log_frequency`, `semantic_similarity`, `session_context`, `intent_compatibility`, and `user_feedback`, alongside dynamic weight shifts.
- [ ] **Screenshot 6: Evaluation Dashboard & Ablation Pages**
  - **URL:** `http://localhost:8501` (Navigate to Page 4 and Page 5)
  - **What to highlight:** Native Streamlit metric cards displaying Top-1 ($50.59\%$), MRR ($0.6051$), $P_{50}$ latency ($36.70\text{ ms}$), and the Pareto accuracy-latency frontier chart.
- [ ] **Screenshot 7: FastAPI Interactive OpenAPI Documentation**
  - **URL:** `http://localhost:8000/docs`
  - **What to highlight:** Swagger UI demonstrating all 11 endpoints (`/api/v1/suggest`, `/api/v1/compare`, `/api/v1/explain`, etc.) with Pydantic JSON schemas.
