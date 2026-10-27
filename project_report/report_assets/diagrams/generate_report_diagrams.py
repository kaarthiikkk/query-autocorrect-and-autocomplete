"""Generate academic-grade system diagrams for Chapter 4 of the VIT Project-I Report.

Produces 5 high-resolution (300 DPI) publication diagrams:
1. system_architecture.png (Figure 4.1: Overall System Architecture)
2. data_flow_diagram.png   (Figure 4.2: Data Flow Diagram Level 0 and Level 1)
3. use_case_diagram.png    (Figure 4.3: UML Use Case Diagram)
4. class_diagram.png       (Figure 4.4: System Class & Module Architecture)
5. sequence_diagram.png    (Figure 4.5: End-to-End Query Execution Sequence)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = "project_report/report_assets/diagrams"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. SYSTEM ARCHITECTURE DIAGRAM (Figure 4.1)
# -----------------------------------------------------------------------------
def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Tier Backgrounds
    tiers = [
        ("Layer 1: Presentation & Interface Tier", 6.2, 1.4, "#EBF3FB", "#1B4F72"),
        ("Layer 2: RESTful Orchestration Tier (FastAPI)", 4.5, 1.4, "#E8F8F5", "#117864"),
        ("Layer 3: Query Analysis & Candidate Generation Tier", 2.6, 1.6, "#FEF9E7", "#B7950B"),
        ("Layer 4: Multi-Stage Hybrid Ranking & Adaptation Tier", 0.5, 1.8, "#F5EEF8", "#6C3483"),
    ]

    for title, y, h, bg_color, border_color in tiers:
        rect = patches.FancyBboxPatch(
            (0.5, y), 10, h, boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=bg_color, edgecolor=border_color, linewidth=1.5, linestyle="--"
        )
        ax.add_patch(rect)
        ax.text(0.7, y + h - 0.25, title, fontsize=10, fontweight="bold", color=border_color)

    # Layer 1 Components
    l1_boxes = [
        ("Streamlit Research Lab\n(dashboard.py - 5 Pages)", 0.8, 6.4, 2.8, 0.8),
        ("FastAPI OpenAPI Docs\n(Interactive Swagger /docs)", 4.1, 6.4, 2.8, 0.8),
        ("Command-Line Interface\n(Interactive CLI: main.py)", 7.4, 6.4, 2.8, 0.8),
    ]
    for text, x, y, w, h in l1_boxes:
        r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor="#FFFFFF", edgecolor="#1B4F72", linewidth=1.2)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8.5, fontweight="medium", color="#1B4F72")

    # Layer 2 Components
    l2_boxes = [
        ("/api/v1/suggest\n(Full Ranking Pipeline)", 0.8, 4.7, 2.2, 0.8),
        ("/api/v1/compare\n(Baseline Benchmark)", 3.3, 4.7, 2.1, 0.8),
        ("/api/v1/explain\n(7-Signal Radar)", 5.7, 4.7, 2.1, 0.8),
        ("/api/v1/feedback & metrics\n(Online Learner & Eval)", 8.1, 4.7, 2.1, 0.8),
    ]
    for text, x, y, w, h in l2_boxes:
        r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor="#FFFFFF", edgecolor="#117864", linewidth=1.2)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8, fontweight="medium", color="#117864")

    # Layer 3 Components
    l3_boxes = [
        ("Language & Script\nDetector\n(Unicode + Unigrams)", 0.8, 2.8, 2.1, 1.0),
        ("Error Classifier &\nIntent Detector\n(Typo, Prefix, OOV)", 3.2, 2.8, 2.1, 1.0),
        ("Dual-Trie\nAutocomplete\n(333k EN / 50k TA)", 5.6, 2.8, 2.1, 1.0),
        ("Pruned Damerau-\nLevenshtein (OSA)\n(5 Heuristic Filters)", 8.0, 2.8, 2.2, 1.0),
    ]
    for text, x, y, w, h in l3_boxes:
        r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor="#FFFFFF", edgecolor="#B7950B", linewidth=1.2)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8, fontweight="medium", color="#7D6608")

    # Layer 4 Components
    l4_boxes = [
        ("Dynamic Weight\nAdaptation Matrix\n(Dirichlet Norm)", 0.8, 0.7, 1.7, 1.1),
        ("7-Signal Hybrid\nScoring Engine\n(Lexical, Edit, Freq...)", 2.7, 0.7, 1.8, 1.1),
        ("Random Forest\nML Reranker\n(35 Trees, 28 Feats)", 4.7, 0.7, 1.8, 1.1),
        ("Temporal Burst &\nVelocity Detector\n(Poisson Spike Model)", 6.7, 0.7, 1.8, 1.1),
        ("Top-k Personalized\nRanker & Feedback\n(Decay + Guardrails)", 8.7, 0.7, 1.7, 1.1),
    ]
    for text, x, y, w, h in l4_boxes:
        r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor="#FFFFFF", edgecolor="#6C3483", linewidth=1.2)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=7.5, fontweight="medium", color="#512E5F")

    # Vertical Inter-tier Connectors
    arrows = [
        ((5.5, 6.4), (5.5, 5.5)),
        ((5.5, 4.7), (5.5, 4.2)),
        ((5.5, 2.8), (5.5, 2.3)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.8))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "system_architecture.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Generated {filepath}")


# -----------------------------------------------------------------------------
# 2. DATA FLOW DIAGRAM (DFD Level 0 & Level 1) (Figure 4.2)
# -----------------------------------------------------------------------------
def generate_data_flow_diagram():
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title / Headers
    ax.text(0.5, 7.6, "Data Flow Diagram (DFD Level 1: Multilingual Predictive Pipeline)", fontsize=11, fontweight="bold", color="#1A252C")

    # External Entities
    user_rect = patches.FancyBboxPatch((0.5, 3.2), 1.6, 1.2, boxstyle="square,pad=0.1", facecolor="#EBF5FB", edgecolor="#2980B9", linewidth=1.5)
    ax.add_patch(user_rect)
    ax.text(1.3, 3.8, "User /\nClient", ha="center", va="center", fontsize=9, fontweight="bold", color="#1B4F72")

    # Processes (Circles / Round Rectangles)
    processes = [
        ("1.0 Language &\nScript Analysis", 2.8, 5.4, "#FDEDEC", "#C0392B"),
        ("2.0 Intent & Error\nClassification", 5.2, 5.4, "#FEF9E7", "#D4AC0D"),
        ("3.0 Dual Candidate\nGeneration", 7.6, 5.4, "#E8F8F5", "#16A085"),
        ("4.0 Dynamic Weight\nAdaptation", 7.6, 3.2, "#EAF2F8", "#2980B9"),
        ("5.0 Hybrid & ML\nCascade Ranking", 5.2, 3.2, "#F4ECF7", "#8E44AD"),
        ("6.0 Personalization &\nTemporal Boost", 2.8, 3.2, "#E8F6F3", "#117A65"),
        ("7.0 Feedback Online\nRecording", 5.2, 1.0, "#EAEDED", "#7F8C8D"),
    ]

    for title, x, y, bg, border in processes:
        r = patches.FancyBboxPatch((x, y), 2.0, 1.1, boxstyle="round,pad=0.1,rounding_size=0.2", facecolor=bg, edgecolor=border, linewidth=1.4)
        ax.add_patch(r)
        ax.text(x + 1.0, y + 0.55, title, ha="center", va="center", fontsize=8, fontweight="bold", color=border)

    # Data Stores (Open-ended rectangles)
    stores = [
        ("D1: Dual Trie & Lexicons", 8.0, 7.0),
        ("D2: ML Weights & Trees", 5.2, 7.0),
        ("D3: User Profiles & Logs", 2.8, 1.0),
    ]
    for name, x, y in stores:
        r = patches.Rectangle((x, y), 2.2, 0.6, facecolor="#F8F9F9", edgecolor="#34495E", linewidth=1.2)
        ax.add_patch(r)
        ax.plot([x, x], [y, y + 0.6], color="#34495E", lw=2)  # open end
        ax.text(x + 1.1, y + 0.3, name, ha="center", va="center", fontsize=8, fontweight="bold", color="#2C3E50")

    # Data Flow Connectors (labeled arrows)
    def draw_flow(p1, p2, label, label_pos=(0, 0)):
        ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", color="#34495E", lw=1.3, shrinkA=3, shrinkB=3))
        mx = (p1[0] + p2[0]) / 2 + label_pos[0]
        my = (p1[1] + p2[1]) / 2 + label_pos[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=7, color="#2C3E50",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFFFFF", edgecolor="none", alpha=0.85))

    draw_flow((2.1, 3.9), (2.8, 5.8), "Raw Query", (-0.1, 0.2))
    draw_flow((4.8, 5.9), (5.2, 5.9), "Lang Tags", (0, 0.15))
    draw_flow((7.2, 5.9), (7.6, 5.9), "Intent/Type", (0, 0.15))
    draw_flow((8.6, 5.4), (8.6, 4.3), "Candidate Pools", (0.35, 0))
    draw_flow((7.6, 3.7), (7.2, 3.7), "Weight Vector", (0, 0.15))
    draw_flow((5.2, 3.7), (4.8, 3.7), "ML Ranked List", (0, 0.15))
    draw_flow((2.8, 3.6), (2.1, 3.6), "Top-K Suggestions", (-0.1, -0.2))

    # Feedback flow
    draw_flow((1.3, 3.2), (5.2, 1.4), "Selected Suggestion (Click)", (0.5, -0.1))
    draw_flow((7.2, 1.4), (8.5, 3.2), "Learned Priors", (0.3, -0.3))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "data_flow_diagram.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Generated {filepath}")


# -----------------------------------------------------------------------------
# 3. USE CASE DIAGRAM (Figure 4.3)
# -----------------------------------------------------------------------------
def generate_use_case_diagram():
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # System Boundary Box
    sys_box = patches.FancyBboxPatch((2.4, 0.5), 5.4, 7.0, boxstyle="round,pad=0.15", facecolor="#FCFDFD", edgecolor="#2C3E50", linewidth=1.5)
    ax.add_patch(sys_box)
    ax.text(5.1, 7.2, "Multilingual Search Intelligence System", ha="center", fontsize=10, fontweight="bold", color="#1B4F72")

    # Actors
    def draw_actor(x, y, label):
        # Head
        head = patches.Circle((x, y), 0.22, facecolor="#EBF5FB", edgecolor="#2C3E50", lw=1.4)
        ax.add_patch(head)
        # Body
        ax.plot([x, x], [y - 0.22, y - 0.7], color="#2C3E50", lw=1.4)
        # Arms
        ax.plot([x - 0.35, x + 0.35], [y - 0.4, y - 0.4], color="#2C3E50", lw=1.4)
        # Legs
        ax.plot([x, x - 0.25], [y - 0.7, y - 1.1], color="#2C3E50", lw=1.4)
        ax.plot([x, x + 0.25], [y - 0.7, y - 1.1], color="#2C3E50", lw=1.4)
        # Label
        ax.text(x, y - 1.3, label, ha="center", fontsize=8.5, fontweight="bold", color="#1A252F")

    draw_actor(1.1, 5.6, "End User /\nSearcher")
    draw_actor(1.1, 2.2, "Research\nEvaluator")
    draw_actor(8.9, 4.0, "Search\nAdmin")

    # Use Cases (Ellipses)
    use_cases = [
        ("Query Multilingual Search", 5.1, 6.4, ["user"]),
        ("Receive Prefix Autocomplete", 5.1, 5.5, ["user"]),
        ("View D-L Typo Corrections", 5.1, 4.6, ["user"]),
        ("Provide Relevance Feedback", 5.1, 3.7, ["user"]),
        ("Inspect 7-Signal Explainability", 5.1, 2.8, ["user", "eval"]),
        ("Run Benchmark Evaluation", 5.1, 1.9, ["eval", "admin"]),
        ("Execute Component Ablation", 5.1, 1.0, ["eval", "admin"]),
    ]

    for label, x, y, roles in use_cases:
        ellipse = patches.Ellipse((x, y), 3.2, 0.65, facecolor="#E8F8F5", edgecolor="#117864", linewidth=1.2)
        ax.add_patch(ellipse)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="medium", color="#0E6251")

        # Connect actors to use cases
        if "user" in roles:
            ax.plot([1.4, x - 1.6], [5.2, y], color="#7F8C8D", lw=1.0, linestyle="-")
        if "eval" in roles:
            ax.plot([1.4, x - 1.6], [2.2, y], color="#7F8C8D", lw=1.0, linestyle="-")
        if "admin" in roles:
            ax.plot([8.6, x + 1.6], [4.0, y], color="#7F8C8D", lw=1.0, linestyle="-")

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "use_case_diagram.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Generated {filepath}")


# -----------------------------------------------------------------------------
# 4. CLASS & MODULE INTERACTION DIAGRAM (Figure 4.4)
# -----------------------------------------------------------------------------
def generate_class_diagram():
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(0.5, 7.6, "System Class & Module Architecture Diagram", fontsize=11, fontweight="bold", color="#1B4F72")

    def draw_class_box(x, y, w, h, name, attrs, methods, color_theme="#2980B9"):
        header_h = 0.55
        # Outer box
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0.0", facecolor="#FFFFFF", edgecolor=color_theme, linewidth=1.4)
        ax.add_patch(rect)
        # Header box
        hdr = patches.Rectangle((x, y + h - header_h), w, header_h, facecolor=color_theme, edgecolor=color_theme)
        ax.add_patch(hdr)
        ax.text(x + w/2, y + h - header_h/2, name, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#FFFFFF")
        # Attributes
        curr_y = y + h - header_h - 0.2
        for attr in attrs:
            ax.text(x + 0.15, curr_y, attr, fontsize=7, color="#2C3E50")
            curr_y -= 0.22
        # Divider line
        curr_y -= 0.05
        ax.plot([x, x + w], [curr_y, curr_y], color=color_theme, lw=0.8, linestyle=":")
        curr_y -= 0.2
        # Methods
        for m in methods:
            ax.text(x + 0.15, curr_y, m, fontsize=7, color="#1B4F72")
            curr_y -= 0.22

    # Class 1: TrieNode & Trie
    draw_class_box(
        0.6, 4.4, 2.8, 2.8, "Trie (trie.py)",
        ["+ root: TrieNode", "+ word_count: int", "+ unigram_data: dict"],
        ["+ insert(word, freq): void", "+ search(prefix): TrieNode", "+ autocomplete(prefix, k): list", "+ is_prefix(s): bool"],
        "#1B4F72"
    )

    # Class 2: Damerau-Levenshtein Corrector
    draw_class_box(
        3.9, 4.4, 3.2, 2.8, "Corrector (corrector.py)",
        ["+ dict_by_length: dict", "+ max_edit_distance: int", "+ length_prune_margin: int"],
        ["+ damerau_levenshtein(s1, s2): int", "+ correct_query(query, k): list", "+ filter_candidate_pool(): list", "+ exact_match_short_circuit(): bool"],
        "#117864"
    )

    # Class 3: Dynamic Weights
    draw_class_box(
        7.6, 4.4, 2.8, 2.8, "DynamicWeights (dynamic_weights.py)",
        ["+ default_weights: dict", "+ weight_multipliers: dict", "+ normalization_factor: float"],
        ["+ get_dynamic_weights(query, type): dict", "+ normalize_dirichlet(weights): dict", "+ adapt_for_intent(weights, intent)", "+ adapt_for_typos(weights, err_type)"],
        "#7D6608"
    )

    # Class 4: Multilingual Engine Orchestrator
    draw_class_box(
        0.6, 0.7, 3.0, 3.1, "MultilingualSearchEngine (multilingual_search.py)",
        ["+ trie_en: Trie", "+ trie_ta: Trie", "+ ml_model: RandomForestRegressor", "+ temporal_ranker: TemporalRanker"],
        ["+ search(query, top_k, user_id): dict", "+ detect_language(query): tuple", "+ extract_ranking_features(c): list", "+ fuse_candidate_pools(): list"],
        "#6C3483"
    )

    # Class 5: ML Ranker
    draw_class_box(
        4.0, 0.7, 3.1, 3.1, "MLRanker (ml_ranker.py)",
        ["+ model_path: str", "+ n_estimators: int = 35", "+ feature_dimension: int = 28"],
        ["+ load_model(path): RandomForest", "+ extract_features(query, c): list", "+ rerank_candidates(candidates): list", "+ compute_feature_importance(): dict"],
        "#2E4053"
    )

    # Class 6: Personalized & Feedback Learner
    draw_class_box(
        7.6, 0.7, 2.8, 3.1, "PersonalizedRanker (personalized_ranker.py)",
        ["+ profile_db_path: str", "+ session_history: dict", "+ affinity_decay_lambda: float"],
        ["+ record_click(user_id, word): void", "+ compute_affinity(user_id, c): float", "+ apply_personalized_boost(): list", "+ anti_poison_guardrail(): bool"],
        "#A04000"
    )

    # Associations / Dependencies
    def draw_assoc(p1, p2, label=""):
        ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", color="#34495E", lw=1.2))

    draw_assoc((2.1, 3.8), (2.1, 4.4))
    draw_assoc((3.6, 2.2), (4.0, 2.2))
    draw_assoc((3.6, 3.5), (4.5, 4.4))
    draw_assoc((7.1, 2.2), (7.6, 2.2))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "class_diagram.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Generated {filepath}")


# -----------------------------------------------------------------------------
# 5. SEQUENCE DIAGRAM (Figure 4.5)
# -----------------------------------------------------------------------------
def generate_sequence_diagram():
    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(0.5, 8.6, "End-to-End Query Execution Sequence Diagram", fontsize=11, fontweight="bold", color="#1B4F72")

    lifelines = [
        ("User", 1.0),
        ("FastAPI (api.py)", 2.8),
        ("Detector / Classifier", 4.6),
        ("Candidate Generators\n(Trie / Corrector)", 6.4),
        ("Dynamic & Hybrid Ranker", 8.2),
        ("ML Reranker &\nPersonalization", 10.0),
    ]

    # Draw Lifeline Headers & Lines
    for name, x in lifelines:
        box = patches.FancyBboxPatch((x - 0.7, 7.8), 1.4, 0.6, boxstyle="round,pad=0.05", facecolor="#EBF5FB", edgecolor="#2980B9", lw=1.2)
        ax.add_patch(box)
        ax.text(x, 8.1, name, ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1B4F72")
        ax.plot([x, x], [0.8, 7.8], color="#BDC3C7", lw=1.0, linestyle="--")

    # Sequence Messages
    steps = [
        (1.0, 2.8, 7.4, "1: POST /suggest(query='pythn')", "call"),
        (2.8, 4.6, 6.9, "2: analyze_language() & detect_error()", "call"),
        (4.6, 2.8, 6.4, "3: return (lang='english', type='typo')", "return"),
        (2.8, 6.4, 5.9, "4: generate_candidates(k=5)", "call"),
        (6.4, 6.4, 5.4, "5: Pruned D-L matrix search (dist<=2)", "self"),
        (6.4, 2.8, 4.9, "6: return candidate pool ['python', 'pytho', 'path']", "return"),
        (2.8, 8.2, 4.4, "7: adapt_weights() & score_7_signals()", "call"),
        (8.2, 2.8, 3.9, "8: return scored candidates", "return"),
        (2.8, 10.0, 3.4, "9: rerank_ml() & apply_user_affinity()", "call"),
        (10.0, 2.8, 2.9, "10: return final ranked suggestions", "return"),
        (2.8, 1.0, 2.4, "11: 200 OK: Top-5 suggestions in 25ms", "return"),
        (1.0, 2.8, 1.8, "12: POST /feedback(selection='python')", "call"),
        (2.8, 10.0, 1.3, "13: record_click() & update affinity", "call"),
    ]

    for x1, x2, y, text, msg_type in steps:
        if msg_type == "call":
            ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="->", color="#1F618D", lw=1.3))
            ax.text((x1 + x2)/2, y + 0.1, text, ha="center", fontsize=7, color="#1B4F72",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="#FFFFFF", edgecolor="none", alpha=0.9))
        elif msg_type == "return":
            ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.3, linestyle=":"))
            ax.text((x1 + x2)/2, y + 0.1, text, ha="center", fontsize=7, color="#196F3D",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="#FFFFFF", edgecolor="none", alpha=0.9))
        elif msg_type == "self":
            # Self loop
            ax.plot([x1, x1 + 0.5, x1 + 0.5, x1], [y, y, y - 0.3, y - 0.3], color="#7D6608", lw=1.1)
            ax.annotate("", xy=(x1, y - 0.3), xytext=(x1 + 0.1, y - 0.3), arrowprops=dict(arrowstyle="->", color="#7D6608", lw=1.1))
            ax.text(x1 + 0.55, y - 0.15, text, ha="left", va="center", fontsize=6.8, color="#7D6608")

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "sequence_diagram.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Generated {filepath}")


if __name__ == "__main__":
    generate_system_architecture()
    generate_data_flow_diagram()
    generate_use_case_diagram()
    generate_class_diagram()
    generate_sequence_diagram()
    print("All 5 diagrams successfully generated!")
