"""Training and Evaluation Script for Feature 10 ML-Based Reranker.

Trains a Random Forest ranking model using features extracted from the
Feature 1–9 Multilingual Search & Hybrid Ranking pipeline.
Saves model to models/ml_ranker.pkl and prints evaluation benchmarks.
"""

import csv
import collections
import math
import os
import random
import time

from trie import Trie
from data_loader import load_dictionary_and_trie
from multilingual_search import multilingual_search
from ml_ranker import (
    extract_features,
    train_model,
    save_model,
    rerank_candidates,
    get_feature_importance,
    FEATURE_NAMES,
)


def load_environment():
    """Load English and Tamil dictionaries."""
    print("Loading language resources for ML training...")
    trie_en, dict_en = load_dictionary_and_trie("english")
    trie_ta, dict_ta = load_dictionary_and_trie("tamil")

    language_data = {
        "english": (trie_en, dict_en),
        "tamil": (trie_ta, dict_ta),
    }
    user_freq = {
        "python": 12,
        "machine": 7,
        "learning": 6,
        "software": 5,
        "hello": 8,
        "computer": 4,
        "writing": 4,
        "language": 5,
    }
    return language_data, user_freq


def get_training_query_pairs():
    """Construct deterministic query-target pairs across English, Tamil, and Romanized."""
    pairs = []

    # 1. Load benchmark spelling test cases
    test_cases_path = "data/test_cases.csv"
    if os.path.exists(test_cases_path):
        with open(test_cases_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((row["misspelled"].strip(), row["correct"].strip(), "spelling"))

    # 2. English typo & autocomplete queries
    additional_en = [
        ("pythn", "python", "spelling"),
        ("pyt", "python", "autocomplete"),
        ("pyth", "python", "autocomplete"),
        ("machin", "machine", "spelling"),
        ("mach", "machine", "autocomplete"),
        ("learnin", "learning", "spelling"),
        ("lear", "learning", "autocomplete"),
        ("progr", "program", "autocomplete"),
        ("comput", "computer", "autocomplete"),
        ("softw", "software", "autocomplete"),
        ("restarant", "restaurant", "spelling"),
        ("restaur", "restaurant", "autocomplete"),
        ("devlop", "develop", "spelling"),
        ("algotihm", "algorithm", "spelling"),
        ("artifical", "artificial", "spelling"),
        ("intelligenc", "intelligence", "spelling"),
        ("buy iphon", "iphone", "intent"),
        ("what is machin", "machine", "intent"),
        ("restaurants near m", "restaurant", "intent"),
    ]
    pairs.extend(additional_en)

    # 3. Tamil native queries
    tamil_pairs = [
        ("தமி", "தமிழ்", "autocomplete"),
        ("வணக்", "வணக்கம்", "autocomplete"),
        ("சாப்", "சாப்பாடு", "autocomplete"),
        ("மருத்", "மருத்துவமனை", "autocomplete"),
        ("உணவ", "உணவகம்", "autocomplete"),
        ("வேல", "வேலை", "autocomplete"),
        ("பைதான", "பைதான்", "spelling"),
    ]
    pairs.extend(tamil_pairs)

    # 4. Romanized Tamil queries
    romanized_pairs = [
        ("venum", "வேண்டும்", "romanized"),
        ("vendum", "வேண்டும்", "romanized"),
        ("sapadu", "சாப்பாடு", "romanized"),
        ("epdi", "எப்படி", "romanized"),
        ("eppadi", "எப்படி", "romanized"),
        ("nalla", "நல்லா", "romanized"),
        ("ponga", "போங்க", "romanized"),
        ("vaanga", "வாங்க", "romanized"),
    ]
    pairs.extend(romanized_pairs)

    return pairs


def build_dataset(query_pairs, language_data, user_freq):
    """Generate candidate pools and extract feature vectors with relevance labels."""
    print("Generating candidate pools and extracting feature vectors...")
    records = []

    for query, target, q_type in query_pairs:
        res = multilingual_search(
            query,
            language_data,
            user_freq,
            autocomplete_k=6,
            correction_k=6,
            ml_model=None
        )
        candidates = res.get("ranked_candidates", [])
        if not candidates:
            continue

        query_examples = []
        for cand in candidates:
            word = cand.get("word", "").lower()
            features = extract_features(cand, query)

            # Assign graded ground truth label
            if word == target.lower():
                relevance = 3.0  # Exact target match
            elif word.startswith(query.lower()) or target.lower().startswith(word):
                relevance = 2.0  # Strong prefix / target overlap
            elif cand.get("edit_distance", 5) <= 1:
                relevance = 1.0  # Close single-edit neighbor
            else:
                relevance = 0.0  # Distant / irrelevant

            query_examples.append({
                "query": query,
                "target": target,
                "candidate": cand,
                "features": features,
                "label": relevance,
            })

        records.append(query_examples)

    return records


def compute_mrr(ranked_words, target):
    """Calculate Mean Reciprocal Rank for target word in ranked list."""
    for rank, word in enumerate(ranked_words, 1):
        if word.lower() == target.lower():
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(ranked_labels, k=5):
    """Calculate Normalized Discounted Cumulative Gain at rank K."""
    dcg = 0.0
    for i, label in enumerate(ranked_labels[:k]):
        dcg += (2.0 ** label - 1.0) / math.log2(i + 2)

    ideal_labels = sorted(ranked_labels, reverse=True)[:k]
    idcg = 0.0
    for i, label in enumerate(ideal_labels):
        idcg += (2.0 ** label - 1.0) / math.log2(i + 2)

    return (dcg / idcg) if idcg > 0 else 0.0


def evaluate_split(split_data, model=None, alpha=0.35):
    """Compute Precision@5, Recall@5, MRR, and NDCG@5 on a query dataset split."""
    precisions = []
    recalls = []
    mrrs = []
    ndcgs = []
    latencies = []

    for query_group in split_data:
        if not query_group:
            continue

        query = query_group[0]["query"]
        target = query_group[0]["target"]
        candidates = [item["candidate"] for item in query_group]
        labels_by_word = {item["candidate"]["word"]: item["label"] for item in query_group}

        start = time.perf_counter()
        if model is not None:
            ranked = rerank_candidates(query, candidates, model=model, alpha=alpha)
        else:
            # Baseline Hybrid Ranking
            ranked = candidates
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        top_words = [c["word"] for c in ranked[:5]]
        top_labels = [labels_by_word.get(w, 0.0) for w in top_words]

        # Precision@5: ratio of relevant candidates (label >= 1) in top-5
        relevant_in_top5 = sum(1 for lbl in top_labels if lbl >= 1.0)
        precisions.append(relevant_in_top5 / max(len(top_words), 1))

        # Recall@5: whether ground truth target was found in top-5
        target_found = 1.0 if any(w.lower() == target.lower() for w in top_words) else 0.0
        recalls.append(target_found)

        # MRR
        mrrs.append(compute_mrr(top_words, target))

        # NDCG@5
        all_labels = [labels_by_word.get(c["word"], 0.0) for c in ranked]
        ndcgs.append(compute_ndcg_at_k(all_labels, k=5))

    return {
        "precision@5": sum(precisions) / max(len(precisions), 1),
        "recall@5": sum(recalls) / max(len(recalls), 1),
        "mrr": sum(mrrs) / max(len(mrrs), 1),
        "ndcg@5": sum(ndcgs) / max(len(ndcgs), 1),
        "avg_latency_ms": sum(latencies) / max(len(latencies), 1),
    }


def main():
    print("=" * 60)
    print(" FEATURE 10: ML RERANKER TRAINING & EVALUATION PIPELINE")
    print("=" * 60)

    # 1. Load resources
    language_data, user_freq = load_environment()

    # 2. Get training pairs & build candidate records
    query_pairs = get_training_query_pairs()
    print(f"Loaded {len(query_pairs)} training query scenarios.")
    dataset = build_dataset(query_pairs, language_data, user_freq)

    # 3. Train/Validation/Test Split (70% / 15% / 15%) with fixed seed=42
    random.seed(42)
    shuffled_indices = list(range(len(dataset)))
    random.shuffle(shuffled_indices)

    n_total = len(dataset)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    train_groups = [dataset[i] for i in shuffled_indices[:n_train]]
    val_groups = [dataset[i] for i in shuffled_indices[n_train:n_train + n_val]]
    test_groups = [dataset[i] for i in shuffled_indices[n_train + n_val:]]

    print(f"Data Split: {len(train_groups)} Train, {len(val_groups)} Validation, {len(test_groups)} Test queries.")

    # Flatten train set for ML regressor
    X_train = []
    y_train = []
    for group in train_groups:
        for item in group:
            X_train.append(item["features"])
            y_train.append(item["label"])

    print(f"Total training instances: {len(X_train)} samples across {len(FEATURE_NAMES)} features.")

    # 4. Train the Random Forest Ranker
    print("Training Random Forest Regressor (n_estimators=35, max_depth=5)...")
    start_train = time.perf_counter()
    model = train_model(X_train, y_train, n_estimators=35, max_depth=5, random_state=42)
    train_time = (time.perf_counter() - start_train) * 1000
    print(f"Training completed in {train_time:.2f} ms.")

    # 5. Save model
    model_path = "models/ml_ranker.pkl"
    save_model(model, filepath=model_path)
    print(f"Model saved successfully to {model_path}.")

    # 6. Feature Importance Analysis
    importances = get_feature_importance(model)
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("\n--- TOP-10 LEARNED FEATURE IMPORTANCES ---")
    for rank, (name, val) in enumerate(sorted_importances[:10], 1):
        print(f"  {rank:2d}. {name:28s}: {val:.4f} ({val*100:.1f}%)")

    # 7. Evaluate on Held-Out Test Set (Baseline vs ML Reranker)
    print("\n" + "=" * 60)
    print(" EVALUATION RESULTS ON HELD-OUT TEST SET")
    print("=" * 60)

    baseline_metrics = evaluate_split(test_groups, model=None)
    ml_metrics = evaluate_split(test_groups, model=model, alpha=0.35)

    print(f"{'Metric':<20} | {'Baseline Hybrid (F9)':<22} | {'Hybrid + ML Rerank (F10)':<25}")
    print("-" * 72)
    print(f"{'Precision@5':<20} | {baseline_metrics['precision@5']*100:>19.2f}% | {ml_metrics['precision@5']*100:>22.2f}%")
    print(f"{'Recall@5':<20} | {baseline_metrics['recall@5']*100:>19.2f}% | {ml_metrics['recall@5']*100:>22.2f}%")
    print(f"{'MRR':<20} | {baseline_metrics['mrr']:>20.4f} | {ml_metrics['mrr']:>23.4f}")
    print(f"{'NDCG@5':<20} | {baseline_metrics['ndcg@5']:>20.4f} | {ml_metrics['ndcg@5']:>23.4f}")
    print(f"{'Inference Latency':<20} | {baseline_metrics['avg_latency_ms']:>17.2f} ms | {ml_metrics['avg_latency_ms']:>20.2f} ms")
    print("=" * 72)

    return model, baseline_metrics, ml_metrics


if __name__ == "__main__":
    main()
