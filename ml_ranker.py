"""ML-Based Candidate Reranking Module for Multilingual Search.

Operates on the candidates produced by the Hybrid Ranking engine (Feature 9).
Consumes existing lexical, semantic, frequency, context, intent, error, and
protection signals to predict learned candidate relevance.

Dynamic Score Adaptation
        alpha * hybrid_score + (1 - alpha) * normalized_ml_score
"""

import math
import os
import pickle
import random


FEATURE_NAMES = [
    "lexical_prefix_similarity",
    "edit_distance",
    "edit_similarity",
    "is_exact_match",
    "is_prefix_match",
    "len_diff",
    "len_ratio",
    "dictionary_freq",
    "log_frequency",
    "user_score",
    "semantic_score",
    "context_score",
    "has_recent_context",
    "intent_score",
    "intent_navigational",
    "intent_question",
    "intent_transactional",
    "intent_local",
    "intent_general",
    "source_autocomplete",
    "source_correction",
    "source_romanized",
    "provenance_count",
    "is_correction",
    "is_oov",
    "is_rare",
    "should_protect",
    "hybrid_score",
]


def extract_features(candidate, query=None):
    """Extract a 28-dimensional normalized feature vector from candidate metadata.

    Consumes existing Feature 1–9 signals directly:
    - Lexical prefix & edit distances (Features 4, 5, 6)
    - Trie frequency & user frequency (Features 2, 4)
    - Semantic similarity embedding cosine (Feature 9)
    - Context score & reasons (Feature 3)
    - Intent detection & compatibility (Features 8, 9)
    - Candidate source & provenance (Feature 2)
    - OOV & entity protection status (Feature 7)
    - Baseline Hybrid score (Feature 9)
    """
    word = candidate.get("word", "")
    token = candidate.get("token") or (query or "")
    q = query or token

    lexical_sim = float(candidate.get("lexical_score", candidate.get("prefix_score", 0.0)))
    edit_dist = candidate.get("edit_distance")
    if edit_dist is None:
        edit_dist = 0 if candidate.get("source") in {"trie_autocomplete", "romanized_lexicon"} else 3
    edit_dist = float(edit_dist)

    edit_sim = float(candidate.get("edit_score", 1.0 if edit_dist == 0 else max(0.0, 1.0 - edit_dist / max(len(word), len(token), 1))))
    is_exact = 1.0 if (q and word.lower() == q.lower()) or candidate.get("source") == "romanized_lexicon" else 0.0
    is_prefix = 1.0 if (token and word.lower().startswith(token.lower())) or candidate.get("source") == "romanized_lexicon" else 0.0
    if not is_prefix and " " in token and " " in word:
        q_tokens = token.lower().split()
        w_tokens = word.lower().split()
        if len(q_tokens) == len(w_tokens) and all(wt.startswith(qt) for qt, wt in zip(q_tokens, w_tokens)):
            is_prefix = 1.0
    len_diff = float(abs(len(word) - len(token)))
    len_ratio = float(len(word) / max(len(token), 1))

    freq = float(candidate.get("frequency", 1) or 1)
    log_freq = float(candidate.get("frequency_score", math.log(freq + 1) / (math.log(freq + 1) + 10)))
    user_score = float(candidate.get("user_score", 0.0))

    # Consuming Feature 9 semantic similarity signal
    semantic_score = float(candidate.get("semantic_score", 0.0))

    # Consuming Feature 3 context signals
    context_score = float(candidate.get("context_score", 0.0))
    has_recent = 1.0 if candidate.get("context_reasons") else 0.0

    # Consuming Feature 8 & 9 intent signals
    intent_score = float(candidate.get("intent_score", 0.5))
    intent = str(candidate.get("intent", "general_search")).lower()
    intent_nav = 1.0 if intent == "navigational" else 0.0
    intent_q = 1.0 if intent == "question" else 0.0
    intent_tx = 1.0 if intent == "transactional" else 0.0
    intent_local = 1.0 if intent == "local" else 0.0
    intent_gen = 1.0 if intent in {"general_search", "general"} else 0.0

    # Source & Provenance
    source = candidate.get("source", "")
    source_auto = 1.0 if source in {"trie_autocomplete", "autocomplete", "hybrid_autocomplete", "hybrid_phrase"} or candidate.get("is_phrase") else 0.0
    source_corr = 1.0 if source in {"damerau_levenshtein", "correction", "hybrid_correction", "romanized_correction"} or (candidate.get("is_phrase") and word.lower() != q.lower()) else 0.0
    source_rom = 1.0 if source in {"romanized_lexicon", "romanized_correction"} or candidate.get("type") == "romanized" else 0.0
    provenance_count = float(len(candidate.get("provenance", [1])))

    # Error type
    err_type = candidate.get("error_type", "none")
    is_corr = 1.0 if (err_type and err_type != "none") or (candidate.get("is_phrase") and word.lower() != q.lower()) else 0.0

    # OOV / Protection signals from Feature 7
    is_oov = 1.0 if candidate.get("oov_status") == "oov" else 0.0
    is_rare = 1.0 if candidate.get("rare_status") == "rare" else 0.0
    should_protect = 1.0 if candidate.get("should_protect") else 0.0

    # Baseline Hybrid Score from Feature 9
    hybrid_score = float(candidate.get("hybrid_score", candidate.get("global_score", 0.0)))

    return [
        lexical_sim,
        edit_dist,
        edit_sim,
        is_exact,
        is_prefix,
        len_diff,
        len_ratio,
        freq,
        log_freq,
        user_score,
        semantic_score,
        context_score,
        has_recent,
        intent_score,
        intent_nav,
        intent_q,
        intent_tx,
        intent_local,
        intent_gen,
        source_auto,
        source_corr,
        source_rom,
        provenance_count,
        is_corr,
        is_oov,
        is_rare,
        should_protect,
        hybrid_score,
    ]


class DecisionNode:
    """A node in a decision regression tree."""

    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeRegressor:
    """Lightweight pure-Python decision tree regressor for ranking."""

    def __init__(self, max_depth=5, min_samples_split=4, max_features="sqrt"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y, feature_importances=None):
        self.root = self._build_tree(X, y, depth=0, feature_importances=feature_importances)

    def _build_tree(self, X, y, depth, feature_importances=None):
        n_samples = len(y)
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(set(y)) == 1:
            return DecisionNode(value=sum(y) / max(n_samples, 1))

        n_features = len(X[0])
        if self.max_features == "sqrt":
            k = max(1, int(math.sqrt(n_features)))
        else:
            k = n_features

        feature_indices = random.sample(range(n_features), min(k, n_features))
        best_feat, best_thresh, best_var_red = None, None, 0.0
        current_variance = self._variance(y)

        for feat_idx in feature_indices:
            values = [row[feat_idx] for row in X]
            thresholds = set(values)
            if len(thresholds) > 10:
                # Sample 10 quantiles for speed
                sorted_vals = sorted(thresholds)
                step = max(1, len(sorted_vals) // 10)
                thresholds = sorted_vals[::step]

            for thresh in thresholds:
                left_y = [y[i] for i in range(n_samples) if X[i][feat_idx] <= thresh]
                right_y = [y[i] for i in range(n_samples) if X[i][feat_idx] > thresh]
                if not left_y or not right_y:
                    continue

                var_red = current_variance - (
                    (len(left_y) / n_samples) * self._variance(left_y)
                    + (len(right_y) / n_samples) * self._variance(right_y)
                )

                if var_red > best_var_red:
                    best_var_red = var_red
                    best_feat = feat_idx
                    best_thresh = thresh

        if best_feat is None or best_var_red <= 1e-7:
            return DecisionNode(value=sum(y) / max(n_samples, 1))

        if feature_importances is not None:
            feature_importances[best_feat] += best_var_red * n_samples

        left_X = [X[i] for i in range(n_samples) if X[i][best_feat] <= best_thresh]
        left_y = [y[i] for i in range(n_samples) if X[i][best_feat] <= best_thresh]
        right_X = [X[i] for i in range(n_samples) if X[i][best_feat] > best_thresh]
        right_y = [y[i] for i in range(n_samples) if X[i][best_feat] > best_thresh]

        left_child = self._build_tree(left_X, left_y, depth + 1, feature_importances)
        right_child = self._build_tree(right_X, right_y, depth + 1, feature_importances)
        return DecisionNode(feature_idx=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _variance(self, y):
        n = len(y)
        if n <= 1:
            return 0.0
        mean = sum(y) / n
        return sum((val - mean) ** 2 for val in y) / n

    def predict_one(self, x):
        node = self.root
        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


class RandomForestRanker:
    """Random Forest regressor for candidate reranking with feature importance."""

    def __init__(self, n_estimators=35, max_depth=5, min_samples_split=4, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = [0.0] * len(FEATURE_NAMES)

    def fit(self, X, y):
        if not X or not y:
            return self

        random.seed(self.random_state)
        n_samples = len(X)
        self.trees = []
        raw_importances = [0.0] * len(FEATURE_NAMES)

        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
            sample_X = [X[i] for i in indices]
            sample_y = [y[i] for i in indices]

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features="sqrt"
            )
            tree.fit(sample_X, sample_y, feature_importances=raw_importances)
            self.trees.append(tree)

        # Normalize feature importances
        total_imp = sum(raw_importances)
        if total_imp > 0:
            self.feature_importances_ = [imp / total_imp for imp in raw_importances]
        else:
            self.feature_importances_ = [1.0 / len(FEATURE_NAMES)] * len(FEATURE_NAMES)

        return self

    def predict(self, X):
        if not self.trees or not X:
            return [0.0] * len(X)

        predictions = []
        for x in X:
            tree_preds = [tree.predict_one(x) for tree in self.trees]
            predictions.append(sum(tree_preds) / len(tree_preds))
        return predictions


def train_model(X, y, n_estimators=35, max_depth=5, random_state=42):
    """Train and return a RandomForestRanker model."""
    model = RandomForestRanker(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X, y)
    return model


def predict_scores(model, candidate_features):
    """Predict continuous relevance scores for a list of feature vectors."""
    if model is None or not candidate_features:
        return [0.0] * len(candidate_features)
    return model.predict(candidate_features)


def save_model(model, filepath="models/ml_ranker.pkl"):
    """Serialize the trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath="models/ml_ranker.pkl"):
    """Load a serialized ML ranker model from disk if present."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def get_feature_importance(model):
    """Return dictionary of feature importances {feature_name: importance}."""
    if model is None or not hasattr(model, "feature_importances_"):
        return {name: 1.0 / len(FEATURE_NAMES) for name in FEATURE_NAMES}

    return {
        name: round(float(imp), 4)
        for name, imp in zip(FEATURE_NAMES, model.feature_importances_)
    }


def rerank_candidates(query, candidates, model=None, alpha=0.35):
    """Rerank candidates using learned ML relevance scores with dynamic adaptation.

    Formula:
        final_score = alpha * normalized_hybrid_score + (1 - alpha) * normalized_ml_score

    Deterministic Tie-Breaking:
        1. final_score descending
        2. ml_score descending
        3. hybrid_score descending
        4. context_adjusted_score descending
        5. word alphabetically ascending
    """
    if not candidates:
        return []

    if model is None:
        # Fallback to hybrid ranking if ML model is unavailable
        return [
            {
                **c,
                "ml_score": 0.0,
                "final_score": c.get("hybrid_score", c.get("global_score", 0.0)),
                "ml_features": dict(zip(FEATURE_NAMES, extract_features(c, query))),
            }
            for c in candidates
        ]

    # 1. Extract feature vectors
    feature_matrix = [extract_features(c, query) for c in candidates]

    # 2. Predict ML relevance scores
    ml_scores = model.predict(feature_matrix)

    # 3. Dynamic Score Adaptation
    reranked = []
    max_ml = max(ml_scores) if ml_scores else 1.0
    min_ml = min(ml_scores) if ml_scores else 0.0
    ml_range = max_ml - min_ml if (max_ml - min_ml) > 1e-6 else 1.0
    is_phrase_query = (" " in str(query).strip()) or any(c.get("is_phrase") for c in candidates)
    effective_alpha = 0.95 if is_phrase_query else alpha

    for candidate, raw_ml_score, feat_vec in zip(candidates, ml_scores, feature_matrix):
        if is_phrase_query:
            norm_ml = max(0.0, min(1.0, raw_ml_score / 3.0))
        elif ml_range >= 0.25:
            norm_ml = max(0.0, min(1.0, (raw_ml_score - min_ml) / ml_range))
        else:
            norm_ml = max(0.0, min(1.0, raw_ml_score / 3.0))

        hybrid_score = candidate.get("hybrid_score", candidate.get("global_score", 0.0))
        final_score = effective_alpha * hybrid_score + (1.0 - effective_alpha) * norm_ml

        item = dict(candidate)
        item.update({
            "ml_score": round(raw_ml_score, 4),
            "final_score": round(final_score, 4),
            "ml_features": dict(zip(FEATURE_NAMES, feat_vec)),
        })
        reranked.append(item)

    # 4. Deterministic Tie-Breaking Sort
    def _tie_break_key(item):
        f_score = item.get("final_score", 0.0)
        ml_s = item.get("ml_score", 0.0)
        h_score = item.get("hybrid_score", item.get("global_score", 0.0))
        ctx_adj = item.get("context_adjusted_score", item.get("context_score", 0.0))
        edit_dist = item.get("edit_distance")
        if edit_dist is None:
            edit_dist = 0 if item.get("source") in {"trie_autocomplete", "romanized_lexicon"} else 99
        word = item.get("word", "")
        return (-f_score, -ml_s, -h_score, -ctx_adj, edit_dist, word)

    reranked.sort(key=_tie_break_key)
    return reranked
