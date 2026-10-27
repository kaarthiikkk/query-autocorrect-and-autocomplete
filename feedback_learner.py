"""Feature 12: User-Feedback Based Online Learning Module.

Records explicit and implicit user interactions:
- Selected / clicked suggestions (positive reinforcement)
- Accepted spelling corrections (query-target association)
- Rejected / dismissed suggestions (negative penalty)

Maintains persistent user feedback data in data/user_feedback.json and
updates data/user_freq.json. Computes bounded, normalized preference scores
that immediately influence Feature 9 Hybrid Ranking, Feature 10 ML Reranking,
and Feature 11 Dynamic Weights without requiring an application restart.
"""

import json
import os
import threading


DEFAULT_FEEDBACK_PATH = "data/user_feedback.json"
DEFAULT_USER_FREQ_PATH = "data/user_freq.json"


class FeedbackLearner:
    """Online learning engine that tracks user interactions and adapts ranking scores."""

    def __init__(
        self,
        feedback_filepath=DEFAULT_FEEDBACK_PATH,
        freq_filepath=DEFAULT_USER_FREQ_PATH,
        auto_save=True
    ):
        self.feedback_filepath = feedback_filepath
        self.freq_filepath = freq_filepath
        self.auto_save = auto_save
        self._lock = threading.RLock()

        # In-memory feedback stores
        self.word_feedback = {}       # {word: {"positive": int, "negative": int, "accepts": int}}
        self.query_affinity = {}      # {"query:::word": {"positive": int, "negative": int}}
        self.user_freq = {}           # {word: int} (effective integer frequency)

        self.load()

    def load(self):
        """Loads user feedback and frequency dictionaries from disk."""
        with self._lock:
            # Load user frequencies
            if os.path.exists(self.freq_filepath):
                try:
                    with open(self.freq_filepath, "r", encoding="utf-8") as f:
                        self.user_freq = json.load(f)
                except Exception:
                    self.user_freq = {}
            else:
                self.user_freq = {}

            # Load detailed interaction history
            if os.path.exists(self.feedback_filepath):
                try:
                    with open(self.feedback_filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.word_feedback = data.get("word_feedback", {})
                        self.query_affinity = data.get("query_affinity", {})
                except Exception:
                    self.word_feedback = {}
                    self.query_affinity = {}
            else:
                self.word_feedback = {}
                self.query_affinity = {}

    def save(self):
        """Persists user feedback and frequency counts atomically to disk."""
        with self._lock:
            os.makedirs(os.path.dirname(self.freq_filepath), exist_ok=True)
            try:
                # Write user_freq.json
                temp_freq = f"{self.freq_filepath}.tmp"
                with open(temp_freq, "w", encoding="utf-8") as f:
                    json.dump(self.user_freq, f, indent=4)
                os.replace(temp_freq, self.freq_filepath)

                # Write user_feedback.json
                temp_feedback = f"{self.feedback_filepath}.tmp"
                with open(temp_feedback, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "word_feedback": self.word_feedback,
                            "query_affinity": self.query_affinity,
                        },
                        f,
                        indent=4
                    )
                os.replace(temp_feedback, self.feedback_filepath)
            except Exception as e:
                print(f"Warning: Failed to save user feedback data: {e}")

    def record_selection(self, word, query=None):
        """Records a positive click/selection on a suggestion candidate."""
        word = str(word or "").strip().lower()
        if not word:
            return

        with self._lock:
            # Update word-level stats
            if word not in self.word_feedback:
                self.word_feedback[word] = {"positive": 0, "negative": 0, "accepts": 0}
            self.word_feedback[word]["positive"] += 1

            # Update query affinity if query provided
            if query:
                q_clean = str(query).strip().lower()
                key = f"{q_clean}:::{word}"
                if key not in self.query_affinity:
                    self.query_affinity[key] = {"positive": 0, "negative": 0}
                self.query_affinity[key]["positive"] += 1

            # Recompute effective user frequency
            net_pos = max(0, self.word_feedback[word]["positive"] - self.word_feedback[word]["negative"])
            self.user_freq[word] = net_pos

        if self.auto_save:
            self.save()

    def record_acceptance(self, query, correct_word):
        """Records user accepting a spelling correction for a query."""
        correct_word = str(correct_word or "").strip().lower()
        q_clean = str(query or "").strip().lower()
        if not correct_word:
            return

        with self._lock:
            if correct_word not in self.word_feedback:
                self.word_feedback[correct_word] = {"positive": 0, "negative": 0, "accepts": 0}
            self.word_feedback[correct_word]["positive"] += 2
            self.word_feedback[correct_word]["accepts"] += 1

            if q_clean:
                key = f"{q_clean}:::{correct_word}"
                if key not in self.query_affinity:
                    self.query_affinity[key] = {"positive": 0, "negative": 0}
                self.query_affinity[key]["positive"] += 2

            net_pos = max(0, self.word_feedback[correct_word]["positive"] - self.word_feedback[correct_word]["negative"])
            self.user_freq[correct_word] = net_pos

        if self.auto_save:
            self.save()

    def record_rejection(self, word, query=None):
        """Records user dismissing or rejecting an unselected suggestion."""
        word = str(word or "").strip().lower()
        if not word:
            return

        with self._lock:
            if word not in self.word_feedback:
                self.word_feedback[word] = {"positive": 0, "negative": 0, "accepts": 0}
            self.word_feedback[word]["negative"] += 1

            if query:
                q_clean = str(query).strip().lower()
                key = f"{q_clean}:::{word}"
                if key not in self.query_affinity:
                    self.query_affinity[key] = {"positive": 0, "negative": 0}
                self.query_affinity[key]["negative"] += 1

            # Update effective frequency (cannot go below 0)
            net_pos = max(0, self.word_feedback[word]["positive"] - self.word_feedback[word]["negative"])
            self.user_freq[word] = net_pos

        if self.auto_save:
            self.save()

    def record_interaction(
        self,
        query,
        chosen_word=None,
        rejected_words=None,
        interaction_type="select"
    ):
        """Batch records chosen candidate feedback and rejected candidate feedback."""
        if chosen_word:
            if interaction_type == "accept":
                self.record_acceptance(query, chosen_word)
            else:
                self.record_selection(chosen_word, query=query)

        if rejected_words:
            for rej in rejected_words:
                if rej != chosen_word:
                    self.record_rejection(rej, query=query)

    def get_user_score(self, word, query=None):
        """Calculates a bounded, normalized user preference score in [0.0, 1.0].

        Formula:
            net_positive = max(0, positive - 0.5 * negative)
            user_score = net_positive / (net_positive + 10)
        """
        word = str(word or "").strip().lower()
        if not word:
            return 0.0

        with self._lock:
            stats = self.word_feedback.get(word, {"positive": 0, "negative": 0})
            pos = stats["positive"]
            neg = stats["negative"]

            if query:
                q_clean = str(query).strip().lower()
                q_stats = self.query_affinity.get(f"{q_clean}:::{word}", {"positive": 0, "negative": 0})
                pos += q_stats["positive"] * 1.5
                neg += q_stats["negative"] * 0.5

            net = max(0.0, float(pos) - 0.5 * float(neg))
            score = net / (net + 10.0)
            return round(min(1.0, max(0.0, score)), 4)

    def get_user_frequencies(self):
        """Returns a snapshot of the user frequency dictionary for search pipelines."""
        with self._lock:
            return dict(self.user_freq)

    def get_feedback_summary(self, word=None):
        """Returns explainable statistics for inspection and reports."""
        with self._lock:
            if word:
                w_clean = str(word).strip().lower()
                return {
                    "word": w_clean,
                    "stats": self.word_feedback.get(w_clean, {"positive": 0, "negative": 0, "accepts": 0}),
                    "user_freq": self.user_freq.get(w_clean, 0),
                    "user_score": self.get_user_score(w_clean),
                }
            return {
                "total_tracked_words": len(self.word_feedback),
                "total_query_affinities": len(self.query_affinity),
                "user_freq_count": len(self.user_freq),
            }

    def clear(self):
        """Resets all in-memory feedback and frequencies (used for clean testing)."""
        with self._lock:
            self.word_feedback = {}
            self.query_affinity = {}
            self.user_freq = {}
        if self.auto_save:
            self.save()


# Global shared singleton instance for online learning
_GLOBAL_FEEDBACK_LEARNER = None


def get_feedback_learner(auto_save=True):
    """Returns the shared FeedbackLearner singleton instance."""
    global _GLOBAL_FEEDBACK_LEARNER
    if _GLOBAL_FEEDBACK_LEARNER is None:
        _GLOBAL_FEEDBACK_LEARNER = FeedbackLearner(auto_save=auto_save)
    return _GLOBAL_FEEDBACK_LEARNER
