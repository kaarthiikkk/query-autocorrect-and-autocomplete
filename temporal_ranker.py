"""Feature 14: Temporal / Trending Query Adaptation Module.

Provides:
  1. Temporal Query Detection  — identify time-sensitive queries
  2. Recency Scoring            — exponential decay over past interactions
  3. Trending Score             — frequency of recent interactions (short half-life)
  4. Bounded Temporal Boost     — controlled score adjustment on top of ML ranking
  5. Persistence                — atomic JSON storage in data/temporal_stats.json
  6. Explainability             — per-candidate temporal_reason field

Time Decay Formula
------------------
For each past interaction at timestamp t_i (POSIX seconds):

    decay_weight(t_i, halflife_s) = exp(-ln(2) / halflife_s  *  (T_now - t_i))

Recency score (halflife = 86400 s = 24 h):
    recency_score(w, q) = tanh(sum(decay_weights) / 5.0)   in [0, 1)

Trending score (halflife = 3600 s = 1 h):
    trending_score(w, q) = tanh(sum(decay_weights) / 3.0)  in [0, 1)

Temporal Boost Formula
-----------------------
    boost = min(MAX_BOOST, W_TREND * trending_score + W_REC * recency_score)
    MAX_BOOST = 0.20,  W_TREND = 0.60,  W_REC = 0.40

    temporal_adjusted_score = base_score + boost

Thresholds
----------
  temporal_confidence HIGH   >= 0.70
  temporal_confidence MEDIUM >= 0.40
  temporal_confidence LOW    <  0.40
  trending SIGNIFICANT       >= 0.10
"""

import json
import math
import os
import re
import threading
import time

DATA_DIR = "data"
TEMPORAL_STATS_FILE = os.path.join(DATA_DIR, "temporal_stats.json")

# Temporal boost parameters
MAX_BOOST = 0.20
W_TREND = 0.60
W_REC = 0.40

# Decay half-lives in seconds
RECENCY_HALFLIFE_S = 86400.0   # 24 hours
TRENDING_HALFLIFE_S = 3600.0   # 1 hour

_LAMBDA_REC = math.log(2) / RECENCY_HALFLIFE_S
_LAMBDA_TREND = math.log(2) / TRENDING_HALFLIFE_S

_TEMPORAL_PATTERNS = [
    # English
    (r"\btoday\b",                   "time_of_day"),
    (r"\btonight\b",                 "time_of_day"),
    (r"\byesterday\b",               "time_of_day"),
    (r"\btomorrow\b",                "time_of_day"),
    (r"\bnow\b",                     "immediacy"),
    (r"\bcurrent(?:ly)?\b",          "immediacy"),
    (r"\blatest\b",                  "recency"),
    (r"\brecent(?:ly)?\b",           "recency"),
    (r"\bnew(?:est)?\b",             "recency"),
    (r"\btrending\b",                "trending"),
    (r"\bviral\b",                   "trending"),
    (r"\bhot\b",                     "trending"),
    (r"\bpopular(?:ity)?\b",         "trending"),
    (r"\bnews\b",                    "news"),
    (r"\bbreaking\b",                "news"),
    (r"\blive\b",                    "live"),
    (r"\bupdate[ds]?\b",             "update"),
    (r"\bthis\s+week\b",             "period"),
    (r"\bthis\s+month\b",            "period"),
    (r"\bthis\s+year\b",             "period"),
    (r"\blast\s+week\b",             "period"),
    (r"\blast\s+month\b",            "period"),
    (r"\blast\s+year\b",             "period"),
    (r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", "date_reference"),
    (r"\b20\d{2}\b",                 "date_reference"),
    (r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", "weekday"),
    # Tamil Native Script
    (r"இன்று",                      "time_of_day"),
    (r"இப்போது|இப்போ",              "immediacy"),
    (r"புதிய|புதிதாக",               "recency"),
    (r"செய்தி|செய்திகள்",           "news"),
    (r"நேற்று",                      "time_of_day"),
    (r"நாளை",                       "time_of_day"),
    (r"வைரல்",                       "trending"),
    (r"டிரெண்டிங்",                  "trending"),
    (r"நேரலை",                       "live"),
    (r"முக்கிய",                     "news"),
    # Romanized Tamil / Indian
    (r"\biniku\b|\binniku\b",        "time_of_day"),
    (r"\bnethu\b",                   "time_of_day"),
    (r"\bnaalaikku\b|\bnalaiki\b",   "time_of_day"),
    (r"\bippo\b",                    "immediacy"),
    (r"\bputhusu\b|\bpudhu\b",       "recency"),
    (r"\bseithi\b|\bseithigal\b",    "news"),
]
_COMPILED_PATTERNS = [
    (re.compile(p, re.IGNORECASE), cat) for p, cat in _TEMPORAL_PATTERNS
]

_CATEGORY_WEIGHTS = {
    "immediacy":      0.90,
    "trending":       0.85,
    "live":           0.85,
    "news":           0.80,
    "time_of_day":    0.80,
    "recency":        0.75,
    "update":         0.70,
    "period":         0.65,
    "weekday":        0.55,
    "date_reference": 0.50,
}


def detect_temporal_query(query):
    """Detect temporal / trending intent in a query string (English, Tamil & Romanized).

    Returns dict with keys:
        temporal_detected   bool
        temporal_type       str or None
        temporal_signals    list[str]
        temporal_confidence float in [0, 1]
    """
    q = str(query or "").lower()
    matched_signals = []
    matched_categories = []

    for pattern, category in _COMPILED_PATTERNS:
        hits = pattern.findall(q)
        for hit in hits:
            text = hit if isinstance(hit, str) else " ".join(hit)
            matched_signals.append(text.strip())
            matched_categories.append(category)

    if not matched_signals:
        return {
            "temporal_detected": False,
            "temporal_type": None,
            "temporal_signals": [],
            "temporal_confidence": 0.0,
        }

    dominant_cat = max(set(matched_categories),
                       key=lambda c: _CATEGORY_WEIGHTS.get(c, 0.50))
    base_conf = _CATEGORY_WEIGHTS.get(dominant_cat, 0.50)
    n_cats = len(set(matched_categories))
    temporal_confidence = round(min(1.0, base_conf + (n_cats - 1) * 0.05), 4)

    return {
        "temporal_detected": True,
        "temporal_type": dominant_cat,
        "temporal_signals": list(dict.fromkeys(matched_signals)),
        "temporal_confidence": temporal_confidence,
    }


def _decay(t_now, t_event, lam):
    """Single exponential decay weight."""
    return math.exp(-lam * max(0.0, t_now - t_event))


class TemporalRanker:
    """Thread-safe temporal / trending ranker with atomic JSON persistence.

    Storage schema in temporal_stats.json:
        {
            "<word>": {
                "__global__": [t1, t2, ...],
                "<query_key>": [t3, t4, ...]
            }
        }
    """

    def __init__(self, stats_filepath=TEMPORAL_STATS_FILE, auto_save=True):
        self._lock = threading.RLock()
        self._filepath = stats_filepath
        self._auto_save = auto_save
        self._stats = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self):
        if os.path.exists(self._filepath):
            try:
                with open(self._filepath, "r", encoding="utf-8") as f:
                    self._stats = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._stats = {}

    def _save(self):
        dir_ = os.path.dirname(self._filepath)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        tmp = self._filepath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._stats, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._filepath)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_selection(self, word, query=None, timestamp=None):
        """Record that a user selected `word` (optionally in context of `query`)."""
        t = float(timestamp) if timestamp is not None else time.time()
        word = str(word).lower().strip()
        if not word:
            return
        with self._lock:
            entry = self._stats.setdefault(word, {})
            entry.setdefault("__global__", []).append(t)
            if query:
                q_key = str(query).lower().strip()[:80]
                entry.setdefault(q_key, []).append(t)
            if self._auto_save:
                self._save()

    def record_query(self, query, timestamp=None):
        """Record a search query submission to track topic query velocity."""
        t = float(timestamp) if timestamp is not None else time.time()
        q_key = str(query or "").lower().strip()[:80]
        if not q_key:
            return
        with self._lock:
            entry = self._stats.setdefault(f"__query__:{q_key}", {})
            entry.setdefault("__global__", []).append(t)
            if self._auto_save:
                self._save()

    # ------------------------------------------------------------------
    # Scoring & Velocity
    # ------------------------------------------------------------------

    def _raw_score(self, word, query, lam, t_now=None):
        """Sum of decay weights over all matching timestamps."""
        t_current = float(t_now) if t_now is not None else time.time()
        word = str(word).lower().strip()
        with self._lock:
            entry = self._stats.get(word, {})

        timestamps = list(entry.get("__global__", []))
        if query:
            q_key = str(query).lower().strip()[:80]
            timestamps += list(entry.get(q_key, []))
        return sum(_decay(t_current, t, lam) for t in timestamps)

    def get_recency_score(self, word, query=None, t_now=None):
        """Recency score with 24 h half-life, bounded in [0, 1) via tanh."""
        raw = self._raw_score(word, query, _LAMBDA_REC, t_now=t_now)
        return round(math.tanh(raw / 5.0), 6)

    def get_trending_score(self, word, query=None, t_now=None):
        """Trending score with 1 h half-life, bounded in [0, 1) via tanh."""
        raw = self._raw_score(word, query, _LAMBDA_TREND, t_now=t_now)
        return round(math.tanh(raw / 3.0), 6)

    def get_burst_velocity(self, word_or_query, query=None, t_now=None):
        """Compute burst velocity ratio: short-term (1h) rate vs 24h baseline rate.

        Returns (velocity_ratio, velocity_score, is_bursting)
        """
        t_current = float(t_now) if t_now is not None else time.time()
        key = str(word_or_query).lower().strip()
        with self._lock:
            entry = self._stats.get(key, {})

        timestamps = list(entry.get("__global__", []))
        if query:
            q_key = str(query).lower().strip()[:80]
            timestamps += list(entry.get(q_key, []))

        if not timestamps:
            return 0.0, 0.0, False

        count_1h = sum(1 for t in timestamps if (t_current - t) <= TRENDING_HALFLIFE_S)
        count_24h = sum(1 for t in timestamps if (t_current - t) <= RECENCY_HALFLIFE_S)

        baseline_hourly = max(1.0, count_24h / 24.0)
        ratio = round(count_1h / baseline_hourly, 4)
        velocity_score = round(math.tanh((ratio * math.sqrt(max(1, count_1h))) / 4.0), 6)
        is_bursting = (count_1h >= 2 and ratio >= 1.5) or (count_1h >= 5)

        return ratio, velocity_score, is_bursting

    def get_temporal_insights(self, word, query=None, t_now=None):
        """Detailed temporal breakdown for explainability and diagnostics."""
        t_current = float(t_now) if t_now is not None else time.time()
        recency = self.get_recency_score(word, query=query, t_now=t_current)
        trending = self.get_trending_score(word, query=query, t_now=t_current)
        ratio, vel_score, is_bursting = self.get_burst_velocity(word, query=query, t_now=t_current)

        with self._lock:
            entry = self._stats.get(str(word).lower().strip(), {})
            timestamps = entry.get("__global__", [])
            count = len(timestamps)
            last_t = max(timestamps) if timestamps else None

        seconds_ago = round(t_current - last_t, 1) if last_t else None

        return {
            "recency_score": recency,
            "trending_score": trending,
            "burst_ratio": ratio,
            "burst_velocity": vel_score,
            "is_bursting": is_bursting,
            "total_interactions": count,
            "last_interaction_seconds_ago": seconds_ago,
        }

    def get_top_trending(self, limit=10, min_score=0.05, t_now=None):
        """Retrieve globally top-trending topics/queries across the platform."""
        t_current = float(t_now) if t_now is not None else time.time()
        results = []

        with self._lock:
            keys = list(self._stats.keys())

        for k in keys:
            clean_name = k.replace("__query__:", "")
            trending = self.get_trending_score(k, t_now=t_current)
            recency = self.get_recency_score(k, t_now=t_current)
            ratio, vel_score, is_bursting = self.get_burst_velocity(k, t_now=t_current)

            combined_score = round(0.55 * trending + 0.30 * vel_score + 0.15 * recency, 4)
            if combined_score >= min_score or is_bursting:
                results.append({
                    "topic": clean_name,
                    "trending_score": trending,
                    "recency_score": recency,
                    "burst_velocity": vel_score,
                    "burst_ratio": ratio,
                    "is_bursting": is_bursting,
                    "combined_score": combined_score,
                })

        results.sort(key=lambda item: (item["is_bursting"], item["combined_score"]), reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Apply Temporal Boost
    # ------------------------------------------------------------------

    def apply_temporal_boost(self, candidates, raw_query, temporal_info):
        """Annotate each candidate with temporal fields, burst velocity, and re-sort.

        New fields per candidate:
            recency_score, trending_score, burst_velocity, is_bursting,
            temporal_boost, temporal_adjusted_score, temporal_reason
        """
        temporal_detected = temporal_info.get("temporal_detected", False)
        boosted = []

        for cand in candidates:
            cand_copy = dict(cand)
            word = str(cand_copy.get("word", "")).lower().strip()
            base = float(cand_copy.get("final_score",
                                       cand_copy.get("hybrid_score", 0.0)))

            recency = self.get_recency_score(word, raw_query)
            trending = self.get_trending_score(word, raw_query)
            ratio, vel_score, is_bursting = self.get_burst_velocity(word, raw_query)

            # Combined temporal signal with burst acceleration
            if temporal_detected:
                effective_trend = min(1.0, trending + (0.15 if is_bursting else 0.0))
                boost = min(MAX_BOOST, W_TREND * effective_trend + W_REC * recency)
            else:
                effective_rec = min(1.0, recency + (0.05 if is_bursting else 0.0))
                boost = min(0.08, W_REC * effective_rec)

            adjusted = round(min(1.0, base + boost), 6)

            if boost == 0.0:
                if temporal_detected:
                    reason = "temporal query detected but no recent activity for this candidate"
                else:
                    reason = "non-temporal query; no temporal boost applied"
            elif is_bursting:
                reason = (
                    f"🔥 trending burst detected! ratio={ratio:.1f}x, "
                    f"trending={trending:.3f}, recency={recency:.3f}, boost=+{boost:.4f}"
                )
            elif temporal_detected and (trending >= 0.10 or recency >= 0.10):
                reason = (
                    f"temporal boost applied ({temporal_info.get('temporal_type', 'temporal')}): "
                    f"trending={trending:.3f}, recency={recency:.3f}, boost=+{boost:.4f}"
                )
            elif recency > 0.0 or trending > 0.0:
                reason = (
                    f"mild recency micro-boost: trending={trending:.3f}, "
                    f"recency={recency:.3f}, boost=+{boost:.4f}"
                )
            else:
                reason = "no recent temporal activity for this candidate; no boost"

            cand_copy["recency_score"] = recency
            cand_copy["trending_score"] = trending
            cand_copy["burst_velocity"] = vel_score
            cand_copy["is_bursting"] = is_bursting
            cand_copy["temporal_boost"] = round(boost, 6)
            cand_copy["temporal_adjusted_score"] = adjusted
            cand_copy["temporal_reason"] = reason
            boosted.append(cand_copy)

        boosted.sort(key=lambda c: c["temporal_adjusted_score"], reverse=True)
        return boosted

    def get_stats_snapshot(self, word):
        """Return raw interaction timestamps for a word (testing / debugging)."""
        with self._lock:
            return dict(self._stats.get(str(word).lower().strip(), {}))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_GLOBAL_TEMPORAL_RANKER = None
_TEMPORAL_LOCK = threading.Lock()


def get_temporal_ranker():
    """Return the process-level TemporalRanker singleton."""
    global _GLOBAL_TEMPORAL_RANKER
    if _GLOBAL_TEMPORAL_RANKER is None:
        with _TEMPORAL_LOCK:
            if _GLOBAL_TEMPORAL_RANKER is None:
                _GLOBAL_TEMPORAL_RANKER = TemporalRanker()
    return _GLOBAL_TEMPORAL_RANKER
