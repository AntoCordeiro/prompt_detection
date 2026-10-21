"""Baselines: keyword/regex heuristic and TF-IDF n-gram logistic regression."""
import re
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import Detector

TRIGGERS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier|preceding)",
    r"disregard", r"system prompt", r"you are now", r"developer mode", r"jailbreak",
    r"\bDAN\b", r"act as", r"pretend (to be|you)", r"forget (everything|all|your|the)",
    r"bypass", r"override", r"new instructions?", r"reveal (your|the) (prompt|instructions)",
    r"do anything now", r"stay in character", r"without (any )?(restrictions|limitations|filters)",
    r"vergiss|ignoriere|ignorieren",  # Deepset contains German
]
_RX = re.compile("|".join(TRIGGERS), re.IGNORECASE)


class RegexHeuristic(Detector):
    """Score = 1 - exp(-#matches): any single match gives 0.63 (>0.5); more matches rank higher,
    so AUROC is meaningful (PLAN C8). Deliberately naive."""
    name = "Regex"

    def __init__(self, **_):
        super().__init__()
        self.meta = {"n_params": 0, "load_time_s": 0.0, "peak_vram_gb": 0.0}

    def score(self, texts):
        return np.array([1 - np.exp(-len(_RX.findall(str(t)))) for t in texts])


class TfidfLR(Detector):
    name = "TF-IDF+LR"

    def __init__(self, train_df, seed=0, **_):
        super().__init__()
        t = time.perf_counter()
        self.vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
        X = self.vec.fit_transform(train_df["prompt_text"])
        self.lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        self.lr.fit(X, train_df["label"])
        self.meta = {"n_params": int(self.lr.coef_.size), "load_time_s": time.perf_counter() - t, "peak_vram_gb": 0.0}

    def score(self, texts):
        return self.lr.predict_proba(self.vec.transform(list(texts)))[:, 1]
