"""Style / source probes: how much of each benchmark can be solved without understanding injections?

  * label probes   - 5-fold CV AUROC of (a) shallow style features and (b) a char n-gram TF-IDF logistic regression,
                     trained and tested *inside* a benchmark. High AUROC => positives and negatives differ in style/source.
  * source probes  - same, but separating two sources that share the SAME label (e.g. NotInject vs Deepset benign).
                     High AUROC => a classifier can recognise the source regardless of the label (shortcut available).
Output: results/style_probe.csv
"""
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

import config as C
from data import load_splits


def style_features(texts):
    rows = []
    for t in map(str, texts):
        n = max(len(t), 1)
        words = t.split()
        rows.append([
            len(t), len(words), np.mean([len(w) for w in words]) if words else 0,
            sum(c.isupper() for c in t) / n, sum(c.isdigit() for c in t) / n,
            sum((not c.isalnum()) and (not c.isspace()) for c in t) / n,
            sum(ord(c) > 127 for c in t) / n, int(t.startswith("Please")), int("\n" in t),
            len(re.findall(r"[.!?]", t)), int(t.rstrip().endswith("?")), int(t.rstrip().endswith(".")),
        ])
    return np.array(rows, dtype=float)


def cv_auc(texts, y, seed=0):
    y = np.asarray(y)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    X = style_features(texts)
    p1 = cross_val_predict(GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed), X, y, cv=cv,
                           method="predict_proba")[:, 1]
    pipe = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    p2 = cross_val_predict(pipe, list(map(str, texts)), y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p1), roc_auc_score(y, p2)


def main():
    sp = load_splits()
    hack, lvl, dn = sp["bench_hackaprompt"], sp["bench_hack_lvl"], sp["bench_deepset_notinject"]
    ni = dn[dn.source.str.startswith("notinject")]
    ds_b = dn[dn.source == "deepset_test_benign"]
    ds_a = dn[dn.source == "deepset_test_attack"]
    alp = hack[hack.source == "alpaca"]
    hk = hack[hack.source == "hackaprompt"]
    hk_l = lvl[lvl.source == "hackaprompt"]
    ds_only = dn[dn.source.str.startswith("deepset")]

    def two(a, b):  # b is the positive group of the probe
        d = pd.concat([a, b], ignore_index=True)
        return d.prompt_text.tolist(), np.r_[np.zeros(len(a)), np.ones(len(b))]

    probes = [
        # (kind, name, texts, labels)
        ("label", "HackAPrompt benchmark: attacks vs Alpaca benign", hack.prompt_text.tolist(), hack.label.to_numpy()),
        ("label", "Level-split benchmark: unseen-level attacks vs Alpaca benign", lvl.prompt_text.tolist(), lvl.label.to_numpy()),
        ("label", "Deepset+NotInject benchmark: attacks vs benign (all)", dn.prompt_text.tolist(), dn.label.to_numpy()),
        ("label", "Deepset only (source-matched): attacks vs benign", ds_only.prompt_text.tolist(), ds_only.label.to_numpy()),
        ("source|benign", "NotInject vs Deepset benign", *two(ds_b, ni)),
        ("source|benign", "Alpaca vs Deepset benign", *two(ds_b, alp)),
        ("source|benign", "Alpaca vs NotInject", *two(ni, alp)),
        ("source|attack", "HackAPrompt vs Deepset attacks", *two(ds_a, hk)),
    ]
    rows = []
    for kind, name, texts, y in probes:
        a_style, a_char = cv_auc(texts, y)
        rows.append({"kind": kind, "probe": name, "n_neg": int((y == 0).sum()), "n_pos": int((y == 1).sum()),
                     "AUROC style features": round(a_style, 3), "AUROC char-ngram LR": round(a_char, 3)})
        print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS_DIR / "style_probe.csv", index=False)
    return out


if __name__ == "__main__":
    main()
