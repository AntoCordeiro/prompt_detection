"""Metrics, bootstrap CIs, threshold selection, latency protocol, results table."""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

import config as C
from common import cuda_sync


# ---------------------------------------------------------------- point metrics
def binary_metrics(y, s, thr=0.5):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "FPR": fp / (fp + tn) if (fp + tn) else np.nan,
        "AUROC": roc_auc_score(y, s) if both else np.nan,
        "AP": average_precision_score(y, s) if both else np.nan,
        "TPR@1%FPR": tpr_at_fpr(y, s, 0.01) if both else np.nan,
        "TPR@5%FPR": tpr_at_fpr(y, s, 0.05) if both else np.nan,
    }


def tpr_at_fpr(y, s, target):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= target].max())


def bootstrap_ci(y, s, thr=0.5, keys=("F1", "FPR", "AUROC", "Recall"), n=C.BOOTSTRAP_N, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = {k: [] for k in keys}
    for _ in range(n):  # stratified resampling keeps both classes present
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) if len(pos) and len(neg) \
            else rng.choice(len(y), len(y))
        m = binary_metrics(y[idx], s[idx], thr)
        for k in keys:
            vals[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}


def mcnemar_p(y, s_a, s_b, thr_a=0.5, thr_b=0.5):
    """Exact two-sided McNemar test on correctness of two detectors."""
    from scipy.stats import binomtest
    ca = (np.asarray(s_a) >= thr_a).astype(int) == np.asarray(y)
    cb = (np.asarray(s_b) >= thr_b).astype(int) == np.asarray(y)
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)


# ---------------------------------------------------------------- thresholds
def pick_threshold(y_val, s_val, mode="f1", fpr_target=C.FPR_TARGET):
    """Choose a decision threshold on validation scores.
    mode='f1'  : maximise F1.   mode='fpr': highest TPR subject to FPR <= fpr_target.
    Note: the validation set is small, so FPR targets below ~5% are not reliably estimable."""
    y_val, s_val = np.asarray(y_val), np.asarray(s_val, dtype=float)
    cands = np.unique(np.concatenate([s_val, [0.5]]))
    cands = np.concatenate([[cands.min() - 1e-9], cands, [cands.max() + 1e-9]])
    best, best_val = 0.5, -1
    for t in cands:
        m = binary_metrics(y_val, s_val, t)
        if mode == "f1":
            v = m["F1"]
        else:
            v = m["Recall"] if m["FPR"] <= fpr_target else -1 + (1 - m["FPR"]) * 1e-3
        if v >= best_val - 1e-12:  # ties -> highest threshold
            best, best_val = float(t), v
    return best


# ---------------------------------------------------------------- latency
def measure_latency(score_fn, texts, batch_sizes=C.LATENCY_BATCHES, n=C.LATENCY_N):
    """ms/sample at each batch size after a warm-up pass, with CUDA synchronisation."""
    texts = list(texts)[:n]
    out = {}
    score_fn(texts[:2])  # warm-up
    for bs in batch_sizes:
        cuda_sync()
        t = time.perf_counter()
        for i in range(0, len(texts), bs):
            score_fn(texts[i:i + bs])
        cuda_sync()
        out[f"latency_ms_bs{bs}"] = (time.perf_counter() - t) / len(texts) * 1000
    return out


# ---------------------------------------------------------------- results table
def load_predictions():
    frames = []
    for p in sorted(C.PRED_DIR.glob("*.parquet")):
        frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def hn_train_texts():
    """NotInject texts that '+HN' models saw during training or model selection (train_hn + validation_hn);
    they must be excluded from their test-benchmark evaluation."""
    texts = set()
    for name in ("train_hn", "validation_hn"):
        f = C.DATA_DIR / f"{name}.jsonl"
        if f.exists():
            d = pd.read_json(f, lines=True)
            texts |= set(d[d["source"].str.startswith("notinject")]["prompt_text"])
    return texts


_NI_CACHE = {}


def trained_ni_texts(key):
    """NotInject texts a model saw in its training/validation split (empty for models without hard negatives)."""
    from data import split_names
    names = split_names(key)
    if names not in _NI_CACHE:
        texts = set()
        for name in names:
            f = C.DATA_DIR / f"{name}.jsonl"
            if f.exists():
                d = pd.read_json(f, lines=True)
                texts |= set(d[d["source"].str.startswith("notinject")]["prompt_text"])
        _NI_CACHE[names] = texts
    return _NI_CACHE[names]


def exclude_trained(preds):
    """Drop, per model, the NotInject rows it trained/validated on from the test benchmarks (validation splits are kept)."""
    bad = pd.Series(False, index=preds.index)
    for key in preds["model"].unique():
        ni = trained_ni_texts(key)
        if ni:
            bad |= (preds["model"] == key) & ~preds["split"].str.startswith("validation") & preds["text"].isin(ni)
    return preds[~bad]


def hn_paired_table(preds, bench="bench_deepset_notinject", proto=""):
    """D1 ablation on a common row set: each trained family with and without hard negatives is scored on the
    same benchmark rows (those left after removing the NotInject rows the +HN model trained on), so F1/precision are
    comparable. proto='' -> random-split hard negatives; proto='hnWord' -> trigger-word-disjoint hard negatives."""
    suffix = "+HN" + (f"@{proto}" if proto else "")
    hn_models = [k for k in preds.model.unique() if k.endswith(suffix)]
    hn = set().union(*[trained_ni_texts(k) for k in hn_models]) if hn_models else set()
    t = preds[(preds.split == bench) & ~preds.text.isin(hn)]
    rows = []
    for (model, seed), g in t.groupby(["model", "seed"]):
        family = model[:-len(suffix)] if model.endswith(suffix) else model
        if "@" in family or "+HN" in family or f"{family}{suffix}" not in set(preds.model):
            continue
        rows.append({"family": family, "hard_negatives": model.endswith(suffix), "seed": seed, "n": len(g),
                     **binary_metrics(g.label, g.score)})
    return pd.DataFrame(rows)


def build_metrics_table(preds=None, benches=None, ci=True):
    """One row per (model, seed, benchmark) at the default (0.5) and the val-calibrated thresholds."""
    preds = load_predictions() if preds is None else preds
    benches = benches or ["bench_hackaprompt", "bench_deepset_notinject", "notinject_heldout",
                          "bench_hack_lvl", "notinject_heldout_word"]
    rows = []
    # D1: "+HN" models saw part of NotInject in training/validation -> exclude those rows from every benchmark
    preds = exclude_trained(preds)
    for (model, seed), g in preds.groupby(["model", "seed"]):
        from data import split_names
        val_name = split_names(model)[1]
        v = g[g.split == val_name]
        thr_f1 = pick_threshold(v.label, v.score, "f1") if len(v) else 0.5
        thr_fpr = pick_threshold(v.label, v.score, "fpr") if len(v) else 0.5
        for b in benches:
            t = g[g.split == b]
            if t.empty:
                continue
            for tname, thr in (("default", 0.5), ("val-F1", thr_f1), ("val-FPR<=5%", thr_fpr)):
                m = binary_metrics(t.label, t.score, thr)
                row = {"model": model, "seed": seed, "benchmark": b, "threshold_type": tname, "threshold": thr, **m}
                if ci and tname == "default":
                    for k, (lo, hi) in bootstrap_ci(t.label, t.score, thr).items():
                        row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
                rows.append(row)
    return pd.DataFrame(rows)


def per_source_rates(preds, thr=0.5):
    """Positive-rate per data source: FPR for benign sources, recall for attack sources (C1, C2)."""
    d = preds.copy()
    d["flag"] = (d.score >= thr).astype(int)
    d = d[d.split.isin(["bench_hackaprompt", "bench_deepset_notinject", "bench_hack_lvl"])]
    return d.groupby(["model", "seed", "split", "source"]).agg(n=("flag", "size"), flag_rate=("flag", "mean")).reset_index()


def load_latency():
    rows = [json.loads(p.read_text()) for p in sorted((C.RESULTS_DIR / "latency").glob("*.json"))]
    return pd.DataFrame(rows)
