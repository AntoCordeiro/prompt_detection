"""Build train/val/test splits (PLAN C0, C1, C5, D1) and run leakage checks.

Benchmarks
  bench_hackaprompt     : held-out HackAPrompt attacks (1) + generic benign Alpaca prompts (0)
  bench_deepset_notinject: Deepset test (attacks + benign) + all NotInject (benign, trigger words)
  notinject_heldout     : 60% of NotInject, never seen even in the hard-negative ablation
Training sets
  train / validation    : Deepset train + HackAPrompt pool + Alpaca benign pool (headline models)
  train_hn              : train + 40% of NotInject (D1 ablation, evaluated on notinject_heldout only)
"""
import re
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import config as C

COLS = ["prompt_text", "label", "source"]


def _norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _load_deepset():
    ds = load_dataset("deepset/prompt-injections")
    out = {}
    for sp in ("train", "test"):
        df = ds[sp].to_pandas().rename(columns={"text": "prompt_text"})
        df["source"] = f"deepset_{sp}_" + np.where(df["label"] == 1, "attack", "benign")
        out[sp] = df[COLS]
    return out["train"], out["test"]


def _load_hackaprompt():
    df = load_dataset("hackaprompt/hackaprompt-dataset", split="train").to_pandas()
    df = df[(df["correct"] == True) & df["user_input"].notna()].copy()  # noqa: E712
    df["prompt_text"] = df["user_input"].astype(str).str.strip()
    df = df[df["prompt_text"].str.len() > 5]
    df = df.drop_duplicates(subset="prompt_text")
    df = df[~df["prompt_text"].map(_norm).duplicated()]
    df["label"], df["source"] = 1, "hackaprompt"
    return df[COLS]


def _load_notinject():
    ds = load_dataset("leolee99/NotInject")
    parts = []
    for name in ds.keys():
        d = ds[name].to_pandas().rename(columns={"prompt": "prompt_text"})
        d["label"], d["source"] = 0, f"notinject_{name}"
        parts.append(d[COLS])
    return pd.concat(parts, ignore_index=True)


def _load_alpaca():
    df = load_dataset("yahma/alpaca-cleaned", split="train").to_pandas()
    txt = np.where(df["input"].str.len() > 0, df["instruction"] + "\n" + df["input"], df["instruction"])
    df = pd.DataFrame({"prompt_text": txt})
    df = df[df["prompt_text"].str.len().between(10, 1500)].drop_duplicates()
    df["label"], df["source"] = 0, "alpaca"
    return df[COLS]


def _near_dup_mask(train_txt, test_txt, thresh):
    """True for train rows that are (near-)duplicates of any test row."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit(list(train_txt) + list(test_txt))
    A, B = vec.transform(train_txt), vec.transform(test_txt)
    mask = np.zeros(A.shape[0], dtype=bool)
    for i in range(0, A.shape[0], 500):
        sims = (A[i:i + 500] @ B.T).toarray()
        mask[i:i + 500] = sims.max(axis=1) >= thresh
    return mask


def build_splits():
    rng = C.DATA_SEED
    ds_train, ds_test = _load_deepset()
    hack = _load_hackaprompt().sample(frac=1.0, random_state=rng).reset_index(drop=True)
    ni = _load_notinject()
    alp = _load_alpaca().sample(frac=1.0, random_state=rng).reset_index(drop=True)

    hack_train = hack.iloc[:C.N_HACK_TRAIN]
    hack_test = hack.iloc[C.N_HACK_TRAIN:C.N_HACK_TRAIN + C.N_HACK_TEST]
    alp_train = alp.iloc[:C.N_ALPACA_TRAIN]
    alp_test = alp.iloc[C.N_ALPACA_TRAIN:C.N_ALPACA_TRAIN + C.N_ALPACA_TEST]

    # NotInject: stratified 40/60 split for the D1 ablation
    ni_hn, ni_held = train_test_split(ni, train_size=C.HARD_NEG_FRAC, random_state=rng, stratify=ni["source"])

    bench_hack = pd.concat([hack_test, alp_test], ignore_index=True)
    bench_dn = pd.concat([ds_test, ni], ignore_index=True)
    tests = {
        "bench_hackaprompt": bench_hack,
        "bench_deepset_notinject": bench_dn,
        "notinject_heldout": ni_held.reset_index(drop=True),
    }

    pool = pd.concat([ds_train, hack_train, alp_train], ignore_index=True)

    # ---- leakage (C5): drop train rows that duplicate any test/NotInject row ----
    all_test = pd.concat([bench_hack, bench_dn], ignore_index=True)
    test_norm = set(all_test["prompt_text"].map(_norm))
    exact = pool["prompt_text"].map(_norm).isin(test_norm).to_numpy()
    near = _near_dup_mask(pool["prompt_text"].tolist(), all_test["prompt_text"].tolist(), C.NEAR_DUP_THRESH)
    drop = exact | near
    print(f"[leakage] train pool {len(pool)} rows: exact dups={exact.sum()}, near dups(>={C.NEAR_DUP_THRESH})={near.sum()}"
          f" -> dropping {drop.sum()} train rows")
    pool = pool[~drop].reset_index(drop=True)
    # within-test duplicates between the two benchmarks (informational)
    dup_bench = bench_hack["prompt_text"].map(_norm).isin(set(bench_dn["prompt_text"].map(_norm))).sum()
    print(f"[leakage] rows shared between bench_hackaprompt and bench_deepset_notinject: {dup_bench}")

    strat = pool["label"].astype(str) + pool["source"].str.split("_").str[0]
    train, val = train_test_split(pool, test_size=C.VAL_FRAC, random_state=rng, stratify=strat)
    train, val = train.reset_index(drop=True), val.reset_index(drop=True)
    train_hn = pd.concat([train, ni_hn], ignore_index=True).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    # D1: the ablation validation set also gets hard negatives (disjoint from training ones)
    ni_hn_val = ni_hn.sample(frac=0.2, random_state=rng)
    val_hn = pd.concat([val, ni_hn_val], ignore_index=True)
    train_hn = train_hn[~train_hn["prompt_text"].isin(ni_hn_val["prompt_text"])].reset_index(drop=True)

    splits = {"train": train, "validation": val, "train_hn": train_hn, "validation_hn": val_hn, **tests}
    # assertions: no test text appears in any training split
    ni_hn_norm = set(ni_hn["prompt_text"].map(_norm))
    assert not (set(train["prompt_text"].map(_norm)) & test_norm), "train leaks into test"
    assert not ((set(train_hn["prompt_text"].map(_norm)) & test_norm) - ni_hn_norm), "train_hn leaks beyond NotInject-hn"
    assert not (set(ni_held["prompt_text"].map(_norm)) & set(train_hn["prompt_text"].map(_norm))), "heldout NotInject leaks"
    for name, df in splits.items():
        df.to_json(C.DATA_DIR / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
        print(f"{name:26s} n={len(df):5d} benign={int((df.label == 0).sum()):4d} malicious={int((df.label == 1).sum()):4d}")
    return splits


def load_splits():
    names = ["train", "validation", "train_hn", "validation_hn",
             "bench_hackaprompt", "bench_deepset_notinject", "notinject_heldout"]
    missing = [n for n in names if not (C.DATA_DIR / f"{n}.jsonl").exists()]
    if missing:
        build_splits()
    return {n: pd.read_json(C.DATA_DIR / f"{n}.jsonl", lines=True) for n in names}


if __name__ == "__main__":
    build_splits()
    sys.exit(0)
