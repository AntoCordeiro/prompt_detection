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


def _load_hackaprompt(keep_level=False):
    df = load_dataset("hackaprompt/hackaprompt-dataset", split="train").to_pandas()
    df = df[(df["correct"] == True) & df["user_input"].notna()].copy()  # noqa: E712
    df["prompt_text"] = df["user_input"].astype(str).str.strip()
    df = df[df["prompt_text"].str.len() > 5]
    df = df.drop_duplicates(subset="prompt_text")
    df = df[~df["prompt_text"].map(_norm).duplicated()]
    df["label"], df["source"] = 1, "hackaprompt"
    return df[COLS + ["level"]] if keep_level else df[COLS]


def _load_notinject(keep_words=False):
    ds = load_dataset("leolee99/NotInject")
    parts = []
    for name in ds.keys():
        d = ds[name].to_pandas().rename(columns={"prompt": "prompt_text"})
        d["label"], d["source"] = 0, f"notinject_{name}"
        parts.append(d[COLS + ["word_list"]] if keep_words else d[COLS])
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


# ------------------------------------------------------------------ v2 protocols (additive)
EXTRA_NAMES = ["train_lvl", "validation_lvl", "train_noHack", "validation_noHack",
               "train_hackOnly", "validation_hackOnly", "train_hnWord", "validation_hnWord",
               "bench_hack_lvl", "notinject_heldout_word"]


def split_names(key):
    """(train split, validation split) used by a model key, e.g. 'Qwen3B-QLoRA+HN@hnWord' -> train_hnWord."""
    base, _, proto = key.partition("@")
    if proto:
        return f"train_{proto}", f"validation_{proto}"
    if base.endswith("+HN"):
        return "train_hn", "validation_hn"
    return "train", "validation"


def _strat_split(pool, rng):
    strat = pool["label"].astype(str) + pool["source"].str.split("_").str[0]
    tr, va = train_test_split(pool, test_size=C.VAL_FRAC, random_state=rng, stratify=strat)
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def build_extra_splits():
    """Adds the level / out-of-distribution / trigger-word splits next to the existing ones (which are not modified)."""
    rng = C.DATA_SEED
    base = {n: pd.read_json(C.DATA_DIR / f"{n}.jsonl", lines=True) for n in
            ["train", "validation", "bench_hackaprompt", "bench_deepset_notinject"]}
    old_pool = pd.concat([base["train"], base["validation"]], ignore_index=True)
    bench_hack, bench_dn = base["bench_hackaprompt"], base["bench_deepset_notinject"]

    # ---- level split: test attacks come from levels never used for training ----------------------
    hack = _load_hackaprompt(keep_level=True).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    in_old_train = set(old_pool["prompt_text"].map(_norm))
    in_old_bench = set(bench_hack["prompt_text"].map(_norm)) | set(bench_dn["prompt_text"].map(_norm))
    hack["_n"] = hack["prompt_text"].map(_norm)
    # test attacks must not appear verbatim in the old training pool, so old (random-split) models stay comparable
    test_atk = hack[hack.level.isin(C.TEST_LEVELS) & ~hack._n.isin(in_old_train)].iloc[:C.N_HACK_TEST]
    alp_test = bench_hack[bench_hack.source == "alpaca"]          # identical benign half as the old HackAPrompt benchmark
    bench_lvl = pd.concat([test_atk[COLS], alp_test], ignore_index=True)
    lvl_norm = set(bench_lvl["prompt_text"].map(_norm))
    tr_atk = hack[~hack.level.isin(C.TEST_LEVELS) & ~hack._n.isin(in_old_bench | lvl_norm)].iloc[:C.N_HACK_TRAIN]

    all_test = pd.concat([bench_hack, bench_dn, bench_lvl], ignore_index=True)
    test_norm = set(all_test["prompt_text"].map(_norm))
    non_hack = old_pool[old_pool.source != "hackaprompt"]
    pool_lvl = pd.concat([non_hack, tr_atk[COLS]], ignore_index=True)

    def guard(pool, name):
        drop = pool["prompt_text"].map(_norm).isin(test_norm).to_numpy() | \
            _near_dup_mask(pool["prompt_text"].tolist(), all_test["prompt_text"].tolist(), C.NEAR_DUP_THRESH)
        print(f"[leakage:{name}] dropped {drop.sum()} of {len(pool)} train rows")
        return pool[~drop].reset_index(drop=True)

    out = {"bench_hack_lvl": bench_lvl.reset_index(drop=True)}
    out["train_lvl"], out["validation_lvl"] = _strat_split(guard(pool_lvl, "lvl"), rng)
    print("level split: test levels", C.TEST_LEVELS, "| train levels", sorted(set(range(10)) - set(C.TEST_LEVELS)))
    # ---- out-of-distribution protocols ---------------------------------------------------------
    out["train_noHack"], out["validation_noHack"] = _strat_split(guard(non_hack, "noHack"), rng)
    only_hack = old_pool[old_pool.source.isin(["hackaprompt", "alpaca"])]
    out["train_hackOnly"], out["validation_hackOnly"] = _strat_split(guard(only_hack, "hackOnly"), rng)
    # ---- trigger-word split of NotInject ---------------------------------------------------------
    ni = _load_notinject(keep_words=True)
    ni["words"] = ni["word_list"].map(lambda w: [str(x).lower() for x in w])
    vocab = sorted({w for ws in ni["words"] for w in ws})
    r = np.random.default_rng(rng)
    held_vocab = set(r.choice(vocab, size=int(len(vocab) * C.HNW_HELD_FRAC), replace=False))
    held = ni[ni["words"].map(lambda ws: all(w in held_vocab for w in ws))]
    seen = ni[ni["words"].map(lambda ws: all(w not in held_vocab for w in ws))]
    print(f"NotInject trigger words: {len(vocab)} (held out {len(held_vocab)}); prompts: held-out-words={len(held)}, "
          f"train-words={len(seen)}, mixed (unused)={len(ni) - len(held) - len(seen)}")
    out["notinject_heldout_word"] = held[COLS].reset_index(drop=True)
    seen_tr, seen_va = train_test_split(seen[COLS], test_size=0.2, random_state=rng, stratify=seen["source"])
    out["train_hnWord"] = pd.concat([base["train"], seen_tr], ignore_index=True).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    out["validation_hnWord"] = pd.concat([base["validation"], seen_va], ignore_index=True)
    assert not (set(out["train_hnWord"].prompt_text.map(_norm)) & set(held.prompt_text.map(_norm))), "unseen-word prompts leak into training"
    for n, df in out.items():
        df.to_json(C.DATA_DIR / f"{n}.jsonl", orient="records", lines=True, force_ascii=False)
        print(f"{n:26s} n={len(df):5d} benign={int((df.label == 0).sum()):4d} malicious={int((df.label == 1).sum()):4d}")
    return out


def load_splits():
    names = ["train", "validation", "train_hn", "validation_hn",
             "bench_hackaprompt", "bench_deepset_notinject", "notinject_heldout"]
    missing = [n for n in names if not (C.DATA_DIR / f"{n}.jsonl").exists()]
    if missing:
        build_splits()
    if any(not (C.DATA_DIR / f"{n}.jsonl").exists() for n in EXTRA_NAMES):
        build_extra_splits()
    return {n: pd.read_json(C.DATA_DIR / f"{n}.jsonl", lines=True) for n in names + EXTRA_NAMES}


if __name__ == "__main__":
    if "--extra" in sys.argv:
        build_extra_splits()  # only adds the v2 splits; existing splits stay untouched
    else:
        build_splits()
    sys.exit(0)
