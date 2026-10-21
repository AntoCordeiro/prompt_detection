"""QLoRA hyper-parameter ablation (PLAN D3, D6) on Qwen2.5-1.5B, seed 0, headline training data.
    python ablate.py          -> results/ablation.csv   (resumable)
"""
import pandas as pd

import config as C
from common import free_gpu, set_seed
from data import load_splits
from metrics import binary_metrics, pick_threshold

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
CONFIGS = {
    "base (r16, lr2e-4, ep3, all-linear, L128)": {},
    "r8": {"r": 8, "alpha": 16},
    "r32": {"r": 32, "alpha": 64},
    "lr1e-4": {"lr": 1e-4},
    "lr5e-4": {"lr": 5e-4},
    "epochs2": {"epochs": 2},
    "epochs5": {"epochs": 5},
    "attention-only LoRA": {"targets": ATTN},
    "max_length 256": {"max_length_override": 256},
}


def main():
    import qlora
    splits = load_splits()
    out = C.RESULTS_DIR / "ablation.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    for name, ov in CONFIGS.items():
        if len(done) and name in set(done["config"]):
            continue
        print("\n###", name)
        set_seed(0)
        det = qlora.build("Qwen1.5B-QLoRA", seed=0, splits=splits, overrides=ov, tag="_abl_" + name.split()[0].replace("/", ""))
        with det:
            sv = det.score(splits["validation"].prompt_text.tolist())
            thr = pick_threshold(splits["validation"].label, sv, "f1")
            row = {"config": name, "val_F1": binary_metrics(splits["validation"].label, sv)["F1"],
                   "train_time_s": det.meta["train_time_s"], "peak_vram_gb": det.meta["peak_vram_gb"]}
            for b in ("bench_hackaprompt", "bench_deepset_notinject", "notinject_heldout"):
                s = det.score(splits[b].prompt_text.tolist())
                m = binary_metrics(splits[b].label, s)
                row[f"{b}_F1"], row[f"{b}_FPR"], row[f"{b}_AUROC"] = m["F1"], m["FPR"], m["AUROC"]
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        free_gpu()
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
