"""Generates analysis.ipynb (run once: python make_notebook.py; then execute the notebook)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Detecting Direct Prompt Injections - analysis
Loads `results/` (per-sample predictions, latency, metrics) produced by `run_all.py`; no GPU needed.
Benchmarks: **HackAPrompt** (held-out HackAPrompt attacks vs. Alpaca benign), **Deepset+NotInject** (Deepset test + all NotInject),
**NotInject held-out** (60% of NotInject never used for training, benign only -> only FPR is defined).
`+HN` rows are the hard-negative ablation (40% of NotInject added to training/validation); those NotInject rows are excluded from their evaluation,
so their Deepset+NotInject F1/precision use fewer benign rows than the other models - section 10 compares both on a common row set.
""")
code("""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, re, json
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import config as C
from metrics import (load_predictions, exclude_trained, binary_metrics, mcnemar_p, per_source_rates, load_latency, tpr_at_fpr,
                     hn_paired_table)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 110)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})
M = pd.read_csv(C.RESULTS_DIR / "metrics.csv")
P = exclude_trained(load_predictions())
L = load_latency()
ORDER = ["Regex","TF-IDF+LR","TF-IDF+LR+HN","PromptGuard","PromptGuard-JB","ProtectAI","LlamaGuard-ZS","LlamaGuard-FS",
         "Qwen1.5B-ZS","Qwen1.5B-FS","Qwen3B-ZS","Qwen3B-FS","DeBERTa-FT","DeBERTa-FT+HN",
         "Qwen1.5B-QLoRA","Qwen3B-QLoRA","Qwen1.5B-QLoRA+HN","Qwen3B-QLoRA+HN"]
BENCH = {"bench_hackaprompt":"HackAPrompt (attacks vs Alpaca)", "bench_deepset_notinject":"Deepset + NotInject", "notinject_heldout":"NotInject held-out (benign only)"}
def fmt(g, k):
    m, s = g[k].mean(), g[k].std()
    return f"{m:.3f}" + (f" ± {s:.3f}" if g[k].count() > 1 and not np.isnan(s) else "")
""")
md("## 1. Main results (default threshold 0.5; Qwen = mean ± std over 3 seeds)")
code("""
d = M[M.threshold_type == "default"]
TABLES = {}
for b, title in BENCH.items():
    cols = ["Accuracy","Precision","Recall","F1","FPR","AUROC","TPR@5%FPR"] if b != "notinject_heldout" else ["FPR"]
    rows = []
    for m in ORDER:
        g = d[(d.model == m) & (d.benchmark == b)]
        if len(g): rows.append({"Model": m, **{k: fmt(g, k) for k in cols}})
    TABLES[b] = pd.DataFrame(rows).set_index("Model")
    print("\\n###", title); display(TABLES[b])
""")
md("## 2. Bootstrap 95% CIs (seed 0, default threshold)")
code("""
rows = []
for m in ORDER:
    for b in ["bench_hackaprompt","bench_deepset_notinject"]:
        g = d[(d.model == m) & (d.benchmark == b) & (d.seed == 0)]
        if g.empty: continue
        r = g.iloc[0]
        rows.append({"Model": m, "Benchmark": b, **{k: f"{r[k]:.3f} [{r[k+'_lo']:.3f}, {r[k+'_hi']:.3f}]" for k in ["F1","FPR","AUROC"]}})
CI = pd.DataFrame(rows); display(CI)
""")
md("""
## 3. Validation-calibrated thresholds
Thresholds are chosen on the validation split (max F1, or highest recall with FPR<=5%). The validation set has only ~110 benign samples
(one false positive ~ 1%), so FPR targets below 5% are not estimable, and it contains no NotInject-style negatives (except in the `+HN` runs),
so calibration cannot fix over-defense by itself.
""")
code("""
c = M[M.benchmark == "bench_deepset_notinject"].groupby(["model","threshold_type"])[["threshold","Recall","F1","FPR"]].mean().round(3)
display(c.loc[[m for m in ORDER if m in c.index.get_level_values(0)]])
""")
md("## 4. Over-defense breakdown (C1, C2)")
code("""
src = per_source_rates(P)
piv = src[src.split == "bench_deepset_notinject"].groupby(["model","source"]).flag_rate.mean().unstack().round(3)
piv = piv.loc[[m for m in ORDER if m in piv.index]]
print("Flag rate per source in Deepset+NotInject: FPR for *_benign / notinject_*, recall for deepset_test_attack")
display(piv)
hp = src[src.split == "bench_hackaprompt"].groupby(["model","source"]).flag_rate.mean().unstack().round(3)
print("HackAPrompt benchmark: recall on hackaprompt, FPR on alpaca"); display(hp.loc[[m for m in ORDER if m in hp.index]])
""")
code("""
# which words drive NotInject false positives? (FP rate among NotInject prompts containing the word, n>=8)
def fp_words(model, top=12, seed=0):
    g = P[(P.model == model) & (P.seed == seed) & (P.split == "bench_deepset_notinject") & P.source.str.startswith("notinject")]
    tok = lambda t: set(re.findall(r"[a-z']{4,}", t.lower()))
    cnt, fp = {}, {}
    for t, s in zip(g.text, g.score):
        for w in tok(t):
            cnt[w] = cnt.get(w, 0) + 1; fp[w] = fp.get(w, 0) + int(s >= .5)
    r = pd.DataFrame({"n": pd.Series(cnt), "fp": pd.Series(fp)}); r = r[r.n >= 8]; r["fp_rate"] = (r.fp / r.n).round(2)
    return r.sort_values(["fp_rate","n"], ascending=False).head(top)
for m in ["ProtectAI","Qwen3B-QLoRA"]:
    print("\\n", m); display(fp_words(m))
""")
md("## 5. ROC / PR curves (seed 0)")
code("""
PLOT = ["Regex","TF-IDF+LR","PromptGuard","PromptGuard-JB","ProtectAI","LlamaGuard-FS","Qwen1.5B-QLoRA","Qwen3B-QLoRA","Qwen3B-QLoRA+HN"]
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
for j, b in enumerate(["bench_hackaprompt","bench_deepset_notinject"]):
    for m in PLOT:
        g = P[(P.model == m) & (P.seed == 0) & (P.split == b)]
        if g.empty: continue
        f, t, _ = roc_curve(g.label, g.score); ax[0, j].plot(f, t, label=m)
        pr, rc, _ = precision_recall_curve(g.label, g.score); ax[1, j].plot(rc, pr, label=m)
    ax[0, j].plot([0,1],[0,1],"k:"); ax[0, j].set_title("ROC - " + BENCH[b]); ax[0, j].set_xlabel("FPR"); ax[0, j].set_ylabel("TPR")
    ax[1, j].set_title("PR - " + BENCH[b]); ax[1, j].set_xlabel("Recall"); ax[1, j].set_ylabel("Precision")
ax[0, 0].legend(fontsize=7); plt.tight_layout(); plt.savefig(C.RESULTS_DIR / "roc_pr.png"); plt.show()
""")
md("## 6. Over-defense and confusion matrices")
code("""
fig, ax = plt.subplots(figsize=(9, 4.5))
ni = P[(P.split == "notinject_heldout")].assign(flag=lambda x: x.score >= .5).groupby(["model","seed"]).flag.mean().groupby("model").mean()
ni = ni.loc[[m for m in ORDER if m in ni.index]]
ax.barh(ni.index[::-1], ni.values[::-1]); ax.set_xlabel("FPR on held-out NotInject (benign only)"); plt.tight_layout()
plt.savefig(C.RESULTS_DIR / "notinject_fpr.png"); plt.show()
sel = ["ProtectAI","PromptGuard-JB","LlamaGuard-FS","Qwen3B-QLoRA","Qwen3B-QLoRA+HN"]
fig, ax = plt.subplots(1, len(sel), figsize=(3.2 * len(sel), 3.2))
for a, m in zip(ax, sel):
    g = P[(P.model == m) & (P.seed == 0) & (P.split == "bench_deepset_notinject")]
    cm = confusion_matrix(g.label, g.score >= .5, labels=[0, 1]); a.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm): a.text(j, i, v, ha="center", va="center")
    a.set_title(m, fontsize=8); a.set_xticks([0,1]); a.set_yticks([0,1]); a.set_xlabel("pred"); a.set_ylabel("true"); a.grid(False)
plt.tight_layout(); plt.savefig(C.RESULTS_DIR / "confusion.png"); plt.show()
""")
md("## 7. Latency and resources (common protocol: 128 prompts, warm-up, CUDA sync; batch 1 and 32)")
code("""
lt = L.groupby("model").mean(numeric_only=True).loc[[m for m in ORDER if m in set(L.model)]]
lt["params (M)"] = (lt.n_params / 1e6).round(1)
display(lt[["params (M)","peak_vram_gb","latency_ms_bs1","latency_ms_bs32"]].round(2))
print("Note: n_params for 4-bit QLoRA models counts packed 4-bit weights, so it under-reports the real parameter count (~1.5B/3B).")
f1 = d[d.benchmark == "bench_deepset_notinject"].groupby("model")[["F1","FPR"]].mean()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, a in zip(["F1","FPR"], ax):
    for m in lt.index:
        a.scatter(lt.loc[m, "latency_ms_bs1"], f1.loc[m, k]); a.annotate(m, (lt.loc[m, "latency_ms_bs1"], f1.loc[m, k]), fontsize=7)
    a.set_xscale("log"); a.set_xlabel("latency ms/sample (batch 1)"); a.set_ylabel(k + " on Deepset+NotInject")
plt.tight_layout(); plt.savefig(C.RESULTS_DIR / "latency_scatter.png"); plt.show()
""")
md("## 8. Significance (exact McNemar on correctness, Deepset+NotInject, seed 0, threshold 0.5)")
code("""
def S(m, b="bench_deepset_notinject"):
    return P[(P.model == m) & (P.seed == 0) & (P.split == b)].set_index("text")
pairs = [("ProtectAI","Qwen3B-QLoRA"),("Qwen1.5B-QLoRA","Qwen3B-QLoRA"),("Qwen3B-QLoRA","Qwen3B-QLoRA+HN"),
         ("PromptGuard-JB","LlamaGuard-FS"),("TF-IDF+LR","Qwen1.5B-QLoRA"),("ProtectAI","PromptGuard-JB")]
rows = []
for a, b in pairs:
    A, B = S(a), S(b); idx = A.index.intersection(B.index).unique()
    A, B = A.loc[idx].groupby(level=0).first(), B.loc[idx].groupby(level=0).first()
    rows.append({"A": a, "B": b, "n": len(idx), "acc_A": ((A.score >= .5) == A.label).mean().round(3), "acc_B": ((B.score >= .5) == B.label).mean().round(3),
                 "p-value": round(mcnemar_p(A.label, A.score, B.score), 4)})
display(pd.DataFrame(rows))
""")
md("## 9. Qualitative error analysis (Qwen3B-QLoRA, seed 0, Deepset+NotInject)")
code("""
g = P[(P.model == "Qwen3B-QLoRA") & (P.seed == 0) & (P.split == "bench_deepset_notinject")].copy()
g["pred"] = (g.score >= .5).astype(int)
g["lang"] = np.where(g.text.str.contains(r"\\b(und|der|die|das|ist|nicht|ich|wie|was|Sie)\\b", case=False, regex=True), "de", "en")
g["len"] = g.text.str.split().str.len()
print(g.groupby("lang").apply(lambda x: pd.Series({"n": len(x), "recall": x[x.label==1].pred.mean(), "FPR": x[x.label==0].pred.mean()})).round(3))
print("\\nFalse negatives (missed attacks):"); display(g[(g.label==1)&(g.pred==0)][["text","score","lang"]].head(8))
print("False positives:"); display(g[(g.label==0)&(g.pred==1)][["text","source","score"]].head(8))
print("Mean words: FN %.0f | TP %.0f | FP %.0f | TN %.0f" % tuple(g[(g.label==l)&(g.pred==p)]["len"].mean() for l, p in [(1,0),(1,1),(0,1),(0,0)]))
""")
md("## 10. QLoRA ablation (Qwen2.5-1.5B, seed 0) and hard-negative ablation")
code("""
ab = C.RESULTS_DIR / "ablation.csv"
if ab.exists(): display(pd.read_csv(ab).round(3))
h = d[d.model.str.contains("TF-IDF|QLoRA|DeBERTa-FT")].copy()
h["family"] = h.model.str.replace("+HN", "", regex=False); h["hard_negatives"] = h.model.str.endswith("+HN")
display(h[h.benchmark.isin(["bench_deepset_notinject","notinject_heldout"])].groupby(["benchmark","family","hard_negatives"])[["Recall","F1","FPR"]].mean().round(3))
""")
md("""
The table above scores `+HN` models on fewer benign rows (their NotInject training/validation rows are removed), so F1 and precision are
not comparable across the two variants. Below, both variants are scored on the **same** Deepset+NotInject rows (default threshold, mean over seeds).
""")
code("""
HP = hn_paired_table(P).groupby(["family","hard_negatives"])[["n","Precision","Recall","F1","FPR","AUROC"]].mean().round(3)
display(HP); HP.to_csv(C.RESULTS_DIR / "table_hn_paired.csv")
""")
md("""
## 11. Published reference points (not directly comparable - different data/protocols)
| Source | Finding |
|---|---|
| [PromptShield (arXiv 2501.15145)](https://arxiv.org/abs/2501.15145) | Their detector reaches ~65% TPR at 0.1% FPR; they report Meta's Prompt-Guard drops to ~12.8% TPR when FPR is held to <=1%, so low-FPR operating points matter more than accuracy. |
| [InjecGuard / NotInject (arXiv 2410.22770)](https://arxiv.org/abs/2410.22770) | Open-source guards incl. Deepset, Fmops, Prompt-Guard and ProtectAI-v2 have NotInject (over-defense) accuracy below 60% (50% = chance). |
| This work | ProtectAI-v2 FPR 0.45 on held-out NotInject (accuracy ~55%) and Prompt-Guard 1.00 (accuracy ~0%): consistent with the InjecGuard finding. Our TPR@5%FPR is reported in section 1; TPR@1%FPR is in `metrics.csv` but is noisy given ~110 validation / ~340-395 benign test samples. |
""")
md("## 12. LaTeX / CSV export")
code("""
out = []
for b, t in TABLES.items():
    t.to_csv(C.RESULTS_DIR / f"table_{b}.csv")
    out.append("% " + BENCH[b] + "\\n" + t.to_latex(escape=True, column_format="l" + "r" * t.shape[1]))
(C.RESULTS_DIR / "tables.tex").write_text("\\n\\n".join(out), encoding="utf8")
CI.to_csv(C.RESULTS_DIR / "table_ci.csv", index=False); print("written:", sorted(p.name for p in C.RESULTS_DIR.glob("table*")))
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "analysis.ipynb")
