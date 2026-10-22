# Detecting Direct Prompt Injections - results summary

All numbers come from `results/metrics.csv` and `analysis.ipynb` (default threshold 0.5 unless stated; Qwen rows are the mean over 3 seeds).
Hardware: RTX 4070 Ti Super 16 GB, everything run locally.

## Setup in one paragraph
Binary detectors (benign = 0, malicious = 1) scored as P(malicious). Two benchmarks follow the spec's dataset definition:
**HackAPrompt** (339 held-out HackAPrompt attacks vs 339 Alpaca benign prompts; HackAPrompt has no benign data) and
**Deepset + NotInject** (Deepset test: 60 attacks / 56 benign, plus all 339 NotInject benign prompts that contain trigger words).
A third, benign-only set (**NotInject held-out**, 204 prompts) isolates over-defense. Training data: Deepset train + 400 HackAPrompt attacks + 200 Alpaca benign
(near-duplicates of any test row removed: 29 rows). Models: regex and TF-IDF+LogReg baselines; Prompt-Guard-86M and ProtectAI DeBERTa-v3 (zero-shot);
Llama-Guard-3-1B (zero/few-shot, custom prompt-injection category, `sigmoid(logit_unsafe - logit_safe)` score); Qwen2.5-1.5B/3B-Instruct with QLoRA
(4-bit NF4, LoRA r=16 on all linear layers, classification head, class-weighted loss, 3 seeds).

## Main results

| Model | HackAPrompt F1 / FPR | Deepset+NotInject Recall / FPR / F1 | AUROC (D+N) | NotInject held-out FPR | ms/sample (bs 1 / 32) |
|---|---|---|---|---|---|
| Regex | 0.18 / 0.00 | 0.23 / 0.07 / 0.28 | 0.59 | 0.11 | 0.02 / 0.02 |
| TF-IDF + LR | 0.95 / 0.04 | 0.58 / 0.13 / 0.48 | 0.78 | 0.13 | 0.4 / 0.03 |
| Prompt-Guard (1-P(benign)) | 0.66 / 1.00 | 0.95 / 0.97 / 0.23 | 0.30 | 1.00 | 24 / 2.6 |
| Prompt-Guard (JAILBREAK class) | 0.99 / 0.003 | 0.25 / 0.05 / 0.32 | 0.69 | 0.06 | 15 / 2.1 |
| ProtectAI DeBERTa-v3 | 1.00 / 0.003 | 0.37 / 0.37 / 0.19 | 0.53 | 0.45 | 15 / 2.4 |
| Llama-Guard-3-1B zero-shot | 0.43 / 0.015 | 0.17 / 0.05 / 0.22 | 0.68 | 0.06 | 21 / 9.6 |
| Llama-Guard-3-1B few-shot | 0.56 / 0.015 | 0.17 / 0.06 / 0.21 | 0.69 | 0.07 | 21 / 12.7 |
| Qwen2.5-1.5B zero-shot (prompted) | 0.91 / 0.16 | 0.83 / 0.39 / 0.38 | 0.80 | 0.39 | 34 / 15 |
| Qwen2.5-1.5B few-shot (prompted) | 0.95 / 0.06 | 0.85 / 0.28 / 0.46 | 0.86 | 0.30 | 48 / 27 |
| Qwen2.5-3B zero-shot (prompted) | 0.72 / 0.00 | 0.38 / 0.03 / 0.50 | 0.89 | 0.03 | 48 / 25 |
| Qwen2.5-3B few-shot (prompted) | 0.93 / 0.01 | 0.52 / 0.15 / 0.41 | 0.80 | 0.18 | 66 / 41 |
| DeBERTa-v3-base fully fine-tuned (ours) | 0.99 / 0.03 | 0.88 / 0.29 / 0.47 | 0.87 | 0.34 | 20 / 1.9 |
| Qwen2.5-1.5B QLoRA | 0.99 / 0.017 | 0.92 / 0.17 / 0.62 | 0.95 | 0.18 | 81 / 9.0 |
| Qwen2.5-3B QLoRA | 0.99 / 0.008 | 0.93 / 0.15 / 0.64 | 0.95 | 0.18 | 105 / 13 |
| *Ablation:* Qwen2.5-1.5B QLoRA + NotInject hard negatives | 0.99 / 0.014 | 0.92 / 0.009 / 0.94 | 0.98 | 0.007 | 80 / 8.5 |
| *Ablation:* Qwen2.5-3B QLoRA + NotInject hard negatives | 0.99 / 0.009 | 0.92 / 0.009 / 0.94 | 0.98 | 0.007 | 108 / 13 |
| *Ablation:* DeBERTa fine-tuned + NotInject hard negatives | 0.99 / 0.02 | 0.87 / 0.006 / 0.92 | 0.98 | 0.007 | 16 / 1.9 |
| *Ablation:* TF-IDF + LR + hard negatives | 0.95 / 0.03 | 0.55 / 0.02 / 0.67 | 0.92 | 0.03 | 0.4 / 0.03 |

The `+HN` rows are evaluated with the NotInject prompts they saw in training or validation removed, so on Deepset+NotInject they face 260 instead of 395
benign prompts; their F1/precision are therefore **not** directly comparable with the rows above (recall, FPR and AUROC are). The like-for-like comparison is below.

**Hard-negative ablation on a common row set** (Deepset+NotInject rows unseen by every variant: 60 attacks / 260 benign; default threshold; mean over seeds; `results/table_hn_paired.csv`):

| Model | Precision | Recall | F1 | FPR | AUROC |
|---|---|---|---|---|---|
| TF-IDF + LR / + HN | 0.57 / 0.85 | 0.58 / 0.55 | 0.58 / 0.67 | 0.100 / 0.023 | 0.80 / 0.92 |
| DeBERTa fine-tuned / + HN | 0.44 / 0.97 | 0.88 / 0.87 | 0.58 / 0.92 | 0.264 / 0.006 | 0.88 / 0.98 |
| Qwen2.5-1.5B QLoRA / + HN | 0.60 / 0.96 | 0.92 / 0.92 | 0.73 / 0.94 | 0.145 / 0.009 | 0.96 / 0.98 |
| Qwen2.5-3B QLoRA / + HN | 0.61 / 0.96 | 0.93 / 0.92 | 0.73 / 0.94 | 0.142 / 0.009 | 0.95 / 0.98 |

Bootstrap 95% CIs, TPR@1%/5% FPR, validation-calibrated thresholds, per-source flag rates, ROC/PR curves, confusion matrices and McNemar tests are in `analysis.ipynb`.

## Takeaways
1. **Detection quality and over-defense are different axes.** On the HackAPrompt benchmark nearly every trained model is excellent (F1 >= 0.95), but this benchmark is easy
   (attack and benign come from very different sources, so some of the score is source style). The Deepset+NotInject benchmark separates models: ProtectAI, near-perfect on
   HackAPrompt, flags 45% of held-out NotInject prompts (accuracy ~55%), and Prompt-Guard's INJECTION class flags essentially all of them. This agrees with the over-defense
   findings of the NotInject paper (InjecGuard) for open-source guards.
2. **Prompt-Guard must be read through its JAILBREAK class.** The `1 - P(benign)` mapping gives AUROC 0.30 on Deepset+NotInject and FPR 1.0; JAILBREAK-only gives FPR 0.003-0.05 but misses most Deepset-style attacks (recall 0.25).
3. **Llama-Guard-3-1B is a weak prompt-injection detector.** After fixing the prompt (the notebook's version produced constant output), zero-shot AUROC is 0.68-0.88 with recall <= 0.4:
   low over-defense (FPR ~5%) but it misses most attacks. A few in-prompt examples help on HackAPrompt (recall 0.27 -> 0.39) but not on Deepset.
4. **QLoRA-tuned Qwen gives the best detection** (AUROC 0.95 on Deepset+NotInject, TPR@5%FPR ~0.83-0.86 vs <= 0.33 for zero-shot detectors) but, trained on Deepset/HackAPrompt/Alpaca only,
   it over-defends on NotInject (FPR 0.15-0.18, worst for the 3-trigger-word prompts, 0.34-0.38). 3B vs 1.5B differences are within seed noise on most metrics.
5. **Hard negatives fix over-defense without hurting recall.** Adding 40% of NotInject to training cuts held-out NotInject FPR from 0.18 to 0.007 and, scored on the same Deepset+NotInject rows,
   raises QLoRA F1 from 0.73 to 0.94 (FPR 0.14 -> 0.009) with recall unchanged (~0.92). Caveat: the zero-shot encoders never saw NotInject, so the `+HN` rows are an ablation, not a like-for-like comparison.
6. **Latency/cost.** Encoders: ~15-25 ms/sample at batch 1 and ~2.5 ms at batch 32 on ~0.2-0.3B params. Llama-Guard-1B: ~21 ms (bs 1). QLoRA Qwen (4-bit): 80-105 ms (bs 1), 9-13 ms (bs 32); the 4-bit
   kernels make small batches slow. TF-IDF is >30x faster than any neural model but has AUROC 0.78 on Deepset+NotInject.
7. **Calibrating the threshold on validation does not fix everything.** For ProtectAI the validation-optimal threshold (~0) raises recall to 0.70 but FPR to 0.65 because validation has no NotInject-style negatives; for Qwen it moves FPR from 0.165 to 0.11 at a small F1 gain.
8. **Fine-tuning vs prompting (B2).** Zero/few-shot Qwen is far better than Llama-Guard as a detector (3B zero-shot AUROC 0.89 and FPR 0.03 on Deepset+NotInject) but is inconsistent:
   3B zero-shot is conservative (recall 0.38), few-shot helps on HackAPrompt (recall 0.56 -> 0.87) but not on Deepset+NotInject (3B few-shot AUROC 0.80), and the 1.5B model over-flags (FPR 0.28-0.39). QLoRA lifts recall to 0.93 at the same model size (AUROC 0.95), so task-specific training is worth it; few-shot results are single-run and prompt-dependent.
9. **Full fine-tuning of an encoder is the best cost/quality trade-off (B5).** Fine-tuned DeBERTa-v3-base (184M) reaches Qwen-QLoRA-level quality only when hard negatives are included
   (Deepset+NotInject F1 0.92 vs 0.94, FPR 0.006-0.009 for both) but runs 5x faster at batch 1 (16-20 ms vs 104-106 ms), ~7x at batch 32 (1.9 vs 13 ms) and trains in ~1.5 min vs 7-8 min. Without hard negatives it over-defends more than QLoRA Qwen (held-out NotInject FPR 0.34 vs 0.18).

## QLoRA ablation (Qwen2.5-1.5B, seed 0, headline training data; `results/ablation.csv`)
| Config | Val F1 | HackAPrompt F1 | Deepset+NotInject F1 / FPR | NotInject held-out FPR | Train time |
|---|---|---|---|---|---|
| base (r16, lr 2e-4, 3 ep, all-linear, len 128) | 0.996 | 0.993 | 0.66 / 0.12 | 0.13 | 98 s |
| r = 8 | 0.991 | 0.988 | 0.58 / 0.20 | 0.22 | 95 s |
| r = 32 | 0.991 | 0.988 | 0.53 / 0.24 | 0.27 | 94 s |
| lr 1e-4 | 0.987 | 0.984 | 0.56 / 0.22 | 0.24 | 99 s |
| lr 5e-4 | 0.991 | 0.990 | 0.56 / 0.19 | 0.21 | 110 s |
| 2 epochs | 0.991 | 0.988 | 0.70 / 0.10 | 0.10 | 76 s |
| 5 epochs | 0.996 | 0.991 | 0.65 / 0.14 | 0.15 | 183 s |
| attention-only LoRA | 0.991 | 0.991 | 0.60 / 0.18 | 0.20 | 79 s |
| max_length 256 | 0.996 | 0.993 | 0.59 / 0.19 | 0.20 | 110 s |

Validation F1 and the HackAPrompt benchmark are flat (0.987-0.996 / 0.984-0.993), i.e. the in-distribution task is saturated and hyper-parameters barely matter there.
Over-defense on NotInject varies (FPR 0.10-0.27) with no monotonic trend in rank, learning rate or epochs; given the 3-seed std of ~0.04-0.05 for the base config,
most of these differences are within seed noise (single seed per row here). Only the shorter training (2 epochs: lowest FPR) hints at less over-fitting to Deepset/HackAPrompt style.
Longer context (256) and all-linear vs attention-only LoRA give no benefit, so the default (r=16, all-linear, length from the 95th percentile = 128) is kept. The hard-negative data (takeaway 5) is a far larger effect than any hyper-parameter.

## Limitations / honest caveats
- Deepset test has only 60 attacks; CIs for recall/F1 on Deepset+NotInject are wide (see `analysis.ipynb`). The validation set has ~108 benign prompts, so FPR targets below ~5% are not estimable.
- The HackAPrompt benchmark confounds source and label (HackAPrompt vs Alpaca); per-source rates are reported instead of relying only on the mixed F1/AUROC.
- Llama-Guard uses our custom category and few-shot text, not Meta's default taxonomy; results depend on that wording. Few-shot examples were checked not to occur in any split.
- Prompts are passed to the tokenizers as plain text, so a literal special-token string inside a prompt is parsed as that token. This affects one
  test prompt (a HackAPrompt attack containing `<|im_end|>`) plus validation rows; the effect on any metric is at most one sample.
- `n_params` for 4-bit QLoRA models counts packed weights (under-reports real size); `peak_vram_gb` for QLoRA includes the training peak.
- Fine-tuned encoder row uses DeBERTa-v3-base only (not Prompt-Guard). The Qwen prompting results use one hand-written prompt and the same four few-shot examples as Llama-Guard; no prompt tuning was done.
- Published reference points (different data/protocols, not directly comparable): PromptShield ([arXiv 2501.15145](https://arxiv.org/abs/2501.15145)) reports ~65% TPR at 0.1% FPR for its detector and ~12.8% TPR for Prompt-Guard at <= 1% FPR; InjecGuard ([arXiv 2410.22770](https://arxiv.org/abs/2410.22770)) reports NotInject accuracy below 60% for Deepset, Fmops, Prompt-Guard and ProtectAI-v2.

## Reproduce
```
.venv/Scripts/python data.py
.venv/Scripts/python run_all.py --models all --seeds 0 1 2
.venv/Scripts/python ablate.py
.venv/Scripts/python make_notebook.py && jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```
