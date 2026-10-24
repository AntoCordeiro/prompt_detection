# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

University NLP project: *Detecting Direct Prompt Injections*. This is a comparative study of lightweight binary detectors (benign = 0, malicious = 1) that weighs detection quality against over-defense and latency. These are the systems required by the spec:

- **Baseline:** keyword/regex heuristic and a TF-IDF n-gram Logistic Regression.
- **Methodology 1 (encoders):** `meta-llama/Prompt-Guard-86M` and `protectai/deberta-v3-base-prompt-injection-v2`.
- **Methodology 2 (small LLMs):**
  - `meta-llama/Llama-Guard-3-1B`, zero-shot and few-shot.
  - `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen2.5-3B-Instruct` fine-tuned with QLoRA (4-bit NF4 with a sequence-classification head).
- **Datasets:**
  - `deepset/prompt-injections`.
  - HackAPrompt, using only rows with `correct==True` as attacks. It has no benign samples.
  - NotInject, which is all benign and contains trigger words. It is used to measure over-defense.
- **Metrics:** Accuracy, Precision, Recall, F1, FPR on benign prompts, AUROC, and latency.

## Current state

- Pipeline implemented as scripts (PLAN.md steps 0-3 done): `config.py`, `data.py`, `common.py`, `metrics.py`, `baselines.py`, `encoders.py`, `llama_guard.py`, `qlora.py`, `run_all.py`. `NLP_HWp.ipynb` is the archived original Colab notebook.
- Also done: `ablate.py` (QLoRA ablation -> results/ablation.csv), `make_notebook.py` (generates `analysis.ipynb`; execute with nbconvert), `REPORT.md` (results summary + caveats).
- Also done: `qwen_prompt.py` (Qwen zero/few-shot, keys Qwen1.5B/3B-ZS/FS) and `finetune_encoder.py` (DeBERTa-FT, DeBERTa-FT+HN). All PLAN.md items are complete.
- No git repo or tests. Env: `.venv` (Python 3.11, torch 2.6+cu124, transformers 5.x, bitsandbytes 0.50).

## Commands

```
.venv/Scripts/python data.py                                   # rebuild data/*.jsonl (+ leakage checks)
.venv/Scripts/python run_all.py --models all --seeds 0 1 2     # resumable; skips existing results/predictions/*.parquet
.venv/Scripts/python run_all.py --models LlamaGuard-ZS --debug # print rendered prompt / generations / raw labels
```
Model keys (plus `@lvl|@noHack|@hackOnly|@hnWord` variants, see v2 protocols): Regex, TF-IDF+LR, PromptGuard, PromptGuard-JB, ProtectAI, LlamaGuard-ZS/-FS, Qwen1.5B/3B-ZS/-FS, DeBERTa-FT, Qwen1.5B-QLoRA, Qwen3B-QLoRA, each with a `+HN` ablation variant for TF-IDF, DeBERTa-FT and QLoRA (trained with 40% of NotInject as hard negatives). To redo a run, delete its parquet. Run GPU jobs one at a time, otherwise latency numbers are contaminated. Outputs: `results/predictions/`, `results/latency/`, `results/metrics.csv` (default / val-F1 / val-FPR<=5% thresholds, bootstrap CIs).

Design: every model is a `Detector` (common.py) returning P(malicious) in [0,1]; run_all writes per-sample scores for all splits; metrics.py computes everything from those parquets. `+HN` models are evaluated with the NotInject rows used in training removed (metrics.build_metrics_table).

## v2 protocols (added after the first round; additive, old results are the "random-split" reference)
- `python data.py --extra` adds splits without touching the originals: `*_lvl` (HackAPrompt levels {2,5,8} held out), `*_noHack`, `*_hackOnly` (out-of-distribution), `*_hnWord` + `notinject_heldout_word` (trigger-word-disjoint hard negatives), `bench_hack_lvl`.
- Keys carry an `@protocol` suffix (e.g. `Qwen3B-QLoRA@lvl`, `DeBERTa-FT+HN@hnWord`); `data.split_names(key)` maps a key to its train/validation split. `run_all.py --models v2` trains them; `--models all` / `everything` also fills missing eval splits for existing models by reloading checkpoints (`qlora.load`, `finetune_encoder.load`, verified to reproduce stored scores exactly - scoring runs under explicit bf16 autocast, and QLoRA reload needs `prepare_model_for_kbit_training`).
- `probe.py` -> `results/style_probe.csv` (style/source separability of each benchmark). `metrics.exclude_trained` removes, per model, the NotInject rows it trained/validated on.

## Original notebook architecture (ported to the scripts above)

- **Data build (cell 6):** writes JSONL splits.
  - `train` / `val` come from Deepset train plus a 400-sample HackAPrompt pool.
  - `test_deepset`, `test_hackaprompt_notinject` (339 HackAPrompt attacks and 339 NotInject benign) and `test_notinject_only`.
  - NotInject rows keep a `source` column that records the 1/2/3 trigger-word split.
- **Eval harness (cell 8):** `evaluate_model_on_benchmark` and `run_full_evaluation`.
  - Every model is reduced to a function that returns `p_malicious` per prompt. The harness then computes the metrics and appends them to the global `ALL_RESULTS` and `ROC_CURVES`.
  - Re-running cells creates duplicate rows (bug A8).
- **Model cells:**
  - Baselines: cell 11.
  - Encoders, via HF `pipeline`: cell 14. Prompt-Guard is scored as `1 − P(BENIGN)`.
  - Llama-Guard: cell 18. It scores from `safe`/`unsafe` logits. The zero-shot output is currently degenerate (AUROC 0.5).
  - Qwen QLoRA: cell 20, using PEFT and the HF Trainer.
- **Export (cell 22):** writes a CSV, a LaTeX table and ROC plots.

## Environment

- Runs locally on Windows 11 with an RTX 4070 Ti Super (16 GB). The 3B QLoRA job fits in memory.
- Use a Python 3.11/3.12 venv with a CUDA build of torch and `bitsandbytes` ≥ 0.43 (check it with `python -m bitsandbytes`). If bitsandbytes fails on Windows, use WSL2.
- Prompt-Guard, Llama-Guard and HackAPrompt are gated. Authenticate with `hf auth login`, and never put tokens in code.
- Windows specifics:
  - Set `dataloader_num_workers=0`.
  - Guard script entry points with `if __name__ == "__main__":`.
  - Load only one LLM at a time, and free VRAM between models (`del`, `gc.collect()`, `torch.cuda.empty_cache()`).
- Prefer `bf16` autocast, but load trainable non-quantised models with `dtype=torch.float32` (transformers 5 defaults to the checkpoint dtype; DeBERTa fine-tuned in fp16 weights went NaN).
- DeBERTa-v3 tokenizer needs `protobuf` installed (otherwise transformers falls back to a tiktoken parser and fails).
- In `BitsAndBytesConfig`, use `bnb_4bit_use_double_quant`. `bnb_4bit_use_nested_quant` is not a valid argument.

## Gotchas

- **Llama-Guard-3-1B prompts:**
  - Its chat template expects `content` as a list, `[{"type": "text", "text": ...}]`.
  - The default taxonomy has no prompt-injection category. Use a custom category in the official Llama Guard 3 format.
  - Score with `logit_unsafe − logit_safe` in fp32.
- **Prompt-Guard over-firing:** its INJECTION class fires on most instruction-like text. Report the JAILBREAK-only and combined-class variants.
- **Confounded benchmark:** in `test_hackaprompt_notinject`, every positive comes from HackAPrompt and every negative from NotInject. Report NotInject FPR and HackAPrompt recall separately rather than relying only on mixed F1/AUROC.
