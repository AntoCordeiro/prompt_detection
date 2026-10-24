# Detecting Direct Prompt Injections

Comparative study of lightweight binary detectors (benign vs. prompt injection / jailbreak) that weighs **detection quality**, **over-defense** (false positives on benign prompts that contain trigger words) and **latency**. Everything runs locally on a single 16 GB GPU (developed on an RTX 4070 Ti Super, Windows 11).

| Group | Systems |
|---|---|
| Baselines | keyword/regex heuristic; TF-IDF n-gram + logistic regression |
| Methodology 1 (encoders) | `meta-llama/Prompt-Guard-86M` (two score variants), `protectai/deberta-v3-base-prompt-injection-v2`, and our own fully fine-tuned `microsoft/deberta-v3-base` |
| Methodology 2 (small LLMs) | `meta-llama/Llama-Guard-3-1B` zero/few-shot (custom prompt-injection category); `Qwen2.5-1.5B/3B-Instruct` zero/few-shot prompting and QLoRA fine-tuning (4-bit NF4 + sequence-classification head) |

Data: [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections), [HackAPrompt](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset) (attacks only), [NotInject](https://huggingface.co/datasets/leolee99/NotInject) (benign, trigger-word-rich), plus Alpaca-cleaned as generic benign text.
Metrics: accuracy, precision, recall, F1, FPR on benign prompts, AUROC, TPR@FPR, bootstrap CIs, McNemar tests, latency (batch 1 / 32), VRAM.

**Results and discussion: [REPORT.md](REPORT.md)** (Part 2 tests whether the benchmarks are fair: level split, out-of-distribution training, style probe, trigger-word hard negatives) · full tables/plots: [analysis.ipynb](analysis.ipynb) · `results/` (metrics, tables, figures).

## Setup
```bash
python -m venv .venv                       # Python 3.11 or 3.12
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m bitsandbytes       # should print SUCCESS (needed for QLoRA)
.venv/Scripts/hf auth login                # Prompt-Guard, Llama-Guard and HackAPrompt are gated: accept the terms on the Hub first
```
Tokens are stored by the Hugging Face CLI; nothing secret is in the repo.

## Reproduce
```bash
python data.py                                    # build splits (+ leakage checks) -> data/
python data.py --extra                            # v2 splits (level split, OOD, trigger-word hard negatives)
python run_all.py --models everything --seeds 0 1 2   # all models + v2 protocols; resumable; predictions -> results/predictions/, summary -> results/metrics.csv
python probe.py                                   # style/source probe -> results/style_probe.csv
python ablate.py                                  # QLoRA hyper-parameter ablation -> results/ablation.csv
python make_notebook.py && jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```
A full run takes roughly 1-1.5 h on the reference GPU (QLoRA 1.5B ~1.5 min, 3B ~7 min per seed). Run GPU jobs one at a time so latency numbers stay clean.

`data/`, `checkpoints/` and `results/predictions/` are not committed (dataset licences / size); `analysis.ipynb` needs `results/predictions/`, so run `run_all.py` once before re-executing it. The committed notebook already contains its outputs.

## Layout
| File | Purpose |
|---|---|
| `config.py`, `common.py` | paths, seeds, model ids; shared `Detector` interface (every model returns P(malicious)) |
| `data.py` | builds the benchmarks, hard-negative splits and leakage checks |
| `metrics.py` | metrics, bootstrap CIs, threshold selection, McNemar, latency protocol |
| `baselines.py`, `encoders.py`, `llama_guard.py`, `qwen_prompt.py`, `qlora.py`, `finetune_encoder.py` | the models |
| `run_all.py`, `ablate.py` | CLIs |
| `make_notebook.py` → `analysis.ipynb` | reporting notebook |
| `PLAN.md` | the improvement plan this implementation follows; `NLP_HWp.ipynb` is the original Colab notebook (archived) |

## Notes
- The hard-negative (`+HN`) rows train on 40% of NotInject and are evaluated with those rows removed; they are an ablation, not a like-for-like comparison with the zero-shot detectors.
- transformers 5.x: load trainable models with `dtype=torch.float32`; DeBERTa's tokenizer needs `protobuf`.
- Dataset and model licences apply to the data/weights you download; this repository contains only code and aggregate results.
