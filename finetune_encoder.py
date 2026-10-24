"""Full fine-tuning of microsoft/deberta-v3-base on our train split (PLAN B5) - a like-for-like encoder counterpart
to the QLoRA rows (same data, class-weighted loss, best epoch by validation F1).

Keys: "DeBERTa-FT" (headline training data) and "DeBERTa-FT+HN" (adds 40% of NotInject as hard negatives).
"""
import time

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments)

import config as C
from common import Detector, count_params, free_gpu, set_seed
from qlora import WeightedTrainer, _pick_max_length

MODEL_ID = "microsoft/deberta-v3-base"
HP = dict(lr=2e-5, epochs=4, batch=16)


class FTDetector(Detector):
    def __init__(self, key, model, tok, max_length, meta):
        super().__init__()
        self.name, self.model, self.tok, self.max_length, self.meta = key, model.eval(), tok, max_length, meta

    def score(self, texts, bs=32):
        texts = list(map(str, texts))
        order = np.argsort([len(t) for t in texts])
        out = np.zeros(len(texts))
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            enc = self.tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_length).to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # same precision as training (explicit, reproducible)
                out[idx] = torch.softmax(self.model(**enc).logits.double(), dim=-1)[:, 1].cpu().numpy()
        self.meta["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        return out

    def close(self):
        del self.model
        super().close()


def build(key, seed=0, splits=None, **_):
    from data import split_names
    tr_name, va_name = split_names(key)
    train_df, val_df = splits[tr_name], splits[va_name]
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    max_length = _pick_max_length(tok, train_df["prompt_text"].astype(str))
    t0 = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2, dtype=torch.float32).cuda()

    def to_ds(df):
        return Dataset.from_pandas(df[["prompt_text", "label"]].astype({"prompt_text": str})).map(
            lambda b: tok(b["prompt_text"], truncation=True, max_length=max_length),
            batched=True, remove_columns=["prompt_text"]).rename_column("label", "labels")

    counts = np.bincount(train_df["label"], minlength=2)
    weights = torch.tensor(len(train_df) / (2 * counts), dtype=torch.float32)
    out_dir = C.CKPT_DIR / f"{key}_s{seed}"
    args = TrainingArguments(
        output_dir=str(out_dir), per_device_train_batch_size=HP["batch"], per_device_eval_batch_size=32,
        learning_rate=HP["lr"], num_train_epochs=HP["epochs"], bf16=True, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        logging_steps=25, report_to="none", seed=seed, data_seed=seed, dataloader_num_workers=0,
        lr_scheduler_type="linear", warmup_steps=0.1, weight_decay=0.01)
    trainer = WeightedTrainer(model=model, args=args, train_dataset=to_ds(train_df), eval_dataset=to_ds(val_df),
                              processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                              compute_metrics=lambda p: {"f1": f1_score(p.label_ids, p.predictions.argmax(-1))},
                              class_weights=weights)
    trainer.train()
    train_time = time.perf_counter() - t0
    model.save_pretrained(out_dir / "final")
    tok.save_pretrained(out_dir / "final")
    del trainer
    free_gpu()
    meta = {"n_params": count_params(model), "load_time_s": train_time, "train_time_s": train_time,
            "max_length": max_length, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    return FTDetector(key, model, tok, max_length, meta)


def load(key, seed=0, **_):
    """Reload a fine-tuned model from checkpoints/ (used to score new splits without retraining)."""
    import json
    p = C.CKPT_DIR / f"{key}_s{seed}" / "final"
    max_length = json.loads((C.RESULTS_DIR / "latency" / f"{key}__s{seed}.json").read_text())["max_length"]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)  # same tokenizer object type as in training (a re-saved copy tokenises some rows differently)
    model = AutoModelForSequenceClassification.from_pretrained(p, dtype=torch.float32).cuda()
    torch.cuda.reset_peak_memory_stats()
    return FTDetector(key, model, tok, max_length, {"n_params": count_params(model)})
