"""Methodology 2: Qwen2.5-{1.5B,3B}-Instruct + QLoRA with a sequence-classification head.

Keys: "Qwen1.5B-QLoRA", "Qwen3B-QLoRA"; suffix "+HN" trains on `train_hn` (adds 40% of NotInject as hard
negatives) and is the D1 ablation - evaluated only on `notinject_heldout` for a fair comparison.
"""
import time

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

import config as C
from common import Detector, count_params, free_gpu, set_seed

MODEL_IDS = {"Qwen1.5B": C.QWEN_1_5B, "Qwen3B": C.QWEN_3B}


def _pick_max_length(tok, texts, cap=512):
    lens = np.array([len(x) for x in tok(list(texts), add_special_tokens=True)["input_ids"]])
    p95 = int(np.percentile(lens, 95))
    ml = int(min(cap, max(128, int(np.ceil(p95 / 64) * 64))))
    print(f"[qlora] token length p50={int(np.median(lens))} p95={p95} max={lens.max()} -> max_length={ml}")
    return ml


class WeightedTrainer(Trainer):
    def __init__(self, *a, class_weights=None, **k):
        super().__init__(*a, **k)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        out = model(**inputs)
        w = self.class_weights.to(out.logits.device, out.logits.dtype)
        loss = torch.nn.functional.cross_entropy(out.logits.float(), labels, weight=w.float())
        return (loss, out) if return_outputs else loss


class QLoraDetector(Detector):
    def __init__(self, key, model, tok, max_length, meta):
        super().__init__()
        self.name, self.model, self.tok, self.max_length, self.meta = key, model.eval(), tok, max_length, meta

    def score(self, texts, bs=16):
        texts = list(map(str, texts))
        order = np.argsort([len(t) for t in texts])  # length-sort to minimise padding
        out = np.zeros(len(texts))
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            enc = self.tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_length).to("cuda")
            with torch.no_grad():
                out[idx] = torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, 1].cpu().numpy()
        self.meta["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        return out

    def close(self):
        del self.model
        super().close()


def build(key, seed=0, splits=None, overrides=None, tag="", **_):
    hn = key.endswith("+HN")
    size = key.replace("-QLoRA", "").replace("+HN", "")
    model_id = MODEL_IDS[size]
    cfg = {**C.QLORA, **(overrides or {})}
    train_df = splits["train_hn" if hn else "train"]
    val_df = splits["validation_hn" if hn else "validation"]
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    max_length = cfg.get("max_length_override") or _pick_max_length(tok, train_df["prompt_text"].astype(str))

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    t0 = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0})
    model.config.pad_token_id = tok.pad_token_id
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=(size == "Qwen3B"), gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=cfg["r"], lora_alpha=cfg["alpha"],
                                             lora_dropout=cfg["dropout"], target_modules=cfg["targets"]))
    model.print_trainable_parameters()

    def tokenize(b):
        return tok(b["prompt_text"], truncation=True, max_length=max_length)

    def to_ds(df):
        return Dataset.from_pandas(df[["prompt_text", "label"]].astype({"prompt_text": str})).map(
            tokenize, batched=True, remove_columns=["prompt_text"]).rename_column("label", "labels")

    counts = np.bincount(train_df["label"], minlength=2)
    weights = torch.tensor(len(train_df) / (2 * counts), dtype=torch.float32)

    bs, ga = cfg["batch"]["3b" if size == "Qwen3B" else "1.5b"]
    out_dir = C.CKPT_DIR / f"{key}_s{seed}{tag}"
    args = TrainingArguments(
        output_dir=str(out_dir), per_device_train_batch_size=bs, per_device_eval_batch_size=16,
        gradient_accumulation_steps=ga, learning_rate=cfg["lr"], num_train_epochs=cfg["epochs"],
        bf16=True, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        logging_steps=25, report_to="none", seed=seed, data_seed=seed, dataloader_num_workers=0,
        lr_scheduler_type="cosine", warmup_steps=0.05, remove_unused_columns=True,
    )

    def compute_metrics(p):
        return {"f1": f1_score(p.label_ids, p.predictions.argmax(-1))}

    trainer = WeightedTrainer(model=model, args=args, train_dataset=to_ds(train_df), eval_dataset=to_ds(val_df),
                              processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                              compute_metrics=compute_metrics, class_weights=weights)
    trainer.train()
    train_time = time.perf_counter() - t0
    model.save_pretrained(out_dir / "adapter")
    tok.save_pretrained(out_dir / "adapter")
    del trainer
    free_gpu()

    meta = {"n_params": count_params(model), "load_time_s": train_time, "train_time_s": train_time,
            "max_length": max_length, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    return QLoraDetector(key, model, tok, max_length, meta)
