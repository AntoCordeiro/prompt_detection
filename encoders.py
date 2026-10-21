"""Methodology 1: Prompt-Guard-86M (two score variants) and ProtectAI DeBERTa-v3.

Long prompts are split into overlapping 512-token windows and the max score over windows is used
(model-card recommendation) instead of silently truncating (PLAN D8).
"""
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config as C
from common import Detector, count_params, timed_load


class Encoder(Detector):
    """variant: 'combined' -> 1 - P(benign)  |  'jailbreak' -> P(JAILBREAK) only  |  'injection' -> P(INJECTION)"""

    def __init__(self, model_id, name, variant="combined", **_):
        super().__init__()
        self.name = name
        self.variant = variant
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model, t = timed_load(lambda: AutoModelForSequenceClassification.from_pretrained(model_id).cuda().eval())
        self.id2label = {i: l.upper() for i, l in self.model.config.id2label.items()}
        self.benign_ids = [i for i, l in self.id2label.items() if l in ("BENIGN", "SAFE", "LABEL_0")]
        self.meta = {"n_params": count_params(self.model), "load_time_s": t}
        torch.cuda.reset_peak_memory_stats()

    def _probs(self, texts):
        enc = self.tok(list(map(str, texts)), truncation=True, max_length=512, stride=64,
                       return_overflowing_tokens=True, padding=True, return_tensors="pt")
        owner = enc.pop("overflow_to_sample_mapping").numpy()
        enc.pop("num_truncated_tokens", None)
        enc = {k: v.cuda() for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        with torch.no_grad():
            p = torch.softmax(self.model(**enc).logits.double(), dim=-1).cpu().numpy()
        return p, owner

    def _chunk_score(self, p):
        if self.variant == "combined":
            # sum of non-benign classes (avoids 1 - p_benign saturating to exactly 0/1)
            return p[:, [i for i in self.id2label if i not in self.benign_ids]].sum(axis=1)
        want = {"jailbreak": "JAILBREAK", "injection": "INJECTION"}[self.variant]
        return p[:, [i for i, l in self.id2label.items() if l == want]].sum(axis=1)

    def score(self, texts, bs=32):
        texts = list(texts)
        out = np.zeros(len(texts))
        for i in range(0, len(texts), bs):
            p, owner = self._probs(texts[i:i + bs])
            s = self._chunk_score(p)
            for j in range(len(texts[i:i + bs])):
                out[i + j] = s[owner == j].max()  # max over windows
        self.meta["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        return out

    def raw_dump(self, texts):
        """Print raw class probabilities (sanity check of the label mapping)."""
        p, _ = self._probs(texts)
        print(self.id2label)
        for t, row in zip(texts, p):
            print(np.round(row, 3), "|", str(t)[:80].replace("\n", " "))

    def close(self):
        del self.model
        super().close()


def build(key, **_):
    if key == "PromptGuard":
        return Encoder(C.PROMPT_GUARD, "PromptGuard", "combined")
    if key == "PromptGuard-JB":
        return Encoder(C.PROMPT_GUARD, "PromptGuard-JB", "jailbreak")
    if key == "ProtectAI":
        return Encoder(C.PROTECTAI, "ProtectAI", "combined")
    raise KeyError(key)
