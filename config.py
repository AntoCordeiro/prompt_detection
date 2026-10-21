"""Central configuration: paths, seeds, model ids, hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PRED_DIR = RESULTS_DIR / "predictions"
CKPT_DIR = ROOT / "checkpoints"
for _d in (DATA_DIR, PRED_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]
DATA_SEED = 42          # splits are fixed across model seeds

# ---- data ----
N_HACK_TRAIN = 400      # HackAPrompt attacks in the train pool
N_HACK_TEST = 339       # HackAPrompt attacks in benchmark 1 (== NotInject size)
N_ALPACA_TRAIN = 200    # generic benign prompts in the train pool
N_ALPACA_TEST = 339     # generic benign prompts in benchmark 1
VAL_FRAC = 0.20
HARD_NEG_FRAC = 0.40    # share of NotInject moved to train in the D1 ablation
NEAR_DUP_THRESH = 0.90  # char-tfidf cosine above which a train row is dropped

# ---- models ----
PROMPT_GUARD = "meta-llama/Prompt-Guard-86M"
PROTECTAI = "protectai/deberta-v3-base-prompt-injection-v2"
LLAMA_GUARD = "meta-llama/Llama-Guard-3-1B"
QWEN_1_5B = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_3B = "Qwen/Qwen2.5-3B-Instruct"

# ---- QLoRA ----
QLORA = dict(
    r=16, alpha=32, dropout=0.05, lr=2e-4, epochs=3,
    max_length=256,  # overwritten by token-length check in qlora.py if smaller/larger needed
    targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    batch={"1.5b": (8, 2), "3b": (4, 4)},  # (per-device batch, grad accumulation)
)

# ---- evaluation ----
BOOTSTRAP_N = 1000
LATENCY_N = 128         # samples used for latency measurement
LATENCY_BATCHES = (1, 32)
FPR_TARGET = 0.05       # operating point for validation-calibrated threshold
