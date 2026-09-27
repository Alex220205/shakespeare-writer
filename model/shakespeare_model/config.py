"""
Hyperparameters and file locations, as plain module constants.

WHY THIS EXISTS
    Every number that shapes the model or its training is here, so tuning a
    run is editing a number rather than reading the code.

WHAT THE BASELINE DID
    The same thing, at the top of the one file: batch_size, block_size,
    n_embd and the rest as globals, plus the corpus path hardcoded to a
    folder on one particular machine.

WHAT CHANGED AND WHY
    The constants moved into their own module because four modules read them
    now. The paths are built from this file's location, so the project runs
    from any folder on any machine.
"""

from pathlib import Path

# This file -> shakespeare_model -> model
#                     [0]           [1]
MODEL_DIR = Path(__file__).resolve().parents[1]

# Tiny Shakespeare, from karpathy/char-rnn: 1,115,394 bytes of plays.
DATA_PATH = MODEL_DIR / "data" / "input.txt"

# Written by train.py, read by the web service. Committed, so the demo works
# without training first.
CHECKPOINT_PATH = MODEL_DIR / "checkpoints" / "shakespeare.pt"

SEED = 1337

# --- Tokenizer ----------------------------------------------------------------
# 256 byte tokens plus 768 learned merges. Bigger vocabularies shorten the
# text further but leave fewer training tokens in a 1 MB corpus.
VOCAB_SIZE = 1024

# --- Model shape --------------------------------------------------------------
# Sized to train in about 50 minutes on a laptop CPU: 1.3M parameters, about
# five times the baseline's.
BLOCK_SIZE = 256  # tokens of context, about 600 characters
N_EMBD = 128
N_LAYER = 6
N_HEAD = 4  # query heads, 32 dimensions each
N_KV_HEAD = 2  # key/value heads, each shared by two query heads
FFN_HIDDEN = 384  # SwiGLU width, three times N_EMBD
DROPOUT = 0.2  # a 1 MB corpus is easy to memorise

# --- Training -----------------------------------------------------------------
BATCH_SIZE = 16
MAX_ITERS = 1400  # about 2 seconds each on an i3-1215U
EVAL_INTERVAL = 200
EVAL_ITERS = 20  # batches averaged per estimate; the baseline used 200

# Warmup-stable-decay: ramp up over WARMUP_ITERS, hold, then fall linearly to
# zero over the last DECAY_FRACTION of training.
WARMUP_ITERS = 100
DECAY_FRACTION = 0.2

# Muon's learning rate is on a different scale from AdamW's, because Muon
# normalises each update matrix before applying it.
MUON_LR = 0.02
ADAMW_LR = 3e-3
WEIGHT_DECAY = 0.1  # matrices only; embeddings and norm scales are not decayed
GRAD_CLIP = 1.0
