"""
Hyperparameters and file locations, as plain module constants.

WHY THIS EXISTS
    Every number that shapes the model or its training is here, so tuning a
    run is editing a number rather than reading the code.

WHAT train.py DID
    The same thing, at the top of the one file: batch_size, block_size,
    n_embd and the rest as globals, plus the corpus path hardcoded to
    /home/alex/textfiles/input.txt.

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

# --- Tokenizer ----------------------------------------------------------------
# 256 byte tokens plus 768 learned merges. Bigger vocabularies shorten the
# text further but leave fewer training tokens in a 1 MB corpus.
VOCAB_SIZE = 1024
