"""
A byte-level BPE tokenizer, trained on the corpus it will encode.

WHY THIS EXISTS
    The model reads integers, not text. This module turns one into the other.

WHAT THE BASELINE DID
    One integer per character: `chars = sorted(list(set(text)))` gave 65 of
    them, looked up through `stoi` and `itos`. "gentleman" was nine separate
    predictions for the model, and any character outside those 65 (an accent,
    an emoji, a curly quote typed into a prompt) was a KeyError.

WHAT CHANGED AND WHY
    Byte-pair encoding, which is what GPT-4o, Llama 3 and Qwen3 all use.
    Training starts from the 256 possible byte values and keeps merging the
    most frequent neighbouring pair into a new token until there are
    VOCAB_SIZE of them. Common words end up as one token and rare words as a
    few pieces. Because the base alphabet is bytes rather than characters,
    anything a user types can be encoded, even if the corpus never contained
    it.

    Training takes about a second, so train.py retrains the tokenizer on
    every run and saves it inside the model checkpoint. A model and a
    tokenizer from different runs can then never be loaded together.
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from shakespeare_model.config import VOCAB_SIZE


def train_tokenizer(text: str, vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    """Train a byte-level BPE tokenizer on the given text."""
    tokenizer = Tokenizer(models.BPE())

    # Split into words first, with any leading space kept on the word, and
    # map every byte to a printable stand-in. Merges never cross a word
    # boundary, so " king" and " kingdom" are learned as words, not as
    # arbitrary spans of letters.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        # All 256 bytes are in the vocabulary from the start, including ones
        # the corpus never uses. That is what makes any input encodable.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator([text], trainer=trainer)
    return tokenizer
