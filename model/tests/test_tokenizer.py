"""Tests for the byte-level BPE tokenizer."""

import pytest
from tokenizers import Tokenizer

from shakespeare_model.config import DATA_PATH
from shakespeare_model.tokenizer import train_tokenizer

# A slice keeps the suite fast. The first 200,000 characters are the opening
# plays; the held-out slice further on is text the tokenizer never saw.
TRAINING_CHARACTERS = 200_000


@pytest.fixture(scope="module")
def corpus() -> str:
    """Return the Tiny Shakespeare text."""
    return DATA_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tokenizer(corpus: str) -> Tokenizer:
    """Return a tokenizer trained on the start of the corpus."""
    return train_tokenizer(corpus[:TRAINING_CHARACTERS], vocab_size=512)


def test_vocabulary_is_the_requested_size(tokenizer: Tokenizer) -> None:
    """Training stops when the vocabulary reaches vocab_size."""
    assert tokenizer.get_vocab_size() == 512


def test_text_outside_the_corpus_round_trips_exactly(tokenizer: Tokenizer) -> None:
    """Accents, emoji and curly quotes encode and decode back unchanged."""
    # train.py raised KeyError on the first character here, because it had
    # never seen it. Byte-level BPE falls back to raw bytes instead.
    text = "Café, naïve — “quoth” the 🎭 ROMEO:\n\tΩ"

    ids = tokenizer.encode(text).ids

    assert tokenizer.decode(ids) == text


def test_shakespeare_needs_far_fewer_tokens_than_characters(
    tokenizer: Tokenizer, corpus: str
) -> None:
    """Held-out Shakespeare needs well under one token per character."""
    held_out = corpus[500_000:510_000]

    ids = tokenizer.encode(held_out).ids

    # About 1.9 characters per token for this small vocabulary, and 2.4 for
    # the full 1024-token one. train.py was exactly 1.0.
    assert len(ids) < 0.6 * len(held_out)
    assert tokenizer.decode(ids) == held_out
