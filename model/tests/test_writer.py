"""Tests for sampling, generation and checkpoints."""

from pathlib import Path

import pytest
import torch

from shakespeare_model.config import CHECKPOINT_PATH, DATA_PATH
from shakespeare_model.model import ShakespeareModel
from shakespeare_model.tokenizer import train_tokenizer
from shakespeare_model.writer import (
    Writer,
    generate_token_ids,
    sample_next_token,
    save_checkpoint,
)

BLOCK_SIZE = 32


@pytest.fixture(scope="module")
def writer() -> Writer:
    """Return an untrained tiny model with a real, small tokenizer."""
    corpus = DATA_PATH.read_text(encoding="utf-8")
    tokenizer = train_tokenizer(corpus[:20_000], vocab_size=300)
    torch.manual_seed(0)
    model = ShakespeareModel(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=BLOCK_SIZE,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        ffn_hidden=64,
    )
    model.eval()
    return Writer(model, tokenizer)


def test_min_p_rules_out_tokens_far_less_likely_than_the_favourite() -> None:
    """With min_p 0.05 a token at 2% never appears beside a favourite at 60%."""
    probabilities = torch.tensor([[0.60, 0.30, 0.08, 0.02]])
    logits = probabilities.log()
    torch.manual_seed(0)

    drawn = set()
    for _ in range(300):
        drawn.add(sample_next_token(logits, temperature=1.0, min_p=0.05).item())

    # 0.02 is below 0.05 * 0.60 = 0.03, and 0.08 is above it. A sampler that
    # only ever returned the favourite would fail the equality too.
    assert drawn == {0, 1, 2}


def test_streamed_pieces_join_up_to_the_decoded_reply(writer: Writer) -> None:
    """The text sent piece by piece is exactly the text of the whole reply."""
    prompt_ids = writer.tokenizer.encode("ROMEO:\n").ids
    torch.manual_seed(1)
    new_ids = list(generate_token_ids(writer.model, prompt_ids, 20, 0.8))

    torch.manual_seed(1)
    pieces = list(writer.stream("ROMEO:\n", max_new_tokens=20, temperature=0.8))

    assert len(pieces) == 20
    assert "".join(pieces) == writer.tokenizer.decode(new_ids)


def test_only_the_end_of_a_long_prompt_is_read(writer: Writer) -> None:
    """A prompt too long for the context is cut to its last tokens."""
    # Records the token ids of every call to the model. Comparing replies
    # instead is not enough: an untrained model's output barely moves with
    # extra context, so the same seed draws the same tokens either way.
    fed = []

    def record(module: torch.nn.Module, args: tuple, output: object) -> None:
        """Keep the ids passed to this forward call."""
        fed.append(args[0][0].tolist())

    handle = writer.model.register_forward_hook(record)
    try:
        list(generate_token_ids(writer.model, list(range(100)), 10, 0.8))
    finally:
        handle.remove()

    assert fed[0] == list(range(100))[-(BLOCK_SIZE - 10) :]
    tokens_seen = 0
    for ids in fed:
        tokens_seen += len(ids)
    assert tokens_seen <= BLOCK_SIZE


def test_a_reply_longer_than_the_context_is_refused(writer: Writer) -> None:
    """Asking for block_size new tokens would leave no room for the prompt."""
    with pytest.raises(ValueError, match="block size"):
        list(generate_token_ids(writer.model, [1, 2, 3], BLOCK_SIZE, 0.8))


def test_a_saved_checkpoint_writes_the_same_text_when_loaded(
    writer: Writer, tmp_path: Path
) -> None:
    """Weights, shape and tokenizer all survive the round trip to disk."""
    path = tmp_path / "tiny.pt"
    save_checkpoint(path, writer.model, writer.tokenizer)

    loaded = Writer.from_checkpoint(path)

    torch.manual_seed(3)
    original = "".join(writer.stream("JULIET:\n", 15, 0.8))
    torch.manual_seed(3)
    reloaded = "".join(loaded.stream("JULIET:\n", 15, 0.8))
    assert reloaded == original
    assert loaded.model.output.weight is loaded.model.token_embedding.weight


def test_the_committed_checkpoint_still_loads_and_writes() -> None:
    """The trained model in the repository works with the current code."""
    # The web service serves this file. A change to the model's shape or to
    # the checkpoint format would break the demo while every other test,
    # which builds its own tiny model, still passed.
    writer = Writer.from_checkpoint(CHECKPOINT_PATH)

    reply = "".join(writer.stream("ROMEO:\n", max_new_tokens=10, temperature=0.8))

    assert reply != ""
