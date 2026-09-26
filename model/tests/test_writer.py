"""Tests for sampling, generation and checkpoints."""

from itertools import islice
from pathlib import Path

import pytest
import torch

from shakespeare_model.config import CHECKPOINT_PATH, DATA_PATH
from shakespeare_model.model import ShakespeareModel
from shakespeare_model.tokenizer import train_tokenizer
from shakespeare_model.writer import (
    MAX_OVERRUN_CHARACTERS,
    Writer,
    generate_token_ids,
    reply_is_finished,
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
    """The text sent piece by piece is exactly the text of the tokens generated."""
    prompt_ids = writer.tokenizer.encode("ROMEO:\n").ids
    torch.manual_seed(1)
    new_ids = list(islice(generate_token_ids(writer.model, prompt_ids, 0.8), 20))

    torch.manual_seed(1)
    pieces = list(islice(writer.stream("ROMEO:\n", 1000, 0.8), 20))

    assert "".join(pieces) == writer.tokenizer.decode(new_ids)


def test_a_reply_ends_only_once_long_enough_and_between_speeches() -> None:
    """Mid-speech is never the end, and neither is a speech ending too early."""
    long_mid_speech = "ROMEO:\nBut soft, what light through yonder window"
    short_but_ended = "Adieu.\n\n"
    long_and_ended = long_mid_speech + " breaks?\n\n"

    assert not reply_is_finished(long_mid_speech, length=20)
    assert not reply_is_finished(short_but_ended, length=20)
    assert reply_is_finished(long_and_ended, length=20)


def test_a_speaker_name_with_nothing_said_is_not_an_ending() -> None:
    """A reply never ends on a name followed by a blank line."""
    ends_on_bare_name = "ROMEO:\nO, let's away.\n\nJULIET:\n\n"
    ends_on_a_line = "ROMEO:\nO, let's away.\n\n"
    continues_the_prompt = "O, let's away.\n\n"

    assert not reply_is_finished(ends_on_bare_name, length=10)
    assert reply_is_finished(ends_on_a_line, length=10)
    # The prompt held the name, so the reply is the speech itself.
    assert reply_is_finished(continues_the_prompt, length=10)


def test_a_speech_that_never_ends_is_cut_off_eventually() -> None:
    """A model that never writes a blank line still stops, MAX_OVERRUN past length."""
    endless = "la " * 1000

    assert not reply_is_finished(endless[: 99 + MAX_OVERRUN_CHARACTERS], length=100)
    assert reply_is_finished(endless[: 100 + MAX_OVERRUN_CHARACTERS], length=100)


def test_the_reply_carries_on_to_the_end_of_the_speech(
    writer: Writer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the requested length, the reply stops at the next blank line."""
    script = "Good night, good night!\nParting is such sweet sorrow.\n\nROMEO:\nSleep"
    script_ids = writer.tokenizer.encode(script).ids

    def scripted(model: ShakespeareModel, prompt_ids: list[int], temperature: float):
        """Yield the script's tokens instead of sampling."""
        yield from script_ids

    monkeypatch.setattr("shakespeare_model.writer.generate_token_ids", scripted)

    reply = "".join(writer.stream("JULIET:\n", length=10, temperature=0.8))

    # 10 characters are reached halfway through the first line; the reply
    # still finishes Juliet's speech, and does not start Romeo's.
    assert reply == "Good night, good night!\nParting is such sweet sorrow.\n\n"


def test_the_model_never_reads_more_than_its_context_at_once(writer: Writer) -> None:
    """A long prompt is cut to its end, and a long reply re-reads recent text."""
    # (tokens already cached, ids fed) for every call to the model.
    calls = []

    def record(module: torch.nn.Module, args: tuple, kwargs: dict, output: object):
        """Keep what this forward call was given."""
        cache = kwargs.get("cache")
        cached = cache[0][0].size(2) if cache else 0
        calls.append((cached, args[0][0].tolist()))

    handle = writer.model.register_forward_hook(record, with_kwargs=True)
    try:
        new_ids = list(
            islice(generate_token_ids(writer.model, list(range(100)), 0.8), 100)
        )
    finally:
        handle.remove()

    assert calls[0] == (0, list(range(100))[-BLOCK_SIZE:])
    for cached, ids in calls:
        assert cached + len(ids) <= BLOCK_SIZE
    # A hundred new tokens is three times the context, and none were refused.
    assert len(new_ids) == 100


def test_a_saved_checkpoint_writes_the_same_text_when_loaded(
    writer: Writer, tmp_path: Path
) -> None:
    """Weights, shape and tokenizer all survive the round trip to disk."""
    path = tmp_path / "tiny.pt"
    save_checkpoint(path, writer.model, writer.tokenizer)

    loaded = Writer.from_checkpoint(path)

    torch.manual_seed(3)
    original = "".join(islice(writer.stream("JULIET:\n", 1000, 0.8), 15))
    torch.manual_seed(3)
    reloaded = "".join(islice(loaded.stream("JULIET:\n", 1000, 0.8), 15))
    assert reloaded == original
    assert loaded.model.output.weight is loaded.model.token_embedding.weight


def test_the_committed_checkpoint_writes_whole_speeches() -> None:
    """The trained model in the repository loads, and ends where a speech ends."""
    # The web service serves this file. A change to the model's shape or to
    # the checkpoint format would break the demo while every other test,
    # which builds its own tiny model, still passed.
    writer = Writer.from_checkpoint(CHECKPOINT_PATH)
    torch.manual_seed(0)

    reply = "".join(writer.stream("ROMEO:\n", length=100, temperature=0.8))

    assert len(reply) >= 100
    assert reply.endswith("\n\n")
