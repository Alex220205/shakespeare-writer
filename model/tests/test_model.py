"""Tests for the transformer."""

import pytest
import torch

from shakespeare_model.model import ShakespeareModel, apply_rope, rope_angles

VOCAB_SIZE = 64


@pytest.fixture
def model() -> ShakespeareModel:
    """Return a tiny model in eval mode, so dropout cannot make results differ."""
    torch.manual_seed(0)
    tiny = ShakespeareModel(
        vocab_size=VOCAB_SIZE,
        block_size=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        ffn_hidden=64,
    )
    tiny.eval()
    return tiny


def test_changing_a_later_token_does_not_change_earlier_predictions(
    model: ShakespeareModel,
) -> None:
    """Position i only sees tokens 0..i, so editing token 7 leaves 0..6 alone."""
    idx = torch.randint(VOCAB_SIZE, (1, 12))
    edited = idx.clone()
    edited[0, 7] = (idx[0, 7] + 1) % VOCAB_SIZE

    logits, _ = model(idx)
    edited_logits, _ = model(edited)

    assert torch.allclose(logits[:, :7], edited_logits[:, :7], atol=1e-6)
    # The positive counterpart: a mask that hid everything would also pass
    # the line above.
    assert not torch.allclose(logits[:, 7:], edited_logits[:, 7:], atol=1e-6)


def test_decoding_with_the_cache_matches_a_full_forward_pass(
    model: ShakespeareModel,
) -> None:
    """Reading a prompt, then one token at a time, gives the same logits."""
    idx = torch.randint(VOCAB_SIZE, (1, 12))
    full_logits, _ = model(idx)

    logits, cache = model(idx[:, :5], cache=[])
    assert torch.allclose(logits, full_logits[:, :5], atol=1e-5)

    for position in range(5, 12):
        logits, cache = model(idx[:, position : position + 1], cache=cache)
        assert torch.allclose(logits[:, 0], full_logits[:, position], atol=1e-5)


def test_training_mode_keeps_no_cache(model: ShakespeareModel) -> None:
    """Without a cache argument nothing is kept, so training holds no extra memory."""
    _, cache = model(torch.randint(VOCAB_SIZE, (2, 8)))

    assert cache is None


def test_output_layer_stays_tied_to_the_embedding_after_loading(
    model: ShakespeareModel,
) -> None:
    """A model rebuilt from a state dict still shares one matrix, and agrees."""
    fresh = ShakespeareModel(
        vocab_size=VOCAB_SIZE,
        block_size=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        ffn_hidden=64,
    )
    fresh.load_state_dict(model.state_dict())
    fresh.eval()
    idx = torch.randint(VOCAB_SIZE, (1, 10))

    assert fresh.output.weight is fresh.token_embedding.weight
    assert torch.allclose(fresh(idx)[0], model(idx)[0])


def test_rope_scores_depend_only_on_how_far_apart_tokens_are() -> None:
    """Moving a query and a key by the same distance leaves their score unchanged."""
    torch.manual_seed(0)
    head_dim = 16
    query = torch.randn(head_dim)
    key = torch.randn(head_dim)

    def score(query_position: int, key_position: int) -> torch.Tensor:
        """Return the dot product of the rotated query and key."""
        positions = torch.tensor([query_position, key_position])
        cos, sin = rope_angles(head_dim, positions)
        rotated = apply_rope(torch.stack([query, key]), cos, sin)
        return rotated[0] @ rotated[1]

    assert torch.allclose(score(3, 1), score(103, 101), atol=1e-4)
    assert not torch.allclose(score(3, 1), score(3, 2), atol=1e-4)
