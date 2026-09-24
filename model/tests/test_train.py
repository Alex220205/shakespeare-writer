"""Tests for the pieces of the training loop."""

import math

import torch

from shakespeare_model.config import MAX_ITERS, WARMUP_ITERS
from shakespeare_model.model import ShakespeareModel
from shakespeare_model.train import (
    bits_per_byte,
    build_optimizers,
    get_batch,
    lr_multiplier,
)


def test_targets_are_the_inputs_shifted_by_one() -> None:
    """Each target is the token that follows the input at the same position."""
    data = torch.arange(100)

    x, y = get_batch(data, batch_size=4, block_size=8)

    assert x.shape == (4, 8)
    assert torch.equal(y, x + 1)


def test_muon_gets_only_the_matrices_inside_the_blocks() -> None:
    """The embedding and the norm scales go to AdamW; every parameter goes somewhere."""
    model = ShakespeareModel(
        vocab_size=64,
        block_size=16,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        ffn_hidden=64,
    )

    muon, adamw = build_optimizers(model)
    muon_ids = {id(p) for p in muon.param_groups[0]["params"]}
    adamw_ids = {id(p) for p in adamw.param_groups[0]["params"]}

    for parameter in muon.param_groups[0]["params"]:
        assert parameter.ndim == 2
    assert id(model.token_embedding.weight) in adamw_ids
    assert id(model.blocks[0].attention_norm.weight) in adamw_ids
    # 2 layers x 7 matrices (wq, wk, wv, wo, w_gate, w_up, w_down).
    assert len(muon_ids) == 14
    all_ids = {id(p) for p in model.parameters()}
    assert muon_ids | adamw_ids == all_ids
    assert not muon_ids & adamw_ids


def test_learning_rate_warms_up_holds_then_decays_to_zero() -> None:
    """The schedule ramps up, stays flat through the middle, and ends near zero."""
    assert lr_multiplier(0) < 0.05
    assert lr_multiplier(WARMUP_ITERS // 2) < 1.0
    assert lr_multiplier(WARMUP_ITERS) == 1.0
    assert lr_multiplier(MAX_ITERS // 2) == 1.0
    assert lr_multiplier(MAX_ITERS - 1) < 0.01

    decay = [lr_multiplier(step) for step in range(MAX_ITERS // 2, MAX_ITERS)]
    assert decay == sorted(decay, reverse=True)


def test_bits_per_byte_spreads_the_loss_over_the_text() -> None:
    """A token covering two bytes with a loss of one bit is half a bit per byte."""
    one_bit = math.log(2)

    assert math.isclose(bits_per_byte(one_bit, token_count=10, byte_count=10), 1.0)
    assert math.isclose(bits_per_byte(one_bit, token_count=10, byte_count=20), 0.5)
