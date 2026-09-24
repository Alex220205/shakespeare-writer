"""Tests for the float32 Muon optimizer."""

import pytest
import torch

from shakespeare_model.muon import Muon, orthogonalise


@pytest.mark.parametrize("shape", [(64, 192), (192, 64)])
def test_orthogonalise_evens_out_the_singular_values(shape: tuple[int, int]) -> None:
    """Singular values spread over a wide range all end up close to 1."""
    torch.manual_seed(0)
    matrix = torch.randn(shape)
    before = torch.linalg.svdvals(matrix)
    assert before.max() / before.min() > 3

    after = torch.linalg.svdvals(orthogonalise(matrix))

    # Five steps of the tuned iteration land between roughly 0.7 and 1.2,
    # not exactly on 1; Keller Jordan found that trains just as well.
    assert after.min() > 0.5
    assert after.max() < 1.5


@pytest.mark.parametrize("shape", [(64, 192), (192, 64)])
def test_a_step_matches_torch_muon(shape: tuple[int, int]) -> None:
    """Two steps move the weights as torch.optim.Muon does, within bfloat16 rounding."""
    torch.manual_seed(0)
    start = torch.randn(shape) * 0.1
    gradients = [torch.randn(shape), torch.randn(shape)]

    ours = torch.nn.Parameter(start.clone())
    theirs = torch.nn.Parameter(start.clone())
    our_optimizer = Muon([ours], lr=0.02, weight_decay=0.1)
    their_optimizer = torch.optim.Muon([theirs], lr=0.02, weight_decay=0.1)
    for gradient in gradients:
        ours.grad = gradient.clone()
        theirs.grad = gradient.clone()
        our_optimizer.step()
        their_optimizer.step()

    our_change = ours.detach() - start
    their_change = theirs.detach() - start
    difference = (our_change - their_change).norm() / their_change.norm()
    # torch rounds to bfloat16, about 3 significant figures, so the two agree
    # to a few percent. A wrong scale, momentum or decay is off by far more.
    assert difference < 0.05
