"""
Muon, with its Newton-Schulz step in float32 so that it runs at speed on a CPU.

WHY THIS EXISTS
    Muon (Keller Jordan, 2024; used to train Kimi K2 and Moonlight) updates
    the weight matrices inside the transformer blocks. PyTorch has shipped it
    since 2.9 as torch.optim.Muon, and this file would not exist if that ran
    well on a laptop CPU.

WHY NOT torch.optim.Muon
    It converts each update to bfloat16 before orthogonalising it. On a GPU
    that halves the work. A CPU without bfloat16 matrix instructions, which
    includes most laptops and the one this model was trained on, has to
    emulate every bfloat16 multiply. One optimizer step took 5.2 seconds,
    against 13 ms for AdamW. The same maths in float32 is fast.
    tests/test_muon.py checks that this takes the same step as
    torch.optim.Muon, to within bfloat16 rounding.

WHAT IT DOES
    AdamW scales each weight's update by that weight's own gradient history,
    so a few directions in a matrix tend to dominate. Muon takes the momentum
    of the gradient and orthogonalises it: every singular value of the update
    is pushed towards 1, keeping its directions but evening out how far each
    one moves. In practice that reaches a given loss in fewer steps.
"""

import torch

# Coefficients of the five-step quintic Newton-Schulz iteration, tuned by
# Keller Jordan to push singular values towards 1 as fast as possible. They
# are torch.optim.Muon's defaults too.
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5


def orthogonalise(matrix: torch.Tensor) -> torch.Tensor:
    """Push every singular value of a matrix towards 1, keeping its directions."""
    a, b, c = NS_COEFFICIENTS

    # Each step multiplies by x @ x.T, which is smaller for a wide matrix.
    tall = matrix.size(0) > matrix.size(1)
    x = matrix.T if tall else matrix

    # The iteration converges only when no singular value is above 1.
    x = x / x.norm().clamp(min=1e-7)
    for _ in range(NS_STEPS):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x

    return x.T if tall else x


class Muon(torch.optim.Optimizer):
    """Orthogonalised momentum for 2D weight matrices."""

    def __init__(
        self, params, lr: float, momentum: float = 0.95, weight_decay: float = 0.0
    ):
        """Register the matrices to optimise and their settings."""
        defaults = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        """Update every matrix that has a gradient."""
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    self.update(parameter, group)

    def update(self, parameter: torch.Tensor, group: dict) -> None:
        """Apply one Muon step to one matrix."""
        state = self.state[parameter]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(parameter)

        # Nesterov momentum: look ahead along the running average.
        state["momentum"].lerp_(parameter.grad, 1 - group["momentum"])
        direction = parameter.grad.lerp(state["momentum"], group["momentum"])
        direction = orthogonalise(direction)

        # An orthogonalised tall matrix has smaller entries than a wide one of
        # the same size; this scales them back (torch's "original" setting).
        rows, columns = parameter.shape
        lr = group["lr"] * max(1.0, rows / columns) ** 0.5

        parameter.mul_(1 - group["lr"] * group["weight_decay"])
        parameter.add_(direction, alpha=-lr)
