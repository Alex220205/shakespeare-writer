"""
Training: tokenizer, model, and the best checkpoint.

    uv run python -m shakespeare_model.train

WHY THIS EXISTS
    Turns the corpus into a checkpoint the web service can load.

WHAT train.py DID
    The same outline, which is kept: read the text, encode it, split it 90/10,
    sample random windows, average the loss over many batches every so often,
    and print a sample at the end. It trained with AdamW at a fixed 1e-3,
    reported loss in nats per character, and kept only the final weights.

WHAT CHANGED AND WHY
    - Two optimizers. Muon (muon.py; used to train Kimi K2 and Moonlight)
      updates the weight matrices inside the blocks. It orthogonalises each
      update, so every direction in the matrix moves by a similar amount
      instead of a few dominating. Over 250 steps on this corpus it reached
      2.36 bits per byte against AdamW's 2.45. The embedding and the norm
      scales are not matrices that transform vectors, so AdamW keeps them.
    - A warmup-stable-decay learning rate (MiniCPM, DeepSeek-V3): a short ramp,
      a long plateau, then a linear fall to zero, where most of the final
      improvement happens.
    - Gradient clipping, so one bad batch cannot throw the weights off.
    - Validation loss is also reported in bits per byte, which does not
      depend on the tokenizer, so this model and train.py can be compared.
    - The checkpoint is saved whenever validation loss improves. A small
      corpus is memorised eventually, and the best model is rarely the last.
"""

import math
import time

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR

from shakespeare_model.config import (
    ADAMW_LR,
    BATCH_SIZE,
    BLOCK_SIZE,
    CHECKPOINT_PATH,
    DATA_PATH,
    DECAY_FRACTION,
    DROPOUT,
    EVAL_INTERVAL,
    EVAL_ITERS,
    FFN_HIDDEN,
    GRAD_CLIP,
    MAX_ITERS,
    MUON_LR,
    N_EMBD,
    N_HEAD,
    N_KV_HEAD,
    N_LAYER,
    SEED,
    WARMUP_ITERS,
    WEIGHT_DECAY,
)
from shakespeare_model.model import ShakespeareModel
from shakespeare_model.muon import Muon
from shakespeare_model.tokenizer import train_tokenizer
from shakespeare_model.writer import Writer, save_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick random windows of the data, and the same windows shifted by one."""
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(DEVICE), y.to(DEVICE)


def compute_loss(
    model: ShakespeareModel, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Return the cross-entropy of the model's predictions against the targets."""
    logits, _ = model(x)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


@torch.no_grad()
def estimate_loss(
    model: ShakespeareModel, splits: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Average the loss over EVAL_ITERS random batches of each split."""
    model.eval()
    averages = {}
    for name, data in splits.items():
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE)
            losses[k] = compute_loss(model, x, y).item()
        averages[name] = losses.mean().item()
    model.train()
    return averages


def bits_per_byte(loss: float, token_count: int, byte_count: int) -> float:
    """Convert a loss in nats per token into bits per byte of text."""
    # Loss per token cannot be compared between tokenizers: a token here is
    # 2.4 characters on average, and in train.py it was one. Spreading the
    # loss over the bytes of the text removes that difference.
    return loss * token_count / byte_count / math.log(2)


def build_optimizers(model: ShakespeareModel) -> list[torch.optim.Optimizer]:
    """Return Muon for the matrices inside the blocks and AdamW for the rest."""
    matrices = []
    everything_else = []
    # named_parameters lists the tied embedding/output weight only once.
    for name, parameter in model.named_parameters():
        if name.startswith("blocks.") and parameter.ndim == 2:
            matrices.append(parameter)
        else:
            everything_else.append(parameter)

    muon = Muon(matrices, lr=MUON_LR, weight_decay=WEIGHT_DECAY)
    adamw = torch.optim.AdamW(everything_else, lr=ADAMW_LR, weight_decay=0.0)
    return [muon, adamw]


def lr_multiplier(step: int) -> float:
    """Scale the learning rate: warm up, hold steady, then decay to zero."""
    if step < WARMUP_ITERS:
        return (step + 1) / WARMUP_ITERS

    decay_start = int(MAX_ITERS * (1 - DECAY_FRACTION))
    if step < decay_start:
        return 1.0

    progress = (step - decay_start) / (MAX_ITERS - decay_start)
    return 1.0 - progress


def train_step(
    model: ShakespeareModel,
    optimizers: list[torch.optim.Optimizer],
    data: torch.Tensor,
) -> None:
    """Take one optimisation step on a random training batch."""
    x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE)
    loss = compute_loss(model, x, y)

    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    for optimizer in optimizers:
        optimizer.step()


def main() -> None:
    """Train the tokenizer and the model, keeping the best checkpoint."""
    torch.manual_seed(SEED)
    text = DATA_PATH.read_text(encoding="utf-8")
    tokenizer = train_tokenizer(text)

    data = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest validation
    splits = {"train": data[:n], "val": data[n:]}
    val_bytes = len(tokenizer.decode(splits["val"].tolist()).encode("utf-8"))
    print(f"{len(text):,} characters -> {len(data):,} tokens")

    model = ShakespeareModel(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_kv_head=N_KV_HEAD,
        ffn_hidden=FFN_HIDDEN,
        dropout=DROPOUT,
    ).to(DEVICE)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"{parameter_count:,} parameters, training on {DEVICE}")

    optimizers = build_optimizers(model)
    schedulers = [LambdaLR(optimizer, lr_multiplier) for optimizer in optimizers]
    best_val_loss = float("inf")
    started = time.monotonic()

    for step in range(MAX_ITERS + 1):
        # Evaluate every EVAL_INTERVAL steps, and once more after the last.
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS:
            losses = estimate_loss(model, splits)
            val_bpb = bits_per_byte(losses["val"], len(splits["val"]), val_bytes)
            minutes = (time.monotonic() - started) / 60
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, val bits/byte {val_bpb:.3f} "
                f"({minutes:.1f} min)"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(CHECKPOINT_PATH, model, tokenizer)
        if step == MAX_ITERS:
            break

        train_step(model, optimizers, splits["train"])
        for scheduler in schedulers:
            scheduler.step()

    # Sample from the file just written, not the model in memory, so a
    # checkpoint that does not load is found here rather than by the website.
    writer = Writer.from_checkpoint(CHECKPOINT_PATH)
    print("ROMEO:\n" + "".join(writer.stream("ROMEO:\n", 500, temperature=0.8)))


if __name__ == "__main__":
    main()
