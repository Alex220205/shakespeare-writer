"""
Turning a trained model into text: sampling, the KV cache, and checkpoints.

WHY THIS EXISTS
    The model only scores what could come next. This module picks a token,
    feeds it back, and repeats. It also saves and loads the one file that
    holds a trained model, which is what the web service reads.

WHAT train.py DID
    `generate` ran the whole context through the model for every new token,
    kept only the last position's logits, and sampled from the full softmax
    at temperature 1. Every step recomputed keys and values for tokens that
    had not changed, and one unlucky draw from the long tail of unlikely
    tokens could derail a line.

WHAT CHANGED AND WHY
    - A KV cache. The prompt is read once. After that each step feeds only
      the newest token, and attention reuses the keys and values already
      worked out for everything before it.
    - Temperature plus min-p sampling (ICLR 2025, now in Hugging Face
      Transformers, vLLM and SGLang). A token less than MIN_P times as likely
      as the favourite is ruled out. When the model is confident that removes
      almost everything; when it is unsure, many options stay open.
    - A reply runs to about the length asked for, then carries on until the
      speech it is in has ended, so it never stops halfway through a word.
      In the corpus a blank line ends every speech; the next starts with a
      speaker's name.
    - Replies are not limited by the context. When the cache holds the
      block_size tokens the model was trained to read, generation re-reads
      the most recent half with a fresh cache, as train.py's
      idx[:, -block_size:] crop did.
    - The weights, the model's shape and its tokenizer travel together in one
      checkpoint, loaded with weights_only=True, which refuses to run any
      pickled code a tampered file might contain.
"""

from collections.abc import Iterator
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from shakespeare_model.model import ShakespeareModel

# The min-p paper recommends 0.05 to 0.1. Lower keeps more variety.
MIN_P = 0.05

# How far past the requested length a reply may run while it waits for the
# speech to end. Generated speeches are usually a few lines, so this only
# guards against a model that never finishes one.
MAX_OVERRUN_CHARACTERS = 1000


def sample_next_token(
    logits: torch.Tensor, temperature: float, min_p: float = MIN_P
) -> torch.Tensor:
    """Pick the next token id from one position's logits, shape (B, vocab)."""
    probabilities = F.softmax(logits / temperature, dim=-1)

    # Rule out every token less than min_p times as likely as the favourite.
    # The favourite itself always survives, so there is always a choice.
    favourite = probabilities.max(dim=-1, keepdim=True).values
    probabilities = probabilities.masked_fill(probabilities < min_p * favourite, 0.0)

    # multinomial rescales what is left, so it need not sum to one.
    return torch.multinomial(probabilities, num_samples=1)  # (B, 1)


@torch.no_grad()
def generate_token_ids(
    model: ShakespeareModel, prompt_ids: list[int], temperature: float
) -> Iterator[int]:
    """Yield new token ids one at a time, continuing the prompt, until stopped."""
    device = model.token_embedding.weight.device

    # Every id so far, kept so the context can be re-read when the cache
    # fills. A prompt longer than the context loses its beginning.
    ids = list(prompt_ids[-model.block_size :])
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    cache = []
    while True:
        # The first pass reads the whole prompt into the cache. From then on,
        # idx is only the token just chosen.
        logits, cache = model(idx, cache=cache)
        next_token = sample_next_token(logits[:, -1, :], temperature)
        token_id = next_token.item()
        ids.append(token_id)
        yield token_id

        # The model has only learned to read block_size tokens at once. When
        # the cache is full, start a fresh one from the most recent half.
        if cache[0][0].size(2) >= model.block_size:
            recent = ids[-(model.block_size // 2) :]
            idx = torch.tensor([recent], dtype=torch.long, device=device)
            cache = []
        else:
            idx = next_token


def reply_is_finished(reply: str, length: int) -> bool:
    """Return True once the reply is long enough and its last speech has ended."""
    if len(reply) >= length + MAX_OVERRUN_CHARACTERS:
        return True
    if len(reply) < length or not reply.endswith("\n\n"):
        return False

    # A speaker's name on its own, as in "JULIET:" then a blank line, is not a
    # finished speech: nobody has said anything yet.
    last_speech = reply.rstrip("\n").split("\n\n")[-1]
    is_bare_name = "\n" not in last_speech and last_speech.endswith(":")
    return not is_bare_name


def save_checkpoint(path: Path, model: ShakespeareModel, tokenizer: Tokenizer) -> None:
    """Write the weights, the model's shape and the tokenizer to one file."""
    checkpoint = {
        "model": model.state_dict(),
        "shape": model.shape,
        "tokenizer": tokenizer.to_str(),
    }
    torch.save(checkpoint, path)


class Writer:
    """A trained model and its tokenizer, ready to continue a prompt."""

    def __init__(self, model: ShakespeareModel, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_checkpoint(cls, path: Path) -> "Writer":
        """Load a checkpoint written by save_checkpoint, ready to generate."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        shape = checkpoint["shape"]
        model = ShakespeareModel(
            vocab_size=shape["vocab_size"],
            block_size=shape["block_size"],
            n_embd=shape["n_embd"],
            n_layer=shape["n_layer"],
            n_head=shape["n_head"],
            n_kv_head=shape["n_kv_head"],
            ffn_hidden=shape["ffn_hidden"],
        )
        model.load_state_dict(checkpoint["model"])
        model.eval()

        tokenizer = Tokenizer.from_str(checkpoint["tokenizer"])
        return cls(model, tokenizer)

    def stream(self, prompt: str, length: int, temperature: float) -> Iterator[str]:
        """Yield at least `length` characters a token at a time, to a speech's end."""
        prompt_ids = self.tokenizer.encode(prompt).ids
        reply = ""
        for token_id in generate_token_ids(self.model, prompt_ids, temperature):
            piece = self.tokenizer.decode([token_id])
            yield piece
            reply += piece
            if reply_is_finished(reply, length):
                return
