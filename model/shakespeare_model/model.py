"""
The transformer: a 2026-style decoder, small enough to train on a laptop CPU.

WHY THIS EXISTS
    Maps a sequence of token ids to a prediction of the next token at every
    position. Training and the web page are plumbing around this.

WHAT THE BASELINE DID
    The classic small-GPT design. A learned table of position embeddings
    added to the token embeddings, LayerNorm, six separate `Head` modules
    each computing softmax(QK^T)V by hand, a ReLU feed-forward four times as
    wide, a separate output layer, and a bias on every linear layer.

WHAT CHANGED AND WHY
    The parts current open models (Llama 3 and 4, Qwen3, Gemma 3, OLMo 2)
    have converged on:

    - RoPE instead of a position table. Queries and keys are rotated by an
      angle proportional to their position, so the score between two tokens
      depends only on how far apart they are. Nothing is learned.
    - RMSNorm instead of LayerNorm. It skips subtracting the mean and trains
      just as well for less work.
    - Grouped-query attention. Four query heads share two key/value heads,
      which halves the KV cache (writer.py) at almost no cost in quality.
    - QK-norm. Queries and keys are RMS-normalised per head before RoPE, so
      attention scores cannot grow without bound and training tolerates
      higher learning rates.
    - SwiGLU instead of ReLU: a gated feed-forward, where one projection
      decides how much of another gets through.
    - The output layer shares its weight with the token embedding, and no
      linear layer has a bias.
    - One fused attention call, F.scaled_dot_product_attention, instead of the
      hand-written softmax. Same maths, less memory, and it applies the causal
      mask itself.
"""

import torch
from torch import nn
from torch.nn import functional as F

# One (keys, values) pair per layer, holding every token seen so far.
KVCache = list[tuple[torch.Tensor, torch.Tensor]]

# The standard RoPE base. The slowest-turning pair of dimensions completes a
# rotation only every ~60,000 positions, far past any context used here.
ROPE_BASE = 10_000.0


def rope_angles(
    head_dim: int, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the cos and sin of the RoPE rotation angle for each position."""
    # One frequency per pair of dimensions, from one radian per position for
    # the first pair down to a very slow turn for the last.
    pair_index = torch.arange(0, head_dim, 2, device=positions.device).float()
    frequencies = 1.0 / (ROPE_BASE ** (pair_index / head_dim))

    angles = torch.outer(positions.float(), frequencies)  # (T, head_dim / 2)
    # apply_rope pairs dimension i with i + head_dim / 2, so both halves of
    # the vector use the same list of angles.
    angles = torch.cat([angles, angles], dim=-1)  # (T, head_dim)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each pair of dimensions in x by its position's angle."""
    # x is (B, heads, T, head_dim). Each pair (a, b) = (x[i], x[i + half])
    # becomes (a cos - b sin, b cos + a sin): a 2D rotation.
    half = x.size(-1) // 2
    first = x[..., :half]
    second = x[..., half:]
    rotated = torch.cat([-second, first], dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    """Causal self-attention with grouped-query heads, QK-norm and RoPE."""

    def __init__(self, n_embd: int, n_head: int, n_kv_head: int, dropout: float):
        """Create the query, key, value and output projections and the QK-norms."""
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head

        self.wq = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(n_head * self.head_dim, n_embd, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        # Dropout on the output only, not on the attention weights as
        # the baseline did. The weights are B x heads x T x T values per layer,
        # and drawing a random mask over them took a third of every training
        # step on a laptop CPU. The output is B x T x n_embd, 32 times fewer.
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attend over this layer's cached tokens and the new ones in x."""
        B, T, _ = x.shape

        # (B, T, heads * head_dim) -> (B, heads, T, head_dim)
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Normalise first, then rotate: RoPE has to be the last thing done to
        # q and k, or the rotation would be partly undone.
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)

        if layer_cache is not None:
            cached_k, cached_v = layer_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        # is_causal=T > 1 covers the only two cases that happen. Training and
        # the first pass over a prompt see T > 1 tokens with nothing cached,
        # and need the triangular mask. Generation then feeds one token at a
        # time, and that token may look at everything before it.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=T > 1,
            enable_gqa=True,  # each key/value head serves n_head / n_kv_head queries
        )
        out = out.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)
        return self.output_dropout(self.wo(out)), (k, v)


class FeedForward(nn.Module):
    """SwiGLU: a feed-forward layer where one projection gates another."""

    def __init__(self, n_embd: int, hidden: int, dropout: float):
        """Create the gate, up and down projections."""
        super().__init__()
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)
        self.w_up = nn.Linear(n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand, gate, and project back down."""
        # silu(gate) decides, for each hidden unit, how much of `up` passes.
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.dropout(self.w_down(gated))


class Block(nn.Module):
    """One transformer layer: attention, then feed-forward, each normed first."""

    def __init__(
        self, n_embd: int, n_head: int, n_kv_head: int, ffn_hidden: int, dropout: float
    ):
        """Create the attention and feed-forward layers, each with its own norm."""
        super().__init__()
        self.attention_norm = nn.RMSNorm(n_embd)
        self.attention = Attention(n_embd, n_head, n_kv_head, dropout)
        self.feed_forward_norm = nn.RMSNorm(n_embd)
        self.feed_forward = FeedForward(n_embd, ffn_hidden, dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Add attention's output, then the feed-forward's, to the residual stream."""
        attended, new_layer_cache = self.attention(
            self.attention_norm(x), cos, sin, layer_cache
        )
        x = x + attended
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x, new_layer_cache


def init_weights(module: nn.Module) -> None:
    """Start every weight small, the way GPT-2 did."""
    if isinstance(module, nn.Linear | nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class ShakespeareModel(nn.Module):
    """A decoder-only transformer that predicts the next token."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        n_kv_head: int,
        ffn_hidden: int,
        dropout: float = 0.0,
    ):
        """Create the embedding, the blocks and the tied output layer."""
        super().__init__()
        # The longest sequence training ever showed the model. writer.py reads
        # it to know when the cache is full and the text must be re-read.
        self.block_size = block_size
        self.head_dim = n_embd // n_head

        # Saved in the checkpoint beside the weights, so a model can be
        # rebuilt at the right size without the config it was trained with.
        # Dropout is left out: it only matters while training.
        self.shape = {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "n_embd": n_embd,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_kv_head": n_kv_head,
            "ffn_hidden": ffn_hidden,
        }

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList()
        for _ in range(n_layer):
            self.blocks.append(Block(n_embd, n_head, n_kv_head, ffn_hidden, dropout))
        self.final_norm = nn.RMSNorm(n_embd)
        self.output = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: the output layer scores each token by how closely the
        # final vector matches that token's embedding. One matrix does both
        # jobs, which is a sixth of this model's parameters saved.
        self.output.weight = self.token_embedding.weight

        self.apply(init_weights)

    def forward(
        self, idx: torch.Tensor, cache: KVCache | None = None
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Return next-token logits for every position, and the updated cache."""
        B, T = idx.shape

        # cache is None while training, and nothing is kept. For generation
        # it is a list, empty before the prompt has been read, and the new
        # tokens are placed after everything already in it.
        start = 0
        if cache:
            start = cache[0][0].size(2)
        positions = torch.arange(start, start + T, device=idx.device)
        cos, sin = rope_angles(self.head_dim, positions)

        x = self.token_embedding(idx)  # (B, T, n_embd)
        new_cache = []
        for layer_index, block in enumerate(self.blocks):
            layer_cache = cache[layer_index] if cache else None
            x, new_layer_cache = block(x, cos, sin, layer_cache)
            new_cache.append(new_layer_cache)
        logits = self.output(self.final_norm(x))  # (B, T, vocab_size)

        if cache is None:
            return logits, None
        return logits, new_cache
