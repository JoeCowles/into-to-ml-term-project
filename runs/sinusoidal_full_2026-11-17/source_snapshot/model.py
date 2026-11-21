"""Term-project decoder-only Transformer, trained without pretrained weights."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 99
    block_size: int = 1024
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    attention: str = "sdpa"
    position_encoding: str = "learned"
    position_scale: float = 1.0

    def __post_init__(self):
        if min(self.vocab_size, self.block_size, self.n_embd, self.n_head, self.n_layer) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.attention not in {"manual", "sdpa"} or not 0 <= self.dropout < 1:
            raise ValueError("Invalid attention implementation or dropout")
        if self.position_encoding not in {"learned", "sinusoidal"}:
            raise ValueError("Invalid position encoding")
        if not math.isfinite(self.position_scale) or self.position_scale <= 0:
            raise ValueError("Position scale must be finite and positive")


class SinusoidalPositionEmbedding(nn.Module):
    """Fixed sin/cos positions with frequencies 10000^(-2i / width)."""

    def __init__(self, length, width):
        super().__init__()
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, width, 2, dtype=torch.float32)
                                * (-math.log(10000.0) / width))
        angles = positions * frequencies
        table = torch.empty(length, width)
        table[:, 0::2] = angles.sin()
        table[:, 1::2] = angles[:, :width // 2].cos()
        self.register_buffer("weight", table)

    def forward(self, positions):
        return self.weight[positions]


def causal_mask(t, device=None):
    return torch.ones(t, t, dtype=torch.bool, device=device).tril()


def attention(q, k, v, mask):
    """Explicit attention: scaled QK^T, causal mask, softmax, then V."""
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    weights = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
    return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.d_k = config.n_embd // config.n_head
        self.implementation = config.attention
        # Three slices are W_Q, W_K and W_V; proj is W_O.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = [z.view(b, t, self.n_head, self.d_k).transpose(1, 2)
                   for z in self.qkv(x).split(c, dim=-1)]
        if self.implementation == "manual":
            out = attention(q, k, v, causal_mask(t, x.device))
        else:
            # Fused tensor operation, not a prebuilt Transformer. Output dropout
            # matches the manual path; no attention-weight dropout.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.dropout(self.proj(out.transpose(1, 2).contiguous().view(b, t, c)))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.n_embd
        self.net = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(),
                                 nn.Linear(4 * c, c), nn.Dropout(config.dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(config.n_embd), nn.LayerNorm(config.n_embd)
        self.attn, self.ffwd = MultiHeadAttention(config), FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)
        # Replace only after initialization so shared weights and the RNG state
        # exactly match the learned-position baseline for the same seed.
        if config.position_encoding == "sinusoidal":
            self.position_embedding = SinusoidalPositionEmbedding(config.block_size, config.n_embd)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        if idx.ndim != 2 or not 0 < idx.shape[1] <= self.block_size:
            raise ValueError(f"Expected (batch, time) with 1 <= time <= {self.block_size}")
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.token_embedding(idx) + self.config.position_scale * self.position_embedding(pos)
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=192, temperature=0.0, stop_id=None, banned_ids=()):
        """Single-prompt decoding; temperature zero selects greedy next tokens."""
        if idx.shape[0] != 1 or temperature < 0 or max_new_tokens < 0:
            raise ValueError("Expected one prompt and nonnegative generation settings")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.block_size:])
                logits = logits[:, -1, :].float()
                logits[:, list(banned_ids)] = -float("inf")
                if temperature == 0:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    next_id = torch.multinomial((logits / temperature).softmax(dim=-1), 1)
                idx = torch.cat((idx, next_id), dim=1)
                if stop_id is not None and next_id.item() == stop_id:
                    break
            return idx
        finally:
            self.train(was_training)

    def config_dict(self):
        return asdict(self.config)
