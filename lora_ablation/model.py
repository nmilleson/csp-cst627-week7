"""Small pre-LN transformer encoder that forecasts the next 24 hours.

Every timestep (168 history + 24 horizon) is one token. Horizon tokens carry
covariates but a masked demand value; the head reads only the horizon tokens.
Linear layers are named (q_proj, k_proj, v_proj, out_proj, ff1, ff2) so LoRA
can target them by name.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape

        def heads(z):
            return z.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2)

        o = F.scaled_dot_product_attention(
            heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x)),
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(o.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.drop(self.ff2(F.gelu(self.ff1(self.ln2(x)))))


class DemandForecaster(nn.Module):
    def __init__(self, n_features: int, context: int, horizon: int, d_model: int,
                 n_heads: int, n_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.horizon = horizon
        self.in_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, context + horizon, d_model) * 0.02)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x) + self.pos
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h[:, -self.horizon:])).squeeze(-1)


def build_model(cfg: dict) -> DemandForecaster:
    from .windows import N_FEATURES

    m = cfg["model"]
    return DemandForecaster(
        n_features=N_FEATURES, context=cfg["window"]["context_hours"],
        horizon=cfg["window"]["horizon_hours"], d_model=m["d_model"], n_heads=m["n_heads"],
        n_layers=m["n_layers"], d_ff=m["d_ff"], dropout=m["dropout"],
    )
