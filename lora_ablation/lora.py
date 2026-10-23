"""Minimal, dependency-free LoRA (Hu et al., 2021).

    W x  ->  W x + (alpha / r) * B A x,   A: r x d_in,  B: d_out x r

B starts at zero, so an adapted model is exactly the base model at step 0.
The scale alpha/r is held constant across ranks in this study (alpha = 2r),
so changing r changes adapter capacity, not the effective learning rate.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev = base.weight.device
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.merged:
            out = out + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return out

    @torch.no_grad()
    def merge(self) -> None:
        """Fold BA into W so inference costs exactly what the base model costs."""
        if not self.merged:
            self.base.weight += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True


def apply_lora(model: nn.Module, rank: int, alpha: float, target_modules: list[str],
               dropout: float = 0.0) -> nn.Module:
    """Freeze the model, then wrap every nn.Linear whose attribute name is targeted."""
    for p in model.parameters():
        p.requires_grad_(False)
    targets = [
        (parent, name, child)
        for parent in model.modules()
        for name, child in parent.named_children()
        if isinstance(child, nn.Linear) and name in target_modules
    ]
    if not targets:
        raise ValueError(f"No nn.Linear named any of {target_modules}")
    for parent, name, child in targets:
        setattr(parent, name, LoRALinear(child, rank, alpha, dropout))
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()
    return model


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
