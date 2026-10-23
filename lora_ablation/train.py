"""Training loop, evaluation, and cost measurement (latency, peak memory)."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .windows import SeriesBank


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    return float("nan")  # CPU peak is not resettable per run; reported as n/a


@torch.no_grad()
def predict(model: nn.Module, bank: SeriesBank, sidx, starts, context: int, horizon: int,
            batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (pred_mw [N,H], true_mw [N,H], normalised MSE)."""
    was_training = model.training
    model.eval()
    preds, trues, sq, n = [], [], 0.0, 0
    for i in range(0, len(starts), batch_size):
        x, yn, mu, sd, y = bank.batch(sidx[i:i + batch_size], starts[i:i + batch_size], context, horizon)
        out = model(x)
        sq += F.mse_loss(out, yn, reduction="sum").item()
        n += yn.numel()
        preds.append((out * sd + mu).cpu().numpy())
        trues.append(y.cpu().numpy())
    model.train(was_training)
    return np.concatenate(preds), np.concatenate(trues), sq / n


def fit(model: nn.Module, bank: SeriesBank, train_pairs, val_pairs, *, context: int, horizon: int,
        steps: int, batch_size: int, lr: float, weight_decay: float, warmup_steps: int,
        eval_every: int, seed: int, log=print) -> dict:
    """AdamW + linear warmup + cosine decay; keeps the best-validation weights.

    Batch indices come from a CPU generator seeded with `seed`, so every ablation
    arm with the same seed sees the identical sequence of training windows.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    gen = torch.Generator().manual_seed(seed)
    train_s, train_t = train_pairs
    # Step 0 = the starting point; if no step beats it, the starting weights are kept.
    *_, best_val = predict(model, bank, *val_pairs, context, horizon)
    best_step = 0
    best_state = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
    curve, running = [{"step": 0, "train_loss": float("nan"), "val_loss": best_val}], []
    model.train()
    for step in range(1, steps + 1):
        pick = torch.randint(len(train_t), (batch_size,), generator=gen)
        x, yn, *_ = bank.batch(train_s[pick], train_t[pick], context, horizon)
        loss = F.mse_loss(model(x), yn)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        running.append(loss.item())
        if step % eval_every == 0 or step == steps:
            *_, val = predict(model, bank, *val_pairs, context, horizon)
            curve.append({"step": step, "train_loss": float(np.mean(running)), "val_loss": val})
            running = []
            if val < best_val:
                best_val, best_step = val, step
                best_state = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
            log(f"  step {step:5d}  train {curve[-1]['train_loss']:.4f}  val {val:.4f}")
    model.load_state_dict(best_state, strict=False)
    return {"curve": curve, "best_val_loss": best_val, "best_step": best_step}


@torch.no_grad()
def latency_ms(model: nn.Module, bank: SeriesBank, sidx, starts, context: int, horizon: int,
               batch_size: int, warmup: int, iters: int) -> float:
    """Median forward-pass latency for one batch (model already on device)."""
    reps = math.ceil(batch_size / len(starts))
    s = sidx.repeat(reps)[:batch_size]
    t = starts.repeat(reps)[:batch_size]
    x = bank.batch(s, t, context, horizon)[0]
    model.eval()
    for _ in range(warmup):
        model(x)
    sync(bank.device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        sync(bank.device)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def forecast_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    peak_true = true.max(axis=1)
    return {
        "test_mae_mw": float(np.abs(err).mean()),
        "test_rmse_mw": float(np.sqrt((err**2).mean())),
        "test_mape_pct": float((np.abs(err) / true).mean() * 100),
        "test_peak_err_pct": float((np.abs(pred.max(axis=1) - peak_true) / peak_true).mean() * 100),
    }
