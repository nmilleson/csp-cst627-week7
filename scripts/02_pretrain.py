"""Stage 2 - pretrain the shared base forecaster on the source co-ops.

Runs once with a fixed seed; every ablation arm starts from this exact checkpoint,
so rank is the only thing that differs between arms.
"""
import time

import pandas as pd
import torch

import _bootstrap  # noqa: F401
from _bootstrap import parse_args

from lora_ablation.lora import count_params
from lora_ablation.model import build_model
from lora_ablation.train import fit, forecast_metrics, peak_memory_mb, predict, reset_peak_memory
from lora_ablation.utils import (config_hash, environment_snapshot, get_device, load_config,
                                 resolve, set_seed, sha256_file, write_json)
from lora_ablation.windows import SeriesBank, load_frames


def main(argv=None):
    args = parse_args(__doc__, argv)
    cfg = load_config(args.config)
    pcfg, d, w = cfg["pretrain"], cfg["data"], cfg["window"]
    data_dir, art_dir, res_dir = resolve(cfg, "data_dir"), resolve(cfg, "artifact_dir"), resolve(cfg, "results_dir")
    device = get_device()
    set_seed(pcfg["seed"])
    print(f"device: {device}  config: {config_hash(cfg)}")

    source, _ = load_frames(data_dir)
    bank = SeriesBank(source, device)
    L, H = w["context_hours"], w["horizon_hours"]
    train_starts = bank.window_starts(d["start"], d["pretrain_train_end"], L, H, context_inside=False)
    val_starts = bank.window_starts(d["pretrain_train_end"], d["pretrain_val_end"], L, H,
                                    context_inside=False, stride=pcfg["val_stride_hours"])
    train_pairs, val_pairs = bank.all_pairs(train_starts), bank.all_pairs(val_starts)
    print(f"source series={len(bank.coop_ids)}  train windows={len(train_pairs[0]):,}  val windows={len(val_pairs[0]):,}")

    model = build_model(cfg).to(device)
    reset_peak_memory(device)
    t0 = time.perf_counter()
    result = fit(model, bank, train_pairs, val_pairs, context=L, horizon=H, steps=pcfg["steps"],
                 batch_size=pcfg["batch_size"], lr=pcfg["lr"], weight_decay=pcfg["weight_decay"],
                 warmup_steps=pcfg["warmup_steps"], eval_every=pcfg["eval_every"], seed=pcfg["seed"])
    elapsed = time.perf_counter() - t0
    pred, true, _ = predict(model, bank, *val_pairs, L, H)

    ckpt = art_dir / "base_model.pt"
    torch.save({"state_dict": model.state_dict(), "model_cfg": cfg["model"], "config_hash": config_hash(cfg)}, ckpt)
    pd.DataFrame(result["curve"]).to_csv(res_dir / "pretrain_curve.csv", index=False)
    summary = {
        "config_hash": config_hash(cfg),
        "params": count_params(model, trainable_only=False),
        "best_step": result["best_step"],
        "best_val_loss": result["best_val_loss"],
        "source_val_metrics": forecast_metrics(pred, true),
        "train_time_s": elapsed,
        "peak_mem_mb": peak_memory_mb(device),
        "checkpoint_sha256": sha256_file(ckpt),
        "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "environment": environment_snapshot(),
    }
    write_json(summary, res_dir / "pretrain_summary.json")
    print(f"base model: {summary['params']:,} params  best val {result['best_val_loss']:.4f} @ step "
          f"{result['best_step']}  source-val MAPE {summary['source_val_metrics']['test_mape_pct']:.2f}%  "
          f"time {elapsed:.0f}s")


if __name__ == "__main__":
    main()
