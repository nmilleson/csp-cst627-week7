"""Stage 3 - the controlled ablation: LoRA rank x seed on the target co-op.

Controlled: every arm starts from the same base checkpoint, sees the same 90-day
training split, the same batch order for a given seed, the same step budget,
learning rate and alpha/r scale. Only the rank changes. Zero-shot (no
adaptation) and full fine-tuning are run as brackets.

Resumable: each finished run is appended to results/runs.csv immediately and is
skipped on re-launch, so a Colab disconnect loses at most one run. --fresh
discards previous results.
"""
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401
from _bootstrap import parse_args

from lora_ablation.lora import apply_lora, count_params, merge_lora
from lora_ablation.model import build_model
from lora_ablation.train import (fit, forecast_metrics, latency_ms, peak_memory_mb, predict,
                                 reset_peak_memory, sync)
from lora_ablation.utils import (config_hash, environment_snapshot, get_device, load_config,
                                 resolve, set_seed, write_json)
from lora_ablation.windows import SeriesBank, load_frames


def settings_for(acfg):
    out = []
    if acfg["include_zero_shot"]:
        out.append(("zero_shot", "none", 0))
    if acfg["include_full_finetune"]:
        out.append(("full_ft", "full", 0))
    out += [(f"lora_r{r}", "lora", r) for r in acfg["ranks"]]
    return out


def main(argv=None):
    args = parse_args(__doc__, argv, extra=lambda p: p.add_argument("--fresh", action="store_true"))
    cfg = load_config(args.config)
    acfg, d, w = cfg["ablation"], cfg["data"], cfg["window"]
    L, H = w["context_hours"], w["horizon_hours"]
    data_dir, art_dir, res_dir = resolve(cfg, "data_dir"), resolve(cfg, "artifact_dir"), resolve(cfg, "results_dir")
    runs_path, curve_dir, pred_dir = res_dir / "runs.csv", res_dir / "curves", res_dir / "predictions"
    if args.fresh:
        runs_path.unlink(missing_ok=True)
        shutil.rmtree(curve_dir, ignore_errors=True)
        shutil.rmtree(pred_dir, ignore_errors=True)
    curve_dir.mkdir(exist_ok=True)
    pred_dir.mkdir(exist_ok=True)
    device = get_device()

    ckpt = torch.load(art_dir / "base_model.pt", map_location="cpu")
    _, target = load_frames(data_dir)
    bank = SeriesBank(target, device)
    t_train = pd.Timestamp(d["target_train_start"])
    t_val = t_train + pd.Timedelta(days=d["target_train_days"])
    t_test = t_val + pd.Timedelta(days=d["target_val_days"])
    t_end = t_test + pd.Timedelta(days=d["target_test_days"])
    stride = w["eval_stride_hours"]
    splits = {
        "train": bank.all_pairs(bank.window_starts(t_train, t_val, L, H, context_inside=True)),
        "val": bank.all_pairs(bank.window_starts(t_val, t_test, L, H, context_inside=False, stride=stride)),
        "test": bank.all_pairs(bank.window_starts(t_test, t_end, L, H, context_inside=False, stride=stride)),
    }
    print(f"device {device}  config {config_hash(cfg)}  target windows: "
          + "  ".join(f"{k}={len(v[0])}" for k, v in splits.items()))
    write_json({"config_hash": config_hash(cfg), "config": cfg, "environment": environment_snapshot(),
                "splits": {"train": [str(t_train), str(t_val)], "val": [str(t_val), str(t_test)],
                           "test": [str(t_test), str(t_end)]}},
               res_dir / "ablation_meta.json")

    def fresh_base():
        m = build_model(cfg)
        m.load_state_dict(ckpt["state_dict"])
        return m.to(device)

    # Warm-up: absorb CUDA context / kernel-selection overhead so run 1 is not penalised.
    warm = fresh_base()
    x, yn, *_ = bank.batch(splits["train"][0][:32], splits["train"][1][:32], L, H)
    for _ in range(5):
        torch.nn.functional.mse_loss(warm(x), yn).backward()
    sync(device)
    del warm

    done = set()
    if runs_path.exists():
        prev = pd.read_csv(runs_path)
        done = set(zip(prev.setting, prev.seed))
        print(f"resuming: {len(done)} runs already in {runs_path.name}")

    seeds = acfg["seeds"]
    plan = [(seed, s) for seed in seeds for s in settings_for(acfg)
            if not (s[1] == "none" and seed != seeds[0])]  # zero-shot is deterministic: run once
    t_all = time.perf_counter()
    for i, (seed, (setting, method, rank)) in enumerate(plan, 1):
        if (setting, seed) in done:
            continue
        print(f"[{i}/{len(plan)}] {setting}  seed={seed}")
        set_seed(seed)
        reset_peak_memory(device)
        baseline_mb = torch.cuda.memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
        model = fresh_base()
        lr = acfg["lr"]
        if method == "lora":
            apply_lora(model, rank, acfg["alpha_over_rank"] * rank, acfg["target_modules"], acfg["lora_dropout"])
        elif method == "full":
            lr = acfg["full_finetune_lr"]
        else:
            model.requires_grad_(False)
        trainable = count_params(model)

        train_time, fit_result = 0.0, {"curve": [], "best_val_loss": float("nan"), "best_step": 0}
        if method != "none":
            sync(device)
            t0 = time.perf_counter()
            fit_result = fit(model, bank, splits["train"], splits["val"], context=L, horizon=H,
                             steps=acfg["steps"], batch_size=acfg["batch_size"], lr=lr,
                             weight_decay=acfg["weight_decay"], warmup_steps=acfg["warmup_steps"],
                             eval_every=acfg["eval_every"], seed=seed, log=lambda *_: None)
            sync(device)
            train_time = time.perf_counter() - t0
        peak_train = peak_memory_mb(device) if method != "none" else float("nan")
        *_, val_loss = predict(model, bank, *splits["val"], L, H)
        pred, true, _ = predict(model, bank, *splits["test"], L, H)
        metrics = forecast_metrics(pred, true)

        lat = {}
        for merged in (False, True):
            if merged:
                merge_lora(model)
            for bs in acfg["latency"]["batch_sizes"]:
                key = f"latency_b{bs}{'_merged' if merged else ''}_ms"
                lat[key] = latency_ms(model, bank, *splits["test"], L, H, bs,
                                      acfg["latency"]["warmup_iters"], acfg["latency"]["timed_iters"])

        row = {
            "setting": setting, "method": method, "rank": rank, "seed": seed,
            "trainable_params": trainable, "adapter_kib": trainable * 4 / 1024,
            "best_step": fit_result["best_step"], "val_loss": val_loss, **metrics,
            "train_time_s": train_time, "peak_mem_train_mb": peak_train,
            "peak_mem_increment_mb": peak_train - baseline_mb, **lat,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        pd.DataFrame([row]).to_csv(runs_path, mode="a", header=not runs_path.exists(), index=False)
        if fit_result["curve"]:
            pd.DataFrame(fit_result["curve"]).assign(setting=setting, seed=seed).to_csv(
                curve_dir / f"{setting}_seed{seed}.csv", index=False)
        if seed == seeds[0]:
            origins = bank.timestamps[splits["test"][1].numpy() + L]
            np.savez_compressed(pred_dir / f"{setting}.npz", pred=pred, true=true,
                                origins_unix_s=origins.values.astype("datetime64[s]").astype("int64"))
        first_lat = lat["latency_b%d_ms" % acfg["latency"]["batch_sizes"][0]]
        print(f"    MAPE {metrics['test_mape_pct']:.3f}%  MAE {metrics['test_mae_mw']:.3f} MW  "
              f"params {trainable:,}  train {train_time:.1f}s  peak {peak_train:.0f} MiB  "
              f"lat(b1) {first_lat:.2f} ms")
        del model
    print(f"ablation finished in {(time.perf_counter() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
