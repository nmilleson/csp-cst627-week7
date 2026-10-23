"""Matplotlib figures. Static PNGs, light theme, one y-axis per panel.

Colour roles: LoRA ranks are ordinal, so they use one sequential blue ramp
(light = low rank, dark = high rank). The two brackets get fixed identities:
zero-shot = neutral grey, full fine-tune = orange. Observed truth = near-black.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ZERO_SHOT, FULL_FT, TRUTH = "#8a8984", "#eb6834", "#0b0b0b"
SOURCE, TARGET = "#2a78d6", "#eb6834"
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2,
    "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 150,
})


def rank_colors(ranks: list[int]) -> dict[int, str]:
    idx = np.linspace(0, len(BLUE_RAMP) - 1, len(ranks)).round().astype(int)
    return {r: BLUE_RAMP[i] for r, i in zip(sorted(ranks), idx)}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def data_overview(source: pd.DataFrame, target: pd.DataFrame, path, week_start: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    t0 = pd.Timestamp(week_start)
    sel = lambda d: d[(d.timestamp >= t0) & (d.timestamp < t0 + pd.Timedelta(days=14))]  # noqa: E731
    src_id = source.coop_id.iloc[0]
    s, t = sel(source[source.coop_id == src_id]), sel(target)
    ax.plot(s.timestamp, s.demand_mw / s.demand_mw.mean(), color=SOURCE, label=f"Source ({src_id})")
    ax.plot(t.timestamp, t.demand_mw / t.demand_mw.mean(), color=TARGET, label="Target (Nordic, heat pump + EV)")
    ax.set_title("Two weeks of demand, scaled to each co-op's mean")
    ax.set_ylabel("Demand / mean")
    ax.legend(loc="upper left")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1]
    daily = lambda d: d.set_index("timestamp").resample("D").agg(  # noqa: E731
        {"demand_mw": "mean", "temperature_c": "mean"})
    for cid, g in source.groupby("coop_id"):
        dd = daily(g)
        ax.scatter(dd.temperature_c, dd.demand_mw / dd.demand_mw.mean(), s=3, color=SOURCE,
                   alpha=0.08, linewidths=0, label=f"Source co-ops ({source.coop_id.nunique()})" if cid == src_id else None)
    dd = daily(target)
    ax.scatter(dd.temperature_c, dd.demand_mw / dd.demand_mw.mean(), s=8, color=TARGET,
               alpha=0.7, linewidths=0, label="Target co-op")
    ax.set_title("Daily demand vs temperature: the shift LoRA must learn")
    ax.set_xlabel("Daily mean temperature (°C)")
    ax.set_ylabel("Demand / mean")
    leg = ax.legend(loc="upper right", markerscale=3)
    for h in leg.legend_handles:
        h.set_alpha(1)
    _save(fig, path)


def accuracy_vs_rank(runs: pd.DataFrame, metric: str, label: str, path) -> None:
    lora = runs[runs.method == "lora"]
    ranks = sorted(lora["rank"].astype(int).unique())
    colors = rank_colors(ranks)
    pos = {r: i for i, r in enumerate(ranks)}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, color, text in [("zero_shot", ZERO_SHOT, "Zero-shot base"), ("full_ft", FULL_FT, "Full fine-tune")]:
        g = runs[runs.setting == name][metric]
        if len(g):
            ax.axhline(g.mean(), color=color, linestyle="--", linewidth=1.5)
            if len(g) > 1:
                ax.axhspan(g.mean() - g.std(), g.mean() + g.std(), color=color, alpha=0.12, linewidth=0)
            ax.text(len(ranks) + 0.55, g.mean(), text, color=INK_2, va="bottom", ha="right", fontsize=8)
    rng = np.random.default_rng(0)
    for r in ranks:
        v = lora[lora["rank"] == r][metric].to_numpy()
        x = pos[r] + rng.uniform(-0.12, 0.12, len(v))
        ax.scatter(x, v, s=36, color=colors[r], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if len(v) > 1:
            ax.errorbar(pos[r] + 0.28, v.mean(), yerr=v.std(ddof=1), fmt="o", color=INK,
                        markersize=5, capsize=3, linewidth=1.2, zorder=4)
    ax.set_xticks(range(len(ranks)), [f"r = {r}" for r in ranks])
    ax.set_xlim(-0.5, len(ranks) + 0.6)
    ax.set_xlabel("LoRA rank (dots = individual seeds; black = mean ± 1 SD)")
    ax.set_ylabel(label)
    ax.set_title(f"Test {label} by LoRA rank")
    _save(fig, path)


def cost_panels(summary: pd.DataFrame, path) -> None:
    """2x2 small multiples; each panel has its own single y-axis."""
    order = summary.sort_values("order")
    labels = order.label.tolist()
    colors = order.color.tolist()
    panels = [
        ("trainable_params_mean", "Trainable parameters (log scale)", True),
        ("train_time_s_mean", "Fine-tune wall-clock (s)", False),
        ("peak_mem_train_mb_mean", "Peak GPU memory, training (MiB)", False),
        ("latency_b1_ms_mean", "Inference latency, batch 1 (ms, unmerged)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    for ax, (col, title, log) in zip(axes.flat, panels):
        vals = order[col].to_numpy(dtype=float)
        sd_col = col.replace("_mean", "_std")
        err = order[sd_col].to_numpy(dtype=float) if sd_col in order else None
        if np.all(np.isnan(vals)):
            ax.text(0.5, 0.5, "n/a on CPU", ha="center", va="center", color=INK_2, transform=ax.transAxes)
        else:
            ax.bar(range(len(vals)), vals, color=colors, width=0.7, edgecolor=SURFACE, linewidth=2,
                   yerr=None if log else err, ecolor=INK, capsize=3)
        if log:
            ax.set_yscale("log")
            bottom = ax.get_ylim()[0]
            for i, v in enumerate(vals):
                ax.annotate(f"{int(v):,}", (i, max(v, bottom)), textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=8, color=INK_2)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    if "latency_b1_merged_ms_mean" in order:
        merged = order["latency_b1_merged_ms_mean"].to_numpy(dtype=float)
        axes[1, 1].scatter(range(len(merged)), merged, marker="_", s=300, color=INK, zorder=4,
                           label="after merging adapters")
        axes[1, 1].legend(loc="lower right")
    _save(fig, path)


def pareto(summary: pd.DataFrame, metric: str, label: str, path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x_min = summary.loc[summary.trainable_params_mean > 0, "trainable_params_mean"].min()
    for _, row in summary.sort_values("order").iterrows():
        x, y, sd = row["trainable_params_mean"], row[f"{metric}_mean"], row[f"{metric}_std"]
        if x == 0:
            ax.axhline(y, color=row.color, linestyle="--", linewidth=1.5)
            ax.text(x_min, y, "zero-shot base (0 trainable params)", color=INK_2, va="bottom", fontsize=8)
            continue
        ax.errorbar(x, y, yerr=0 if np.isnan(sd) else sd, fmt="o", color=row.color, markersize=8,
                    capsize=3, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(row.label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel(label)
    ax.set_title(f"Accuracy vs adapter size (mean ± 1 SD over seeds)")
    _save(fig, path)


def learning_curves(curves: pd.DataFrame, ranks: list[int], path) -> None:
    colors = rank_colors(ranks)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for setting, g in curves.groupby("setting", sort=False):
        m = g.groupby("step").val_loss.mean()
        if setting == "full_ft":
            ax.plot(m.index, m.values, color=FULL_FT, label="Full fine-tune")
        else:
            r = int(setting.split("_r")[1])
            ax.plot(m.index, m.values, color=colors[r], label=f"LoRA r = {r}")
    ax.set_xlabel("Fine-tuning step")
    ax.set_ylabel("Validation loss (normalised MSE)")
    ax.set_title("Validation loss during adaptation (mean over seeds)")
    ax.legend(ncol=2)
    _save(fig, path)


def forecast_example(series: dict[str, tuple[np.ndarray, str]], truth: np.ndarray, times, path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(times, truth, color=TRUTH, linewidth=2.5, label="Observed demand")
    for name, (values, color) in series.items():
        ax.plot(times, values, color=color, label=name)
    ax.set_ylabel("Demand (MW)")
    ax.set_title("Day-ahead forecasts on the held-out test period (seed 0, forecasts issued at midnight)")
    ax.legend(ncol=4, loc="upper left")
    _save(fig, path)
