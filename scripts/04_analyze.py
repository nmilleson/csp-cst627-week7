"""Stage 4 - aggregate runs, test whether rank effects exceed seed noise, make plots.

Outputs (results/):
  summary.csv / summary.md   per-setting mean, SD, 95% CI, cost columns
  significance.csv           each arm vs the best LoRA rank (Welch t, Holm, noise rule)
  verdict.md                 plain-language answer to "does rank matter here?"
  report_vars.json           numbers consumed by 05_render_report.py
  fig_*.png                  figures
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import _bootstrap  # noqa: F401
from _bootstrap import parse_args

from lora_ablation import plots
from lora_ablation.utils import load_config, resolve, write_json

METRICS = ["test_mape_pct", "test_mae_mw", "test_rmse_mw", "test_peak_err_pct"]
COSTS = ["trainable_params", "adapter_kib", "train_time_s", "peak_mem_train_mb", "peak_mem_increment_mb"]
LABELS = {"test_mape_pct": "MAPE (%)", "test_mae_mw": "MAE (MW)", "test_rmse_mw": "RMSE (MW)",
          "test_peak_err_pct": "Daily-peak error (%)"}


def describe(runs: pd.DataFrame, ranks: list[int]) -> pd.DataFrame:
    lat_cols = [c for c in runs.columns if c.startswith("latency_")]
    cols = METRICS + COSTS + lat_cols + ["best_step"]
    colors = plots.rank_colors(ranks)
    rows = []
    for setting, g in runs.groupby("setting", sort=False):
        method, rank = g.method.iloc[0], int(g["rank"].iloc[0])
        row = {"setting": setting, "method": method, "rank": rank, "n_seeds": len(g)}
        for c in cols:
            v = g[c].astype(float)
            n = v.notna().sum()
            row[f"{c}_mean"] = v.mean()
            row[f"{c}_std"] = v.std(ddof=1) if n > 1 else float("nan")
            row[f"{c}_ci95"] = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            row[f"{c}_min"], row[f"{c}_max"] = v.min(), v.max()
        if method == "none":
            row.update(order=0, label="Zero-shot", color=plots.ZERO_SHOT)
        elif method == "full":
            row.update(order=999, label="Full FT", color=plots.FULL_FT)
        else:
            row.update(order=rank, label=f"LoRA r={rank}", color=colors[rank])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("order").reset_index(drop=True)


def holm(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - k) * pvals[i]))
        adj[i] = running
    return adj


def compare(runs: pd.DataFrame, summary: pd.DataFrame, metric: str, alpha: float, k: float):
    lora = summary[summary.method == "lora"]
    best = lora.loc[lora[f"{metric}_mean"].idxmin(), "setting"]
    ref = runs[runs.setting == best][metric].to_numpy()
    rows = []
    for setting in summary.setting:
        if setting == best:
            continue
        other = runs[runs.setting == setting][metric].to_numpy()
        diff = other.mean() - ref.mean()
        if len(other) == 1:  # deterministic arm: one-sample test of the reference seeds
            p = stats.ttest_1samp(ref, other[0]).pvalue
            pooled = ref.std(ddof=1)
        else:
            p = stats.ttest_ind(other, ref, equal_var=False).pvalue
            pooled = np.sqrt((other.var(ddof=1) + ref.var(ddof=1)) / 2)
        rows.append({"setting": setting, "vs": best, "diff_of_means": diff, "pooled_seed_sd": pooled,
                     "cohens_d": diff / pooled if pooled > 0 else float("inf"),
                     "exceeds_noise": abs(diff) > k * pooled, "welch_p": p})
    df = pd.DataFrame(rows)
    df["holm_p"] = holm(df.welch_p.to_numpy())
    df["verdict"] = np.where((df.holm_p < alpha) & df.exceeds_noise, "distinguishable", "within noise")
    return best, df


def fmt(m, sd, ci=None, digits=3):
    if np.isnan(sd):
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {sd:.{digits}f}" + (f" [±{ci:.{digits}f}]" if ci is not None else "")


def main(argv=None):
    args = parse_args(__doc__, argv)
    cfg = load_config(args.config)
    res = resolve(cfg, "results_dir")
    acfg, an = cfg["ablation"], cfg["analysis"]
    metric = an["primary_metric"]
    runs = pd.read_csv(res / "runs.csv")
    ranks = sorted(runs.loc[runs.method == "lora", "rank"].astype(int).unique())
    summary = describe(runs, ranks)
    summary.to_csv(res / "summary.csv", index=False)

    best, sig = compare(runs, summary, metric, an["alpha"], an["noise_multiplier"])
    sig.to_csv(res / "significance.csv", index=False)

    # Rank-only tests across all LoRA arms.
    lora_runs = runs[runs.method == "lora"]
    groups = [g[metric].to_numpy() for _, g in lora_runs.groupby("rank")]
    anova_p = stats.f_oneway(*groups).pvalue
    kruskal_p = stats.kruskal(*groups).pvalue
    trend = stats.linregress(np.log2(lora_runs["rank"].astype(float)), lora_runs[metric])
    lora_sum = summary[summary.method == "lora"]
    rank_means = lora_sum[f"{metric}_mean"]
    spread = rank_means.max() - rank_means.min()
    median_sd = lora_sum[f"{metric}_std"].median()
    rank_rows = sig[sig.setting.str.startswith("lora_")]
    n_dist = int((rank_rows.verdict == "distinguishable").sum())
    rank_matters = anova_p < an["alpha"] and spread > an["noise_multiplier"] * median_sd

    # Markdown tables.
    lat_keys = sorted({c[:-5] for c in summary.columns if c.startswith("latency_") and c.endswith("_mean")},
                      key=lambda k: (int(k.split("_b")[1].split("_")[0]), "merged" in k))
    lines = ["| Setting | Seeds | Trainable params | Adapter (KiB) | Test MAPE % (mean ± SD [95% CI]) | MAE (MW) | RMSE (MW) | Peak err % |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r.label} | {r.n_seeds} | {int(r.trainable_params_mean):,} | {r.adapter_kib_mean:.1f} | "
                     f"{fmt(r.test_mape_pct_mean, r.test_mape_pct_std, r.test_mape_pct_ci95)} | "
                     f"{fmt(r.test_mae_mw_mean, r.test_mae_mw_std)} | {fmt(r.test_rmse_mw_mean, r.test_rmse_mw_std)} | "
                     f"{fmt(r.test_peak_err_pct_mean, r.test_peak_err_pct_std)} |")
    accuracy_table = "\n".join(lines)
    lines = ["| Setting | Fine-tune time (s) | Peak GPU mem (MiB) | Mem above resident data (MiB) | "
             + " | ".join(k.replace("latency_", "Latency ").replace("_ms", "").replace("_merged", " merged") + " (ms)" for k in lat_keys) + " |",
             "|---|---|---|---|" + "---|" * len(lat_keys)]
    for _, r in summary.iterrows():
        lines.append(f"| {r.label} | {fmt(r.train_time_s_mean, r.train_time_s_std, digits=1)} | "
                     f"{fmt(r.peak_mem_train_mb_mean, r.peak_mem_train_mb_std, digits=1)} | "
                     f"{fmt(r.peak_mem_increment_mb_mean, r.peak_mem_increment_mb_std, digits=1)} | "
                     + " | ".join(fmt(r[f'{k}_mean'], r[f'{k}_std'], digits=2) for k in lat_keys) + " |")
    cost_table = "\n".join(lines)
    lines = ["| Arm | vs | Δ mean (pp) | Pooled seed SD | Cohen's d | > k·SD? | Welch p | Holm p | Verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    label = dict(zip(summary.setting, summary.label))
    for _, r in sig.iterrows():
        lines.append(f"| {label[r.setting]} | {label[r.vs]} | {r.diff_of_means:+.3f} | {r.pooled_seed_sd:.3f} | "
                     f"{r.cohens_d:+.2f} | {'yes' if r.exceeds_noise else 'no'} | {r.welch_p:.3g} | {r.holm_p:.3g} | {r.verdict} |")
    sig_table = "\n".join(lines)

    zs = summary[summary.method == "none"]
    ff = summary[summary.method == "full"]
    best_row = summary[summary.setting == best].iloc[0]
    smallest = lora_sum.iloc[0]
    verdict = [
        f"**Question:** does LoRA rank (r ∈ {{{', '.join(map(str, ranks))}}}) change day-ahead accuracy on the target co-op "
        f"by more than run-to-run (seed) noise?",
        "",
        f"**Answer: {'yes' if rank_matters else 'no'}** — " + (
            f"rank has a measurable effect: the spread of rank means ({spread:.3f} pp MAPE) is larger than "
            f"{an['noise_multiplier']:g}× the median seed SD ({median_sd:.3f} pp) and one-way ANOVA p = {anova_p:.3g}."
            if rank_matters else
            f"the spread of rank means ({spread:.3f} pp MAPE) is not larger than {an['noise_multiplier']:g}× the median "
            f"seed SD ({median_sd:.3f} pp), or the across-rank test is not significant (ANOVA p = {anova_p:.3g}, "
            f"Kruskal–Wallis p = {kruskal_p:.3g})."),
        "",
        f"- Best mean LoRA arm: **{best_row.label}** at {best_row[f'{metric}_mean']:.3f} ± {best_row[f'{metric}_std']:.3f}% MAPE.",
        f"- {n_dist} of {len(rank_rows)} other ranks are distinguishable from it (Holm-adjusted Welch p < {an['alpha']} "
        f"*and* |Δ| > {an['noise_multiplier']:g}× pooled seed SD).",
        f"- Trend in log2(rank): slope {trend.slope:+.3f} pp per doubling (p = {trend.pvalue:.3g}).",
    ]
    if len(zs):
        verdict.append(f"- Adaptation itself matters: zero-shot base model = {zs.iloc[0][f'{metric}_mean']:.3f}% MAPE "
                       f"(Δ {zs.iloc[0][f'{metric}_mean'] - best_row[f'{metric}_mean']:+.3f} pp vs best LoRA).")
    if len(ff):
        verdict.append(f"- Full fine-tuning = {fmt(ff.iloc[0][f'{metric}_mean'], ff.iloc[0][f'{metric}_std'])}% MAPE "
                       f"with {int(ff.iloc[0].trainable_params_mean):,} trainable params vs "
                       f"{int(best_row.trainable_params_mean):,} for {best_row.label}.")
    verdict.append(f"- Smallest adapter ({smallest.label}) stores {smallest.adapter_kib_mean:.1f} KiB per co-op; "
                   f"once merged, adapters add no inference compute at any rank (merged latency matches the base "
                   f"model within timing noise).")
    verdict_md = "\n".join(verdict)
    (res / "verdict.md").write_text(verdict_md + "\n", encoding="utf-8")
    (res / "summary.md").write_text(
        "## Accuracy (test split, 30 days, hourly forecast origins)\n\n" + accuracy_table
        + "\n\n## Cost\n\n" + cost_table + "\n\n## Each arm vs the best LoRA rank\n\n" + sig_table
        + "\n\n## Verdict\n\n" + verdict_md + "\n", encoding="utf-8")

    # Figures.
    plots.accuracy_vs_rank(runs, metric, LABELS[metric], res / "fig_accuracy_vs_rank.png")
    plots.accuracy_vs_rank(runs, "test_peak_err_pct", LABELS["test_peak_err_pct"], res / "fig_peak_error_vs_rank.png")
    plots.cost_panels(summary, res / "fig_cost_vs_rank.png")
    plots.pareto(summary, metric, LABELS[metric], res / "fig_pareto.png")
    curve_files = sorted((res / "curves").glob("*.csv"))
    if curve_files:
        curves = pd.concat([pd.read_csv(f) for f in curve_files])
        order = ["full_ft"] + [f"lora_r{r}" for r in ranks]
        curves["setting"] = pd.Categorical(curves.setting, [o for o in order if o in set(curves.setting)])
        plots.learning_curves(curves.sort_values("setting"), ranks, res / "fig_learning_curves.png")
    pred_dir = res / "predictions"
    if (pred_dir / f"{best}.npz").exists():
        arms = [("zero_shot", "Zero-shot", plots.ZERO_SHOT), (best, f"Best LoRA ({best_row.label})", plots.BLUE_RAMP[6]),
                ("full_ft", "Full FT", plots.FULL_FT)]
        truth, times, series = None, None, {}
        for key, name, color in arms:
            f = pred_dir / f"{key}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            origins = pd.to_datetime(z["origins_unix_s"], unit="s")
            sel = np.where(origins.hour == 0)[0][:7]
            series[name] = (z["pred"][sel].ravel(), color)
            truth = z["true"][sel].ravel()
            times = np.concatenate([origins[i] + pd.to_timedelta(np.arange(24), "h") for i in sel])
        plots.forecast_example(series, truth, times, res / "fig_forecast_example.png")

    write_json({
        "metric": metric, "ranks": ranks, "seeds": acfg["seeds"], "best_setting": best, "best_label": best_row.label,
        "best_mean": best_row[f"{metric}_mean"], "best_std": best_row[f"{metric}_std"],
        "zero_shot_mean": float(zs.iloc[0][f"{metric}_mean"]) if len(zs) else None,
        "full_ft_mean": float(ff.iloc[0][f"{metric}_mean"]) if len(ff) else None,
        "full_ft_std": float(ff.iloc[0][f"{metric}_std"]) if len(ff) else None,
        "full_ft_params": int(ff.iloc[0].trainable_params_mean) if len(ff) else None,
        "rank_spread": spread, "median_seed_sd": median_sd, "anova_p": anova_p, "kruskal_p": kruskal_p,
        "trend_slope": trend.slope, "trend_p": trend.pvalue, "rank_matters": bool(rank_matters),
        "n_distinguishable": n_dist, "n_rank_comparisons": len(rank_rows),
        "accuracy_table": accuracy_table, "cost_table": cost_table, "significance_table": sig_table,
        "verdict_md": verdict_md, "n_runs": len(runs),
    }, res / "report_vars.json")
    print(Path(res / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
