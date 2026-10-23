"""Stage 5 - render the Annex IV-style technical documentation from results.

Fills docs/templates/annex_iv_template.md with the numbers produced by stages
1-4, so the document always matches the committed results tables.
"""
import json
from datetime import datetime, timezone
from string import Template

import numpy as np
import pandas as pd
from scipy import stats

import _bootstrap  # noqa: F401
from _bootstrap import ROOT, parse_args

from lora_ablation.utils import config_hash, load_config, resolve


def recommendation(summary: pd.DataFrame, sig: pd.DataFrame, rv: dict) -> str:
    """Parsimony rule: the smallest rank not distinguishable from the best rank."""
    lora = summary[summary.method == "lora"].sort_values("rank")
    within = set(sig.loc[sig.verdict == "within noise", "setting"]) | {rv["best_setting"]}
    pick = lora[lora.setting.isin(within)].iloc[0]
    text = (f"Adopt **{pick.label}** as the default adapter rank. It is the smallest rank whose test MAPE "
            f"({pick[rv['metric'] + '_mean']:.3f} ± {pick[rv['metric'] + '_std']:.3f}%) is not distinguishable from "
            f"the best arm ({rv['best_label']}), and it stores {pick.adapter_kib_mean:.1f} KiB per co-op.")
    if pick.setting != rv["best_setting"]:
        text += (f" Choosing {rv['best_label']} instead would add "
                 f"{int(lora[lora.setting == rv['best_setting']].trainable_params_mean.iloc[0] - pick.trainable_params_mean):,} "
                 f"parameters per co-op for a difference that seed noise cannot rule out.")
    ff = sig[sig.setting == "full_ft"]
    if len(ff):
        diff = ff.diff_of_means.iloc[0]
        text += (f" Full fine-tuning is {'worse' if diff > 0 else 'better'} than the best LoRA arm by {abs(diff):.3f} pp "
                 f"({ff.verdict.iloc[0]}) and needs {rv['full_ft_params']:,} trainable parameters per co-op.")
    return text + " Approval is conditional on replicating this result on real demand data (§6, item 1)."


def main(argv=None):
    args = parse_args(__doc__, argv)
    cfg = load_config(args.config)
    res, data_dir = resolve(cfg, "results_dir"), resolve(cfg, "data_dir")
    rv = json.loads((res / "report_vars.json").read_text(encoding="utf-8"))
    pre = json.loads((res / "pretrain_summary.json").read_text())
    meta = json.loads((res / "ablation_meta.json").read_text())
    manifest = json.loads((data_dir / "manifest.json").read_text())
    summary = pd.read_csv(res / "summary.csv")
    sig = pd.read_csv(res / "significance.csv")
    env, d, a = meta["environment"], cfg["data"], cfg["ablation"]

    n = len(a["seeds"])
    df = 2 * n - 2
    mde_sd = (stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df)) * np.sqrt(2 / n)
    fmt_split = lambda s: f"{s[0][:10]} → {s[1][:10]}"  # noqa: E731

    values = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git_commit": env["git_commit"], "config_hash": config_hash(cfg),
        "python": env["python"], "torch": env["torch"],
        "gpu": f"{env['gpu']} ({env['gpu_total_memory_mb']} MiB), CUDA {env['cuda']}" if env["gpu"] else "CPU only",
        "n_layers": cfg["model"]["n_layers"], "d_model": cfg["model"]["d_model"], "n_heads": cfg["model"]["n_heads"],
        "base_params": f"{pre['params']:,}", "pretrain_seed": cfg["pretrain"]["seed"],
        "n_source": d["n_source_coops"], "pretrain_val_end": d["pretrain_val_end"],
        "pretrain_mape": f"{pre['source_val_metrics']['test_mape_pct']:.2f}", "pretrain_time": f"{pre['train_time_s']:.0f}",
        "ranks": ", ".join(map(str, a["ranks"])), "n_seeds": n, "n_runs": rv["n_runs"],
        "steps": a["steps"], "batch_size": a["batch_size"], "lr": a["lr"], "full_lr": a["full_finetune_lr"],
        "eval_every": a["eval_every"],
        "generator_sha": manifest["generator_sha256"][:16], "data_seed": manifest["seed"],
        "source_sha": manifest["files"]["source_coops.csv.gz"]["sha256"][:16],
        "target_sha": manifest["files"]["target_coop.csv.gz"]["sha256"][:16],
        "data_start": d["start"], "data_end": d["end"],
        "train_split": fmt_split(meta["splits"]["train"]), "val_split": fmt_split(meta["splits"]["val"]),
        "test_split": fmt_split(meta["splits"]["test"]),
        "accuracy_table": rv["accuracy_table"], "cost_table": rv["cost_table"],
        "significance_table": rv["significance_table"], "verdict_md": rv["verdict_md"],
        "recommendation": recommendation(summary, sig, rv),
        "mde_sd": f"{mde_sd:.1f}", "mde_pp": f"{mde_sd * rv['median_seed_sd']:.2f}",
    }
    template = Template((ROOT / "docs/templates/annex_iv_template.md").read_text(encoding="utf-8"))
    out = ROOT / cfg["paths"]["report"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template.substitute(values), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}  (~{len(out.read_text(encoding='utf-8').split()):,} words)")


if __name__ == "__main__":
    main()
