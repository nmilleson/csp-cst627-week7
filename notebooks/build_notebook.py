"""Generates colab_lora_rank_ablation.ipynb. The notebook only calls the scripts
and displays their outputs; all logic lives in scripts/ and lora_ablation/.

    python notebooks/build_notebook.py
"""
import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Cobalt Grid: LoRA-rank ablation for day-ahead demand forecasting

Runs the whole study in one free-tier Colab GPU session (~30 min on a T4). Every cell calls a script from the
repository; open the `.py` files to see the logic.

**Before you start:** *Runtime → Change runtime type → T4 GPU*. Then *Runtime → Run all*.
If the session disconnects during stage 3, re-run from the top; finished runs are kept and skipped.
"""),
    code("""
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"""),
    code("""
import os
REPO_URL = "https://github.com/nmilleson/csp-cst627-week7.git"
if not os.path.exists("csp-cst627-week7"):
    !git clone -q $REPO_URL
%cd /content/csp-cst627-week7
!git log -1 --format="commit %h  %s"
!pip install -q -r requirements-colab.txt
!python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
"""),
    code("""
CONFIG = "configs/ablation.yaml"      # "configs/smoke.yaml" = 2-minute dry run of the same pipeline
RESULTS = "results_smoke" if "smoke" in CONFIG else "results"

from IPython.display import Image, Markdown, display
def show(name, width=950):
    display(Image(filename=f"{RESULTS}/{name}", width=width))
"""),
    md("""
## 1 · Data: synthetic co-op demand (seeded, hashed)
24 source co-ops (US + Central EU) for pretraining; one Nordic target co-op with heat pumps and EV charging.
A rerun prints `MATCH` when the files are byte-identical to the recorded manifest.
"""),
    code("""
!python scripts/01_make_data.py --config $CONFIG
show("fig_data_overview.png")
"""),
    md("""
## 2 · Pretrain the shared base model (fixed seed, run once)
"""),
    code("""
!python scripts/02_pretrain.py --config $CONFIG
"""),
    md("""
## 3 · Ablation: LoRA rank × seed (+ zero-shot and full fine-tune brackets)
One line per run: accuracy, wall-clock, peak GPU memory and latency. Resumable.
"""),
    code("""
!python scripts/03_run_ablation.py --config $CONFIG
"""),
    md("""
## 4 · Analysis: means, seed spread, and whether rank beats noise
"""),
    code("""
!python scripts/04_analyze.py --config $CONFIG > /dev/null
display(Markdown(open(f"{RESULTS}/summary.md", encoding="utf-8").read()))
"""),
    code("""
for fig in ["fig_accuracy_vs_rank.png", "fig_pareto.png", "fig_cost_vs_rank.png",
            "fig_learning_curves.png", "fig_peak_error_vs_rank.png", "fig_forecast_example.png"]:
    show(fig)
"""),
    md("""
## 5 · Render the Annex IV-style technical documentation from these results
"""),
    code("""
!python scripts/05_render_report.py --config $CONFIG
"""),
    md("""
## 6 · Bring the results back to the repository
Downloads a zip of `results/` and the rendered document. Unzip it at the repository root locally and commit.
"""),
    code("""
!zip -qr cobalt_results.zip $RESULTS docs/Annex_IV_Technical_Documentation.md data/manifest.json data/data_card.json
from google.colab import files
files.download("cobalt_results.zip")
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = Path(__file__).with_name("colab_lora_rank_ablation.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print(f"wrote {out}")
