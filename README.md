# Does LoRA rank matter for day-ahead demand forecasting?

A controlled, reproducible ablation for **Cobalt Grid**, which sells short-horizon electricity-demand forecasts to
rural utility co-ops in the EU and US. Cobalt Grid adapts one shared forecasting model to each new co-op with a small
LoRA adapter. This study answers one release question: **which adapter rank should we ship, and are the differences
between ranks larger than run-to-run noise?**

| | |
|---|---|
| **Factor** | LoRA rank r ∈ {1, 2, 4, 8, 16, 32}, with α = 2r (constant scale α/r) |
| **Replication** | 5 seeds per rank, plus two brackets: zero-shot (no adaptation) and full fine-tuning |
| **Task** | Forecast the next 24 h of territory demand (MW) from 168 h of history, temperature and calendar |
| **Data** | Synthetic, seeded and hashed: 24 source co-ops for pretraining, 1 Nordic target co-op (heat pumps + EV) with 90 days of history |
| **Reported** | MAPE / MAE / RMSE / daily-peak error as mean ± seed SD with 95% CIs, a noise test, wall-clock, peak GPU memory, latency, adapter size |
| **Budget** | One free-tier Colab T4 session, about 30 minutes end to end |

The results summary is in [results/summary.md](results/summary.md) and the plain-language answer is in
[results/verdict.md](results/verdict.md). Compliance documentation is in
[docs/Annex_IV_Technical_Documentation.md](docs/Annex_IV_Technical_Documentation.md).

---

## 1. Reproduce it

### Option A: Google Colab (the reference environment)

1. Open [`notebooks/colab_lora_rank_ablation.ipynb`](notebooks/colab_lora_rank_ablation.ipynb) in Colab
   (*File → Open notebook → GitHub* and paste this repository's URL).
2. *Runtime → Change runtime type → **T4 GPU***.
3. *Runtime → **Run all***. The notebook clones the repo, installs the pinned dependencies and runs the five stages. It
   then displays the tables and figures and downloads `cobalt_results.zip`.
4. To store the results with the code, unzip `cobalt_results.zip` at the repository root and commit.

If Colab disconnects during stage 3, run all cells again. Finished runs are read from `results/runs.csv` and skipped.

### Option B: any machine with Python 3.11–3.12

```bash
git clone https://github.com/nmilleson/csp-cst627-week7.git
cd csp-cst627-week7
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/run_all.py --config configs/smoke.yaml     # ~2 min dry run on CPU; writes *_smoke/ only
python scripts/run_all.py --config configs/ablation.yaml  # full study (GPU strongly recommended)
```

On CPU the full study takes many hours: about 2 s per step for the full-size model. The smoke config runs the
identical code path with a tiny model and finishes in about 2 minutes.

### Running stages individually

Every stage reads the same YAML config and can be rerun on its own:

| Stage | Command | Produces |
|---|---|---|
| 1 Data | `python scripts/01_make_data.py --config configs/ablation.yaml` | `data/*.csv.gz`, `data/manifest.json` (SHA-256), `data/data_card.json`, `results/fig_data_overview.png` |
| 2 Pretrain | `python scripts/02_pretrain.py ...` | `artifacts/base_model.pt`, `results/pretrain_summary.json`, `results/pretrain_curve.csv` |
| 3 Ablation | `python scripts/03_run_ablation.py ... [--fresh]` | `results/runs.csv` (one row per run), `results/curves/`, `results/predictions/`, `results/ablation_meta.json` |
| 4 Analysis | `python scripts/04_analyze.py ...` | `results/summary.{csv,md}`, `results/significance.csv`, `results/verdict.md`, `results/fig_*.png` |
| 5 Report | `python scripts/05_render_report.py ...` | `docs/Annex_IV_Technical_Documentation.md` |

### Checking that you reproduced it

- **Data:** stage 1 prints `Reproducibility check vs previous manifest: MATCH` when the regenerated files are
  byte-identical to the committed [data/manifest.json](data/manifest.json).
- **Results:** compare your `results/summary.md` with the committed one. On the same GPU type, metrics agree to about 3
  decimal places. Some CUDA attention kernels are not bit-deterministic, so the last digits can differ. Timing and
  memory depend on the GPU and on co-tenancy.

---

## 2. Study design

```
 24 source co-ops (2022-01 → 2024-07)          Nordic target co-op (2024-08 → 2024-12)
 ┌───────────────────────────────┐             ┌───────────────────────────────────────────┐
 │ pretrain base transformer     │  frozen     │ train 90 d │ val 14 d │ test 30 d           │
 │ (seed 0, 556k params)         ├────────────▶│ LoRA r ∈ {1..32} × seeds {0..4}           │
 └───────────────────────────────┘  weights    │ + zero-shot, + full fine-tune brackets    │
                                               └───────────────────────────────────────────┘
```

**What changes:** only the LoRA rank r, applied to `q_proj, k_proj, v_proj, out_proj, ff1, ff2` in every block.

**What is held constant** (the controls):

- the base checkpoint;
- the data splits;
- the batch order for a given seed (a seeded sampler);
- 800 steps × batch 128, AdamW at lr 2e-3, 50-step warm-up then cosine decay;
- α/r = 2;
- checkpoint selection by validation loss, evaluated every 50 steps.

The seed controls adapter initialisation and batch order.

**Why α = 2r?** With the common fixed-α convention, the update scale α/r shrinks as r grows. That would confound
"more capacity" with "smaller steps". Holding α/r constant isolates capacity.

**Why synthetic data?** It gives a known, documented distribution shift: the target co-op has a 2× steeper heating
response from heat pumps and a late-evening EV charging peak, neither of which appears in the source population. It
also reruns byte-identically with no download or licence dependency. The generator is about 200 readable lines in
[lora_ablation/data.py](lora_ablation/data.py). Each co-op has its own simulated weather, a simulated day-ahead
temperature-forecast error, thermal inertia, irrigation, holidays, load growth and autocorrelated noise.

**How "exceeds noise" is decided** (fixed in `configs/ablation.yaml` before the run):

1. Each run's test metric is one observation. The seed is the unit of replication; overlapping forecast windows are
   never counted as independent samples.
2. Each arm is compared with the best LoRA rank by Welch's t-test, with a Holm correction across comparisons.
3. An arm is **distinguishable** only if Holm p < 0.05 **and** |difference of means| > 2 × pooled seed SD.
4. Rank as a whole is tested with a one-way ANOVA (and Kruskal–Wallis) across ranks, plus a log2(rank) trend test.

**Costs logged per run:**

- fine-tune wall-clock (CUDA-synchronised);
- peak GPU memory, both total and above the resident dataset;
- trainable parameters and adapter size;
- median forward latency at batch 1 and batch 256, measured **before and after merging** the adapter into the base
  weights.

## 3. Results

These figures are generated by stage 4 (see [results/summary.md](results/summary.md) for all tables).

| | |
|---|---|
| ![accuracy](results/fig_accuracy_vs_rank.png) | ![pareto](results/fig_pareto.png) |
| ![cost](results/fig_cost_vs_rank.png) | ![curves](results/fig_learning_curves.png) |

![forecast](results/fig_forecast_example.png)
![data](results/fig_data_overview.png)

## 4. What this study cannot conclude

This is a short version; the full list is §6 of the Annex IV document.

- **No real-world accuracy claim.** The evidence covers one synthetic target co-op and one 30-day autumn test window.
- **One base model.** Variance between pretraining seeds is not measured.
- **One training recipe.** Higher ranks might prefer a different learning rate. The question answered is "rank at a
  fixed recipe", not "each rank at its best".
- **Limited power.** With 5 seeds, only effects of about 2 seed-SDs or more are detectable. "Within noise" means "no
  effect large enough to see", not "no effect".
- **Small model.** The model has 556k parameters, so memory and latency trends will not transfer to LLM-scale LoRA.

## 5. Repository layout

```
configs/ablation.yaml          the study (every seed and hyper-parameter)
configs/smoke.yaml             tiny dry run; inherits ablation.yaml
lora_ablation/                 library code
  data.py                      seeded synthetic co-op generator
  windows.py                   GPU-resident series and on-the-fly windowing
  model.py                     transformer forecaster with named linear layers
  lora.py                      LoRALinear, apply_lora, merge_lora (no external PEFT dependency)
  train.py                     training loop, evaluation, latency and memory measurement
  plots.py                     figures
  utils.py                     config, seeding, hashing, environment snapshot
scripts/01..05_*.py, run_all.py   entry points (all accept --config)
notebooks/                     Colab notebook (+ build_notebook.py, which generates it)
docs/                          Annex IV-style documentation, its template, video walkthrough outline
results/                       committed outputs of the reference run
data/manifest.json, data_card.json   provenance; the CSVs themselves are regenerated
```

## 6. Environment

- `requirements.txt` pins everything, including `torch==2.14.0`.
- `requirements-colab.txt` pins everything **except torch**. Colab's preinstalled CUDA build is used so the CUDA build
  matches the driver.
- The torch, CUDA, GPU and library versions actually used are written to `results/ablation_meta.json` and into the
  Annex IV document.
- Tested locally on Python 3.11.9 / Windows 11 (CPU, smoke config). The full study targets a Colab T4 GPU.
