# Technical Documentation Summary (EU AI Act Annex IV-style)

## Cobalt Grid Short-Horizon Demand Forecaster: LoRA-rank adaptation study

| Field | Value |
|---|---|
| Provider | Cobalt Grid (ML Research) |
| Document owner | ML Research Engineer (study author) |
| Reviewers | Dr. Ingrid Solberg (Head of Research): release approval · Marcus Adeyemi (Compliance): conformity file |
| Document status | Draft summary for the conformity file, rendered 2026-10-24 16:42 UTC. Results produced by repository commit `b564b72`, config hash `6e2e71d9b599` |
| Evidence | Repository `README.md`, `configs/ablation.yaml`, `results/` (tables, plots, run logs), `data/manifest.json` |

**Scope of this summary.** This document summarises the technical documentation for one release decision: *which LoRA rank Cobalt Grid should use when adapting its shared day-ahead demand-forecasting model to a new co-op*. It follows the structure of EU AI Act Annex IV (referenced as "AIA IV §n") and cross-references the NIST AI Risk Management Framework 1.0 ("RMF") and ISO/IEC 42001:2023 Annex A ("42001 A.x"). The evidence comes from a controlled study on **synthetic data**. It supports the engineering decision about rank. It does **not** establish field accuracy on any real co-op.

**Regulatory status (for Compliance to confirm).** Annex III point 2 lists AI systems used as *safety components* in the management and operation of electricity supply. The forecaster informs co-op planning (power purchasing, peak shaving, maintenance windows), and a human planner reviews its output. It does not directly actuate grid equipment. Cobalt Grid may therefore be able to rely on the Art. 6(3) "preparatory task" derogation, which requires a documented assessment (Art. 6(4)). Until Compliance records that decision, this file is maintained to the high-risk standard (Arts. 9–15) as a conservative default. A customer integration that feeds forecasts into automated load control without review would change the classification and requires re-assessment.

---

## 1. General description of the system (AIA IV §1)

**Intended purpose.** Produce hourly electricity-demand forecasts for the next 24 hours for a rural distribution co-op's whole service territory (aggregate MW). The forecast is refreshed every hour. Users are co-op operations and power-supply planners. Outputs feed day-ahead purchasing, peak-shaving decisions and demand-charge management. Every output is reviewed by a qualified planner before it is acted on.

**Out of scope / prohibited uses.** Protection or real-time balancing control. Forecasting individual households or meters. Billing. Any automated curtailment without human review. Territories with fewer than 90 days of hourly history.

**Inputs → outputs.** Input: the previous 168 h of aggregate demand; observed temperature for the history window; a day-ahead temperature forecast for the horizon; calendar features and the regional holiday calendar. Output: 24 hourly demand values (MW). No personal data is processed. The inputs are territory-level aggregates.

**Deployment form.** One shared, frozen *base model* is pretrained on many co-ops. Each customer co-op receives a small *LoRA adapter* trained on its own history. Adapters can be merged into the base weights at deployment, so inference costs the same as the base model whatever the rank. Software stack: Python 3.13.16, PyTorch 2.11.0+cu130. The study ran on: Tesla T4 (14913 MiB), CUDA 13.0.

## 2. Development process and design (AIA IV §2(a)–(c))

**Base model.** A pre-LayerNorm transformer encoder (4 layers, d_model 128, 4 heads, 556,289 parameters). Each hourly timestep is one token. History tokens carry instance-normalised demand. Horizon tokens carry covariates and a mask flag instead of demand. A linear head reads the 24 horizon tokens. It was pretrained once (seed 0) on 24 source co-ops (US and Central-EU) with data up to 2024-08-01. Source-validation MAPE was 1.59%, with 453 s of training.

**Adaptation (the factor under study).** LoRA (Hu et al., 2021) is applied to every attention projection (q, k, v, out) and both feed-forward layers in every block. The update is W + (α/r)·BA, with B initialised to zero. The base model, embeddings, LayerNorms and head stay frozen. **Design choice:** α = 2r, so the scale α/r = 2 is the same at every rank. Changing r therefore changes adapter capacity without silently changing the effective step size.

**Study design (controlled ablation).** Ranks r ∈ {1, 2, 4, 8, 16, 32} × 5 seeds. Two bracketing arms are added: *zero-shot* (base model, no adaptation, deterministic, 1 run) and *full fine-tuning* (all 556,289 parameters, same seeds). That makes 36 runs in total. The following are held identical across arms:

- the base checkpoint (SHA-256 recorded);
- the data splits;
- the training-batch order for a given seed (a seeded sampler);
- the step budget (800 steps, batch 128);
- the optimiser (AdamW, warm-up + cosine);
- the LoRA learning rate (0.002; full fine-tuning uses 0.0002);
- checkpoint selection (lowest validation loss, evaluated every 50 steps).

The seed controls adapter initialisation, dropout and batch order. Everything runs in one free-tier Colab GPU session from one entry point (`scripts/run_all.py`).

## 3. Data provenance and governance (AIA IV §2(d); Art. 10)

| Item | Detail |
|---|---|
| Source | Synthetic. Generated by `lora_ablation/data.py` (generator SHA-256 `761bdb9b308bf661`). No external download, no personal data, no licence restrictions. |
| Rationale | It gives a controlled, documented distribution shift (heat pumps + EV charging) with known ground truth, and it reruns byte-identically with no access or licensing dependency. |
| Reproducibility | Master seed 20260101. The manifest of this run records hashes of the gzip files only (`source_coops` `75a0099458e2cdeb`, `target_coop` `8d73955277cf4177`). These match on reruns on the same platform but differ between zlib builds, so they are not a cross-platform check. Stage 1 now also records a hash of the uncompressed content for that purpose. Stage 1 re-checks the hashes on every rerun and reports MATCH or MISMATCH. |
| Content | Hourly, 2022-01-01 to 2024-12-31. 24 source co-ops (base load 15–80 MW). Each has its own simulated weather (seasonal + diurnal + autocorrelated anomalies), a simulated day-ahead temperature-forecast error, heating/cooling response with thermal inertia, irrigation, commercial and residential profiles, holiday calendars, load growth and autocorrelated noise. Per-co-op parameters are in `data/data_card.json`. |
| Target co-op | Nordic EU territory (40 MW). Heat-pump heating slope 0.05/°C, about 2× the top of the source range. Late-evening EV charging at 25% of base load, which the source co-ops lack. No cooling or irrigation load. |
| Splits (target) | Train 2024-08-01 → 2024-10-30 (90 days; history windows also kept inside the split). Validation 2024-10-30 → 2024-11-13 (checkpoint selection only). Test 2024-11-13 → 2024-12-13 (scored once per run). Pretraining data ends before the target train window opens. |
| Known gaps vs real data | No meter gaps, outages, reconfigurations, DER/rooftop solar or large-customer step changes. Forecast-error tails are Gaussian rather than fat-tailed. The functional form is the generator's own (see §6). |

## 4. Training methodology and reproducibility (AIA IV §2(a), §2(g); Art. 12)

- **Entry point.** Five config-driven stages (`01_make_data` → `02_pretrain` → `03_run_ablation` → `04_analyze` → `05_render_report`). Pinned dependencies are in `requirements.txt`. The runtime environment (GPU, CUDA, library versions, git commit) is recorded in `results/ablation_meta.json`.
- **Determinism.** Python, NumPy and torch are seeded. cuDNN runs in deterministic mode with deterministic algorithms enforced (`warn_only`). Attention-backward kernels on CUDA can still vary at about 1e-6, so reruns agree to several decimal places but are not bit-exact.
- **Logging (record-keeping).** Each run appends one row to `results/runs.csv`: metrics, wall-clock, peak memory, latency, device and UTC timestamp. Per-run validation curves are written to `results/curves/`. The runner is resumable, so an interrupted session loses at most one run.

## 5. Evaluation metrics and results (AIA IV §2(g), §3, §4; Art. 15)

**Metrics and why they fit (AIA IV §4).** The **primary** metric is MAPE, chosen because co-ops compare forecast quality across territories of different size. MAE and RMSE are reported in MW for operational sizing. The **daily-peak error** is the absolute % error of the forecast 24-h maximum. It matters because demand charges and capacity purchases are driven by peaks. All metrics are computed on 697 hourly forecast origins across the 30-day test window. Each run's test metric is one observation, and **the seed is the unit of replication**. Overlapping windows are never treated as independent samples. Costs are logged for every run: fine-tuning wall-clock, peak GPU memory (total, and above the resident dataset), trainable parameters and adapter size, and median forward latency at batch 1 and batch 256, both before and after merging the adapters.

**Decision rule, fixed before analysis.** An arm differs from the best LoRA rank only if two things are true. First, the Holm-adjusted Welch t-test gives p < 0.05. Second, the difference of means exceeds 2× the pooled seed SD. Rank as a whole is judged with a one-way ANOVA across ranks and a log2-rank trend test.

### 5.1 Accuracy (mean ± seed SD [95% CI half-width])

| Setting | Seeds | Trainable params | Adapter (KiB) | Test MAPE % (mean ± SD [95% CI]) | MAE (MW) | RMSE (MW) | Peak err % |
|---|---|---|---|---|---|---|---|
| Zero-shot | 1 | 0 | 0.0 | 2.616 | 1.952 | 2.566 | 1.916 |
| LoRA r=1 | 5 | 7,168 | 28.0 | 1.784 ± 0.061 [±0.076] | 1.285 ± 0.042 | 1.765 ± 0.062 | 1.398 ± 0.044 |
| LoRA r=2 | 5 | 14,336 | 56.0 | 1.797 ± 0.043 [±0.053] | 1.294 ± 0.026 | 1.784 ± 0.030 | 1.358 ± 0.020 |
| LoRA r=4 | 5 | 28,672 | 112.0 | 1.876 ± 0.044 [±0.054] | 1.343 ± 0.023 | 1.825 ± 0.019 | 1.359 ± 0.060 |
| LoRA r=8 | 5 | 57,344 | 224.0 | 2.099 ± 0.070 [±0.086] | 1.504 ± 0.051 | 1.998 ± 0.060 | 1.504 ± 0.108 |
| LoRA r=16 | 5 | 114,688 | 448.0 | 2.190 ± 0.110 [±0.136] | 1.566 ± 0.074 | 2.067 ± 0.094 | 1.500 ± 0.164 |
| LoRA r=32 | 5 | 229,376 | 896.0 | 2.216 ± 0.157 [±0.195] | 1.581 ± 0.105 | 2.059 ± 0.141 | 1.578 ± 0.130 |
| Full FT | 5 | 556,289 | 2173.0 | 2.005 ± 0.076 [±0.094] | 1.438 ± 0.046 | 1.949 ± 0.043 | 1.376 ± 0.022 |

### 5.2 Cost

| Setting | Fine-tune time (s) | Peak GPU mem (MiB) | Mem above resident data (MiB) | Latency b1 (ms) | Latency b1 merged (ms) | Latency b256 (ms) | Latency b256 merged (ms) |
|---|---|---|---|---|---|---|---|
| Zero-shot | 0.0 | nan | nan | 5.82 | 5.85 | 42.85 | 42.99 |
| LoRA r=1 | 71.6 ± 0.1 | 671.9 ± 0.0 | 653.3 ± 0.0 | 11.32 ± 0.52 | 6.36 ± 0.14 | 64.07 ± 0.16 | 43.44 ± 0.15 |
| LoRA r=2 | 74.7 ± 0.1 | 674.1 ± 0.0 | 655.6 ± 0.0 | 11.39 ± 0.43 | 6.21 ± 0.20 | 66.93 ± 0.11 | 43.46 ± 0.10 |
| LoRA r=4 | 74.9 ± 0.1 | 678.7 ± 0.0 | 660.1 ± 0.0 | 11.53 ± 0.36 | 6.30 ± 0.28 | 67.06 ± 0.21 | 43.42 ± 0.15 |
| LoRA r=8 | 75.3 ± 0.1 | 687.7 ± 0.0 | 669.2 ± 0.0 | 11.49 ± 0.26 | 6.08 ± 0.26 | 67.64 ± 0.12 | 43.40 ± 0.07 |
| LoRA r=16 | 75.5 ± 0.1 | 706.4 ± 0.0 | 687.8 ± 0.0 | 11.31 ± 0.56 | 6.22 ± 0.17 | 68.60 ± 0.17 | 43.33 ± 0.14 |
| LoRA r=32 | 77.3 ± 0.1 | 742.1 ± 0.0 | 723.6 ± 0.0 | 11.54 ± 0.62 | 6.37 ± 0.27 | 70.48 ± 0.10 | 43.43 ± 0.09 |
| Full FT | 62.2 ± 0.1 | 696.3 ± 0.0 | 677.8 ± 0.0 | 6.16 ± 0.33 | 6.12 ± 0.24 | 43.49 ± 0.10 | 43.47 ± 0.14 |

### 5.3 Does rank matter more than run-to-run noise?

| Arm | vs | Δ mean (pp) | Pooled seed SD | Cohen's d | > k·SD? | Welch p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|
| Zero-shot | LoRA r=1 | +0.832 | 0.061 | +13.58 | yes | 7e-06 | 4.9e-05 | distinguishable |
| LoRA r=2 | LoRA r=1 | +0.013 | 0.053 | +0.25 | no | 0.709 | 0.709 | within noise |
| LoRA r=4 | LoRA r=1 | +0.091 | 0.053 | +1.71 | no | 0.0293 | 0.0586 | within noise |
| LoRA r=8 | LoRA r=1 | +0.315 | 0.066 | +4.81 | yes | 6.87e-05 | 0.000412 | distinguishable |
| LoRA r=16 | LoRA r=1 | +0.405 | 0.089 | +4.56 | yes | 0.000292 | 0.00146 | distinguishable |
| LoRA r=32 | LoRA r=1 | +0.432 | 0.119 | +3.62 | yes | 0.00202 | 0.00607 | distinguishable |
| Full FT | LoRA r=1 | +0.221 | 0.069 | +3.21 | yes | 0.00108 | 0.00434 | distinguishable |

**Question:** does LoRA rank (r ∈ {1, 2, 4, 8, 16, 32}) change day-ahead accuracy on the target co-op by more than run-to-run (seed) noise?

**Answer: yes** — rank has a measurable effect: the spread of rank means (0.432 pp MAPE) is larger than 2× the median seed SD (0.065 pp) and one-way ANOVA p = 1.37e-08.

- Best mean LoRA arm: **LoRA r=1** at 1.784 ± 0.061% MAPE.
- 3 of 5 other ranks are distinguishable from it (Holm-adjusted Welch p < 0.05 *and* |Δ| > 2× pooled seed SD).
- Trend in log2(rank): slope +0.102 pp per doubling (p = 1.79e-10).
- Adaptation itself matters: zero-shot base model = 2.616% MAPE (Δ +0.832 pp vs best LoRA).
- Full fine-tuning = 2.005 ± 0.076% MAPE with 556,289 trainable params vs 7,168 for LoRA r=1.
- Smallest adapter (LoRA r=1) stores 28.0 KiB per co-op; once merged, adapters add no inference compute at any rank (merged latency matches the base model within timing noise).

**Recommendation for release.** Adopt **LoRA r=1** as the default adapter rank. It is the smallest rank whose test MAPE (1.784 ± 0.061%) is not distinguishable from the best arm (LoRA r=1), and it stores 28.0 KiB per co-op. Full fine-tuning is worse than the best LoRA arm by 0.221 pp (distinguishable) and needs 556,289 trainable parameters per co-op. Approval is conditional on replicating this result on real demand data (§6, item 1).

Figures: `results/fig_accuracy_vs_rank.png` (seed-level dots), `fig_pareto.png` (accuracy vs adapter size), `fig_cost_vs_rank.png`, `fig_learning_curves.png`, `fig_forecast_example.png`, `fig_peak_error_vs_rank.png`, `fig_data_overview.png`.

## 6. Limitations: what this study cannot conclude (AIA IV §3)

1. **Synthetic data, single target.** The results describe one simulated Nordic co-op and one 30-day test window in late autumn. They are evidence about *rank under a controlled shift*, not about accuracy on a real territory. The model may exploit regularities that only exist in the generator. Before release, the comparison must be repeated on real open series (e.g., EIA-930 balancing-authority demand, ENTSO-E load) and on customer data held out under contract.
2. **One base model.** All arms share one pretraining seed. Variance *between base models* is not measured, so the ranking of ranks could change with a different base.
3. **Fixed recipe across ranks.** The learning rate, step budget and α/r convention are identical for every rank. The study answers "does rank matter *at this recipe*?". It does not answer "what is each rank's best achievable accuracy?". Higher ranks may need a different learning rate (cf. rank-stabilised LoRA). A different α convention (fixed α) would confound rank with step size.
4. **Checkpoint-selection granularity.** In this run, 86% of adapted runs reached their best validation loss at the first checkpoint (step 50, the end of the 50-step warm-up). After that, validation loss rose while training loss kept falling. The comparison is therefore effectively "each rank after about 50 steps". Higher ranks fit, and over-fit, faster, and their best checkpoint may lie before step 50, which this evaluation grid cannot resolve. The advantage of low ranks is therefore confounded with checkpoint granularity and the fixed learning rate. A follow-up with validation every 5–10 steps, or a lower learning rate, is needed before treating it as a pure capacity effect.
5. **Statistical power.** With 5 seeds per arm, the minimum difference detectable with 80% power at α = 0.05 is about 2.0 pooled seed SDs (≈0.13 pp MAPE here). A "within noise" verdict means "no effect large enough to see". It does not mean "no effect".
6. **Scale.** The model has 556,289 parameters. Rank-versus-memory and rank-versus-latency behaviour at this scale (activation-dominated) does not extrapolate to LLM-scale LoRA, where adapter optimiser state can matter.
7. **Cost measurements.** The Colab GPU is shared, so latency includes co-tenant variance. Batch-1 latency is dominated by kernel launch overhead. Peak memory includes the dataset resident on the GPU, which is constant across arms.
8. **Seasonal coverage.** Adapters see August–October only. Winter cold snaps, when heat-pump load peaks, are outside the training distribution and only partly inside the test window.

## 7. Known risks and mitigations (AIA IV §5; Art. 9)

| Risk | Consequence | Mitigation / control | Owner |
|---|---|---|---|
| Under-forecast during cold snaps (heat-pump load) | Co-op under-buys capacity; spot-price exposure; peak-charge penalties | Report daily-peak error separately; planner review; alert when the temperature forecast falls below the training range | Research + customer ops |
| Adapter staleness (EV/heat-pump growth, new large loads) | Gradual bias | Monthly rolling-origin MAPE monitoring per co-op; retrain the adapter when MAPE exceeds 1.5× its validation level for 14 days; keep a base-model fallback | ML Ops |
| Over-fitting a 90-day adapter | Optimistic validation; poor generalisation | Early stopping on a held-out window; test window never used for selection; seed spread reported | Research |
| Automation bias | Planners accept a bad forecast | Out-of-range input flags; documented limitations in the instructions for use | Product + Compliance |
| Data-quality faults (gaps, meter errors) | Corrupted inputs → silently wrong forecasts | Input validation (ranges, gaps, flat-lines) before inference; reject and fall back | Engineering |
| Cross-customer data use in the shared base | Contractual/data-rights breach | Data-use clauses for pooled pretraining; per-co-op adapters stored separately with hashes | Compliance |
| Supply-chain / integrity | Altered weights or dependencies | Pinned dependencies; SHA-256 for data and checkpoints; adapters signed at release | Engineering |
| Over-claiming from synthetic evidence | Release decision on non-representative data | §6 limitations; real-data replication gate before GA | Head of Research |

**Human oversight (Art. 14) and post-market monitoring (Art. 72; AIA IV §9).** Forecasts are advisory. Planners can override any forecast and can revert a co-op to the base model or a previous adapter. Production logs keep the input hash, the adapter version, the forecast and the realised demand for every issuance (Art. 12). Each adapter version is a tracked change (AIA IV §6). Changing the rank or the base model triggers a rerun of this study's protocol.

## 8. Controls mapped to NIST AI RMF and ISO/IEC 42001

| Control in this study | AIA reference | NIST AI RMF 1.0 | ISO/IEC 42001 Annex A | Evidence |
|---|---|---|---|---|
| Intended purpose, users, out-of-scope uses | IV §1; Art. 13 | MAP 1.1, MAP 3.3 | A.6.2.2, A.9.4 | §1 |
| Classification and impact reasoning | Art. 6(3)–(4) | GOVERN 1.1, MAP 1.1 | A.5.2, A.5.3 | Scope note |
| Synthetic-data provenance, seed, hashes, data card | IV §2(d); Art. 10 | MAP 2.3, MEASURE 2.1 | A.7.2, A.7.3, A.7.5, A.7.6 | `data/manifest.json`, `data_card.json` |
| Temporal splits; test never used for selection | IV §2(g); Art. 10(3) | MEASURE 2.1, MEASURE 2.3 | A.7.4, A.6.2.4 | §3, config |
| Controlled ablation, seeds, pre-registered decision rule | IV §2(b),(g) | MAP 2.3, MEASURE 1.1, MEASURE 2.5 | A.6.2.4 | `configs/`, §5 |
| Seed spread, CIs, multiple-comparison correction | IV §2(g), §4 | MEASURE 2.3, MEASURE 2.5 | A.6.2.4 | `results/significance.csv` |
| Cost logging (time, memory, latency, adapter size) | IV §2(c) | MEASURE 2.12 | A.4.5 | `results/runs.csv` |
| Pinned environment, config hash, env snapshot | IV §2(c), §1(c) | MEASURE 2.1, GOVERN 1.4 | A.4.4, A.6.2.3, A.6.2.7 | `requirements*.txt`, `ablation_meta.json` |
| Per-run logs; resumable execution | Art. 12 | MEASURE 2.8 | A.6.2.8 | `results/runs.csv`, `curves/` |
| Limitations & "cannot conclude" statement | IV §3 | MAP 2.2, MEASURE 2.5 | A.8.2 | §6 |
| Risk register with owners | IV §5; Art. 9 | MANAGE 1.2–1.3, GOVERN 2.1 | Cl. 6.1.2–6.1.3, A.3.2 | §7 |
| Human oversight, fallback, monitoring plan | IV §2(e), §9; Arts. 14, 72 | MANAGE 2.4, MANAGE 4.1, MEASURE 3.1 | A.6.2.6, A.9.2 | §7 |

**Not covered by this summary.** No harmonised standards are applied yet (AIA IV §7). The EU declaration of conformity (AIA IV §8) is pending the classification decision.
