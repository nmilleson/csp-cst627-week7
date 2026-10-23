# Technical Documentation Summary (EU AI Act Annex IV-style)

## Cobalt Grid Short-Horizon Demand Forecaster: LoRA-rank adaptation study

| Field | Value |
|---|---|
| Provider | Cobalt Grid (ML Research) |
| Document owner | ML Research Engineer (study author) |
| Reviewers | Dr. Ingrid Solberg (Head of Research): release approval · Marcus Adeyemi (Compliance): conformity file |
| Document status | Draft summary for the conformity file, generated ${generated_utc} from repository commit `${git_commit}`, config hash `${config_hash}` |
| Evidence | Repository `README.md`, `configs/ablation.yaml`, `results/` (tables, plots, run logs), `data/manifest.json` |

**Scope of this summary.** This document summarises the technical documentation for one release decision: *which LoRA rank Cobalt Grid should use when adapting its shared day-ahead demand-forecasting model to a new co-op*. It follows the structure of EU AI Act Annex IV (referenced as "AIA IV §n") and cross-references the NIST AI Risk Management Framework 1.0 ("RMF") and ISO/IEC 42001:2023 Annex A ("42001 A.x"). The evidence comes from a controlled study on **synthetic data**. It supports the engineering decision about rank. It does **not** establish field accuracy on any real co-op.

**Regulatory status (for Compliance to confirm).** Annex III point 2 lists AI systems used as *safety components* in the management and operation of electricity supply. The forecaster informs co-op planning (power purchasing, peak shaving, maintenance windows), and a human planner reviews its output. It does not directly actuate grid equipment. Cobalt Grid may therefore be able to rely on the Art. 6(3) "preparatory task" derogation, which requires a documented assessment (Art. 6(4)). Until Compliance records that decision, this file is maintained to the high-risk standard (Arts. 9–15) as a conservative default. A customer integration that feeds forecasts into automated load control without review would change the classification and requires re-assessment.

---

## 1. General description of the system (AIA IV §1)

**Intended purpose.** Produce hourly electricity-demand forecasts for the next 24 hours for a rural distribution co-op's whole service territory (aggregate MW). The forecast is refreshed every hour. Users are co-op operations and power-supply planners. Outputs feed day-ahead purchasing, peak-shaving decisions and demand-charge management. Every output is reviewed by a qualified planner before it is acted on.

**Out of scope / prohibited uses.** Protection or real-time balancing control. Forecasting individual households or meters. Billing. Any automated curtailment without human review. Territories with fewer than 90 days of hourly history.

**Inputs → outputs.** Input: the previous 168 h of aggregate demand; observed temperature for the history window; a day-ahead temperature forecast for the horizon; calendar features and the regional holiday calendar. Output: 24 hourly demand values (MW). No personal data is processed. The inputs are territory-level aggregates.

**Deployment form.** One shared, frozen *base model* is pretrained on many co-ops. Each customer co-op receives a small *LoRA adapter* trained on its own history. Adapters can be merged into the base weights at deployment, so inference costs the same as the base model whatever the rank. Software stack: Python ${python}, PyTorch ${torch}. The study ran on: ${gpu}.

## 2. Development process and design (AIA IV §2(a)–(c))

**Base model.** A pre-LayerNorm transformer encoder (${n_layers} layers, d_model ${d_model}, ${n_heads} heads, ${base_params} parameters). Each hourly timestep is one token. History tokens carry instance-normalised demand. Horizon tokens carry covariates and a mask flag instead of demand. A linear head reads the 24 horizon tokens. It was pretrained once (seed ${pretrain_seed}) on ${n_source} source co-ops (US and Central-EU) with data up to ${pretrain_val_end}. Source-validation MAPE was ${pretrain_mape}%, with ${pretrain_time} s of training.

**Adaptation (the factor under study).** LoRA (Hu et al., 2021) is applied to every attention projection (q, k, v, out) and both feed-forward layers in every block. The update is W + (α/r)·BA, with B initialised to zero. The base model, embeddings, LayerNorms and head stay frozen. **Design choice:** α = 2r, so the scale α/r = 2 is the same at every rank. Changing r therefore changes adapter capacity without silently changing the effective step size.

**Study design (controlled ablation).** Ranks r ∈ {${ranks}} × ${n_seeds} seeds. Two bracketing arms are added: *zero-shot* (base model, no adaptation, deterministic, 1 run) and *full fine-tuning* (all ${base_params} parameters, same seeds). That makes ${n_runs} runs in total. The following are held identical across arms:

- the base checkpoint (SHA-256 recorded);
- the data splits;
- the training-batch order for a given seed (a seeded sampler);
- the step budget (${steps} steps, batch ${batch_size});
- the optimiser (AdamW, warm-up + cosine);
- the LoRA learning rate (${lr}; full fine-tuning uses ${full_lr});
- checkpoint selection (lowest validation loss, evaluated every ${eval_every} steps).

The seed controls adapter initialisation, dropout and batch order. Everything runs in one free-tier Colab GPU session from one entry point (`scripts/run_all.py`).

## 3. Data provenance and governance (AIA IV §2(d); Art. 10)

| Item | Detail |
|---|---|
| Source | Synthetic. Generated by `lora_ablation/data.py` (generator SHA-256 `${generator_sha}`). No external download, no personal data, no licence restrictions. |
| Rationale | It gives a controlled, documented distribution shift (heat pumps + EV charging) with known ground truth, and it reruns byte-identically with no access or licensing dependency. |
| Reproducibility | Master seed ${data_seed}. Files are written with a fixed gzip timestamp. `source_coops.csv.gz` SHA-256 `${source_sha}`, `target_coop.csv.gz` SHA-256 `${target_sha}`. Stage 1 re-checks the hashes on every rerun and reports MATCH or MISMATCH. |
| Content | Hourly, ${data_start} to ${data_end}. ${n_source} source co-ops (base load 15–80 MW). Each has its own simulated weather (seasonal + diurnal + autocorrelated anomalies), a simulated day-ahead temperature-forecast error, heating/cooling response with thermal inertia, irrigation, commercial and residential profiles, holiday calendars, load growth and autocorrelated noise. Per-co-op parameters are in `data/data_card.json`. |
| Target co-op | Nordic EU territory (40 MW). Heat-pump heating slope 0.05/°C, about 2× the top of the source range. Late-evening EV charging at 25% of base load, which the source co-ops lack. No cooling or irrigation load. |
| Splits (target) | Train ${train_split} (90 days; history windows also kept inside the split). Validation ${val_split} (checkpoint selection only). Test ${test_split} (scored once per run). Pretraining data ends before the target train window opens. |
| Known gaps vs real data | No meter gaps, outages, reconfigurations, DER/rooftop solar or large-customer step changes. Forecast-error tails are Gaussian rather than fat-tailed. The functional form is the generator's own (see §6). |

## 4. Training methodology and reproducibility (AIA IV §2(a), §2(g); Art. 12)

- **Entry point.** Five config-driven stages (`01_make_data` → `02_pretrain` → `03_run_ablation` → `04_analyze` → `05_render_report`). Pinned dependencies are in `requirements.txt`. The runtime environment (GPU, CUDA, library versions, git commit) is recorded in `results/ablation_meta.json`.
- **Determinism.** Python, NumPy and torch are seeded. cuDNN runs in deterministic mode with deterministic algorithms enforced (`warn_only`). Attention-backward kernels on CUDA can still vary at about 1e-6, so reruns agree to several decimal places but are not bit-exact.
- **Logging (record-keeping).** Each run appends one row to `results/runs.csv`: metrics, wall-clock, peak memory, latency, device and UTC timestamp. Per-run validation curves are written to `results/curves/`. The runner is resumable, so an interrupted session loses at most one run.

## 5. Evaluation metrics and results (AIA IV §2(g), §3, §4; Art. 15)

**Metrics and why they fit (AIA IV §4).** The **primary** metric is MAPE, chosen because co-ops compare forecast quality across territories of different size. MAE and RMSE are reported in MW for operational sizing. The **daily-peak error** is the absolute % error of the forecast 24-h maximum. It matters because demand charges and capacity purchases are driven by peaks. All metrics are computed on 697 hourly forecast origins across the 30-day test window. Each run's test metric is one observation, and **the seed is the unit of replication**. Overlapping windows are never treated as independent samples. Costs are logged for every run: fine-tuning wall-clock, peak GPU memory (total, and above the resident dataset), trainable parameters and adapter size, and median forward latency at batch 1 and batch 256, both before and after merging the adapters.

**Decision rule, fixed before analysis.** An arm differs from the best LoRA rank only if two things are true. First, the Holm-adjusted Welch t-test gives p < 0.05. Second, the difference of means exceeds 2× the pooled seed SD. Rank as a whole is judged with a one-way ANOVA across ranks and a log2-rank trend test.

### 5.1 Accuracy (mean ± seed SD [95% CI half-width])

${accuracy_table}

### 5.2 Cost

${cost_table}

### 5.3 Does rank matter more than run-to-run noise?

${significance_table}

${verdict_md}

**Recommendation for release.** ${recommendation}

Figures: `results/fig_accuracy_vs_rank.png` (seed-level dots), `fig_pareto.png` (accuracy vs adapter size), `fig_cost_vs_rank.png`, `fig_learning_curves.png`, `fig_forecast_example.png`, `fig_peak_error_vs_rank.png`, `fig_data_overview.png`.

## 6. Limitations: what this study cannot conclude (AIA IV §3)

1. **Synthetic data, single target.** The results describe one simulated Nordic co-op and one 30-day test window in late autumn. They are evidence about *rank under a controlled shift*, not about accuracy on a real territory. The model may exploit regularities that only exist in the generator. Before release, the comparison must be repeated on real open series (e.g., EIA-930 balancing-authority demand, ENTSO-E load) and on customer data held out under contract.
2. **One base model.** All arms share one pretraining seed. Variance *between base models* is not measured, so the ranking of ranks could change with a different base.
3. **Fixed recipe across ranks.** The learning rate, step budget and α/r convention are identical for every rank. The study answers "does rank matter *at this recipe*?". It does not answer "what is each rank's best achievable accuracy?". Higher ranks may need a different learning rate (cf. rank-stabilised LoRA). A different α convention (fixed α) would confound rank with step size.
4. **Statistical power.** With ${n_seeds} seeds per arm, the minimum difference detectable with 80% power at α = 0.05 is about ${mde_sd} pooled seed SDs (≈${mde_pp} pp MAPE here). A "within noise" verdict means "no effect large enough to see". It does not mean "no effect".
5. **Scale.** The model has ${base_params} parameters. Rank-versus-memory and rank-versus-latency behaviour at this scale (activation-dominated) does not extrapolate to LLM-scale LoRA, where adapter optimiser state can matter.
6. **Cost measurements.** The Colab GPU is shared, so latency includes co-tenant variance. Batch-1 latency is dominated by kernel launch overhead. Peak memory includes the dataset resident on the GPU, which is constant across arms.
7. **Seasonal coverage.** Adapters see August–October only. Winter cold snaps, when heat-pump load peaks, are outside the training distribution and only partly inside the test window.

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
