## Accuracy (test split, 30 days, hourly forecast origins)

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

## Cost

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

## Each arm vs the best LoRA rank

| Arm | vs | Δ mean (pp) | Pooled seed SD | Cohen's d | > k·SD? | Welch p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|
| Zero-shot | LoRA r=1 | +0.832 | 0.061 | +13.58 | yes | 7e-06 | 4.9e-05 | distinguishable |
| LoRA r=2 | LoRA r=1 | +0.013 | 0.053 | +0.25 | no | 0.709 | 0.709 | within noise |
| LoRA r=4 | LoRA r=1 | +0.091 | 0.053 | +1.71 | no | 0.0293 | 0.0586 | within noise |
| LoRA r=8 | LoRA r=1 | +0.315 | 0.066 | +4.81 | yes | 6.87e-05 | 0.000412 | distinguishable |
| LoRA r=16 | LoRA r=1 | +0.405 | 0.089 | +4.56 | yes | 0.000292 | 0.00146 | distinguishable |
| LoRA r=32 | LoRA r=1 | +0.432 | 0.119 | +3.62 | yes | 0.00202 | 0.00607 | distinguishable |
| Full FT | LoRA r=1 | +0.221 | 0.069 | +3.21 | yes | 0.00108 | 0.00434 | distinguishable |

## Verdict

**Question:** does LoRA rank (r ∈ {1, 2, 4, 8, 16, 32}) change day-ahead accuracy on the target co-op by more than run-to-run (seed) noise?

**Answer: yes** — rank has a measurable effect: the spread of rank means (0.432 pp MAPE) is larger than 2× the median seed SD (0.065 pp) and one-way ANOVA p = 1.37e-08.

- Best mean LoRA arm: **LoRA r=1** at 1.784 ± 0.061% MAPE.
- 3 of 5 other ranks are distinguishable from it (Holm-adjusted Welch p < 0.05 *and* |Δ| > 2× pooled seed SD).
- Trend in log2(rank): slope +0.102 pp per doubling (p = 1.79e-10).
- Adaptation itself matters: zero-shot base model = 2.616% MAPE (Δ +0.832 pp vs best LoRA).
- Full fine-tuning = 2.005 ± 0.076% MAPE with 556,289 trainable params vs 7,168 for LoRA r=1.
- Smallest adapter (LoRA r=1) stores 28.0 KiB per co-op; once merged, adapters add no inference compute at any rank (merged latency matches the base model within timing noise).
