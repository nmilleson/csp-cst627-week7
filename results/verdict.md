**Question:** does LoRA rank (r ∈ {1, 2, 4, 8, 16, 32}) change day-ahead accuracy on the target co-op by more than run-to-run (seed) noise?

**Answer: yes** — rank has a measurable effect: the spread of rank means (0.432 pp MAPE) is larger than 2× the median seed SD (0.065 pp) and one-way ANOVA p = 1.37e-08.

- Best mean LoRA arm: **LoRA r=1** at 1.784 ± 0.061% MAPE.
- 3 of 5 other ranks are distinguishable from it (Holm-adjusted Welch p < 0.05 *and* |Δ| > 2× pooled seed SD).
- Trend in log2(rank): slope +0.102 pp per doubling (p = 1.79e-10).
- Adaptation itself matters: zero-shot base model = 2.616% MAPE (Δ +0.832 pp vs best LoRA).
- Full fine-tuning = 2.005 ± 0.076% MAPE with 556,289 trainable params vs 7,168 for LoRA r=1.
- Smallest adapter (LoRA r=1) stores 28.0 KiB per co-op; once merged, adapters add no inference compute at any rank (merged latency matches the base model within timing noise).
