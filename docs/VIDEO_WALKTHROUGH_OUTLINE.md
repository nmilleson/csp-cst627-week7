# Video walkthrough outline (8–10 min, camera on throughout)

**Audience:** Dr. Ingrid Solberg (Head of Research; she decides on the release) and Marcus Adeyemi (Compliance; he
needs the conformity file). Keep your face in a corner overlay the whole time. Fill each **[bracket]** from
`results/verdict.md` and `results/summary.md` after the Colab run.

| Time | Segment | On screen | Talking points |
|---|---|---|---|
| 0:00–0:45 | **The question** | Face only, then README header | "Ingrid, you asked one narrow question before release: which LoRA rank should we ship when we adapt the shared model to a new co-op, and are the differences between ranks real or just noise? Marcus, everything I show is packaged for the conformity file." |
| 0:45–2:00 | **Data & shift** | `fig_data_overview.png`, `data/manifest.json` | Synthetic by design: known shift, reruns byte-identically. 24 source co-ops vs a Nordic target with heat pumps (steeper heating slope) and EV evening peaks. Show the SHA-256 and the `MATCH` line. Splits: 90 / 14 / 30 days; the test set is never used for selection. |
| 2:00–3:15 | **Controlled design** | `configs/ablation.yaml` | One factor (rank 1→32). α = 2r keeps the scale constant, so capacity is not confounded with step size. Same base checkpoint, same batches per seed, same budget. 5 seeds per arm; zero-shot and full fine-tune as brackets. The decision rule was written before the results: Holm-Welch **and** > 2 seed SDs. |
| 3:15–5:30 | **Results** | `fig_accuracy_vs_rank.png`, `summary.md`, `fig_pareto.png` | Point at the seed dots first: "this is the noise." Then the means. **[Answer: yes/no, rank matters]**. **[Best rank, MAPE ± SD]**. **[How many ranks are distinguishable]**. Adaptation vs zero-shot: **[Δ]**. Full fine-tune: **[result]** with 2–80× more trainable parameters than the adapters (556k vs 7k–229k). Mention the peak-error figure, because co-ops pay for peaks. |
| 5:30–6:30 | **Cost** | `fig_cost_vs_rank.png`, cost table | Wall-clock and peak memory are **[nearly flat / differences]** across ranks at this scale (activations dominate). Adapter storage grows linearly. Merged latency equals the base model, so rank has no serving cost after merging. |
| 6:30–7:15 | **Recommendation** | Annex IV §5 "Recommendation" | **[Adopt r = X]**: the smallest rank not distinguishable from the best. It is conditional on replication on real data. |
| 7:15–8:30 | **What it cannot conclude** | Annex IV §6 | Synthetic, single target, single autumn window. One base model. Fixed recipe (an LR tuned per rank might change the ranking). With 5 seeds only effects ≥ ~2 SD are detectable. Small model, so do not extrapolate memory behaviour to LLMs. |
| 8:30–9:30 | **Package & compliance** | Repo tree, notebook, Annex IV §8 table | A stranger can run Colab → Run all. Pinned dependencies, seeded config, resumable runner, env snapshot. Marcus: walk through the classification note (Annex III pt 2 vs the Art. 6(3) preparatory-task derogation; his call), the risk register, and the NIST AI RMF / ISO 42001 mapping table. |
| 9:30–10:00 | **Close** | Face | One-sentence answer, the one condition (real-data replication), and the next step. |

**Recording tips:** run the notebook *before* recording and scroll through the finished outputs, so you are not
waiting on training on camera. Keep the camera overlay away from the figures' axes.
