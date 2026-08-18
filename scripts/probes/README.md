# One-off probes

Ad-hoc scripts written to answer a specific question during the review. They are kept
because **claims in the main README depend on them**, and a claim whose evidence lived only
on a machine that has since been shut down is not reproducible.

| script | the claim it backs |
|---|---|
| `check_baseline.py` | the union-table accumulator and `cocolite.evaluate_detections` agree to 0.000000 AP on real data — the check that localised the CUDA bug to the accumulator rather than to a stale detection cache |
| `fix_oracle_check.py` | the CUDA divisor bug moves ΔAP by <0.001 and the headline ratio from +15.5% to +15.4% |
| `greedy_flip_probe.py` | **0 of 600** images change their greedy argmax between the buggy and fixed objectives, which is why the committed assignments did not need re-running |
| `apply_oracle_correction.py` | how `results/oracle/*_AP50_all.json` were rewritten, preserving the pre-correction values under `accumulator_correction` |
| `probe_rtdetr.py` | RT-DETR-L's 0.23 AP shortfall is not a threshold artefact — AP is identical at conf ∈ {0.001, 0} and max_det ∈ {300, 1000} |
| `probe_models.py` | which cross-check checkpoints load unfused, which heads each serves, and per-shift timing used to budget the runs |

Run from the repo root with the detection caches present (`make cache-oracle`). Unlike
`scripts/00`–`13` these are not part of the pipeline and are not wired into the Makefile.
