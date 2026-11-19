# R6: is engineered feature 24 (MAD) constant?

Camera-ready roadmap row **R6** (reviewer LrHa, item B6). Run 2026-11-16. Script:
`check_feature24.py` (swm env, `PYTHONPATH=src`).

## Question

Feature 24 is `median|x - median(x)|` of a star's first stored segment
(`src/swm/eval/features.py:139`). It is computed without the 1.4826 factor that the normalisation
divides by. If `x` were the whole normalised segment, the feature would be exactly
1/1.4826 = 0.6745 for every star.

## Result (`feature24_summary.csv`, rotation pool, 106,284 stars)

| mean | sd | 1 % | 99 % | min | max | share exactly 0.6745 |
|---|---|---|---|---|---|---|
| 0.671 | 0.028 | 0.571 | 0.752 | 0.356 | 1.354 | 0.7 % |

- **Footing:** feature 24 was recomputed from the stored windows of 200 random stars. Max difference
  from the cached value: 0.0.
- **Why it varies.** A segment is normalised as a whole. It is then cut into whole 1,024-point
  blocks, and the remainder is dropped (`src/build_sequences_bulk.py:418-456`). So `x` is a truncated
  copy of the normalised segment.
  - No NaN is removed after normalisation: segments are NaN-free at `gap_threshold = 1`.
  - No window is removed by the absmax guard: all 677 files over the guard were already deleted, see
    `processed/flux_outliers_cleanup_manifest.csv`.
- The pool-1 cache `exp10_features/subset_features25.parquet` holds standardised columns and cannot
  answer this question, so it is not read.

## Reproduce

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 /c/Users/user1/.venvs/swm/Scripts/python.exe experiments/feature_mad_check/check_feature24.py 2>&1 | tee experiments/feature_mad_check/run.log
```
