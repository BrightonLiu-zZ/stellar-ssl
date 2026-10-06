# rotation_pool — the pool-3 rotation rescore (arXiv roadmap Phase 1, 2026-09-20 → 09-25)

Design + sign-off: `docs/plans/2026-09-20-rotation-pool-design.md`. Manifest: `experiments/configs/rotation_pool.yaml`.
Pool 3 = the 106,284 corpus stars outside pool 1 (13,789 TARS rotators, 92,495 non-rotators, prevalence 0.130,
no transit/eb/pulsating positive, no star seen in pre-training). Built by `swm.eval.rotation_pool` →
`processed/subset/rotation_pool.parquet`. No encoder pass: mu and features read from `experiments/r8_fullpool/`.

| file | produced by | what |
|---|---|---|
| `probe.csv`, `summary.csv`, `absolute.csv` | `analyze_rotation_pool.py` (linear family, 3 readouts, pool3 + noflare + strict + pool1) | F1-schema probe rows, paired deltas, absolutes |
| `verdict.csv` | `report_rotation_pool.py` | the pre-registered reading (design §7 + A2) |
| `*_xgb.csv` | `analyze_rotation_pool.py --families xgb --tag xgb` | XGBoost readout, `mean`, pool3. **`--tag xgb` is required**: without it the run writes the plain `probe/summary/absolute.csv` and overwrites the linear results |
| `s1/` | `analyze_s1_label_efficiency.py --rotation-pool` (2026-10-05, CPU, ~2 min) | label-efficiency curves for both tasks, same ladder/floor/draws as S1, untrained = `untrained_i0`. FOOTING-1 10/10 under 1e-4 against `absolute.csv`. **rotation narrows** (fusion delta −0.016 at 100 stars vs +0.055 at 74,398; growth −0.071 ± 0.006), **rotation_period widens** (+0.086 at 50 vs +0.034 at 6,150; +0.052 ± 0.008) — the same calls pool 1 gave, so the paper's "narrows on 7 of 10" stands |
| `star_scores.parquet` | `analyze_rotation_pool.py` | per-star test scores at `mean` (bootstrap input) |
| `c1c2/` | `run_rotation_pool_supervised.ps1` (user terminal) → `analyze_c1c2_supervised.py --manifest configs/rotation_pool.yaml` | end-to-end baselines, 24 runs, 76 min. **Its `c1c2_summary.csv` deltas reference the POOL-1 fusion score (hardwired) — ignore them; use `c1c2_absolute.csv`** |
| `bootstrap/` | `prep_rotation_pool_bootstrap.py` → `replay_c1c2_scores.py` → `analyze_paper_bootstrap.py --in-dir experiments/rotation_pool/bootstrap --table experiments/rotation_pool/bootstrap/table1_data.csv --ablation experiments/rotation_pool/bootstrap/dynamics_ablation.csv --tasks rotation rotation_period` | paired star-bootstrap, 1000 resamples, 12 contrasts; `paired_bootstrap.csv` is published. **Warning:** `analyze_paper_bootstrap.py`'s default `--tex-out` is `paper/tables/tableC_bootstrap.tex` (Appendix G). The table writer loops over the paper's ten tasks, so on these two it raises `IndexError` *after* writing `paired_bootstrap.csv` and printing the verdict table, and *before* opening the `.tex` file — that crash (see `bootstrap/run.log`) is what kept Appendix G intact on 2026-09-25, not the arguments. Pass `--tex-out experiments/rotation_pool/bootstrap/unused.tex` anyway, so a fixed writer cannot overwrite Appendix G, and read the 12/12 agreement from the printed `agree` column |

Footing: F1 (pool-1 rows reproduce), F2 (untrained_i0 = paper's untrained; discriminative, deviation D1), F3 (features
match pool 2's cache) — all PASS 2026-09-20. C1/C2 replay reproduces result.json to 6e-10.

## Headline (linear, `mean`, hann0p3_fbwd, 6 seeds, ± 2 SE over seeds)

| task | n_test | features | mu | fusion | gain | untrained gain | trained − untrained | fbwd − off | Conv1D | MLP | counts? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rotation (PR-AUC) | 15,944 | 0.386 | 0.373 | 0.441 | +0.055 ± 0.006 | +0.027 ± 0.003 | +0.029 ± 0.008 | +0.019 ± 0.007 | 0.409 | 0.420 | **no** (A2: untrained also > 2 SE) |
| rotation period (R²) | 1,318 | 0.718 | 0.636 | 0.752 | +0.034 ± 0.002 | −0.003 ± 0.003 | +0.037 ± 0.003 | +0.020 ± 0.003 | 0.697 | 0.773 | **yes** |

Pool 1 (ML4PS): rotation 0.540 / 0.559 / 0.555, gain +0.016 ± 0.016 (no); rotation period 0.703 / 0.677 / 0.717,
gain +0.013 ± 0.010 (yes). Count unchanged at 7 of 10. Sensitivity rows (no-flare, strict) agree on both verdicts.
XGBoost: fusion gain −0.003 ± 0.002 and −0.004 ± 0.003 (features alone 0.518 / 0.829); trained mu still beats
untrained (+0.010, +0.008). Bootstrap agrees with every seed-band verdict (12/12).
