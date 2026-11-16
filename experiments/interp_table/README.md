# X3 — what the 128 latent columns track (interpretation table)

Camera-ready roadmap row **X3** (decisions D25, D33; reviewer gCxF item D1). Built 2026-11-15.
CPU only, < 1 min, no new encoder pass.

## What was measured

- **Arm:** the shipped encoder `exp07_hann0p3_fbwd`, six pre-training seeds (0–5). One comparison
  row uses the **untrained encoder, one initialisation** (the historical seed-0 init cached for exp08).
- **Population:** pool 1 (the 13,470-star pre-training subset). Fit on train (9,428), scored on test
  (2,021).
- **Representation:** star-level mean-pooled $\mu$ over the first segment (the probe's readout).
- **Instrument:** the F-B regression from `experiments/analyze_exp10_fb_mu_predictability.py`. Each
  of the 128 dimensions is regressed on the 25 engineered features with StandardScaler + `RidgeCV`
  (per-target alpha). The score is test $R^2$.
- **Row alignment:** within each seed, dimensions are ranked by their share of $\mu$'s total test
  variance. Rows are rank groups, because $\mu$ coordinates are arbitrary across seeds. (The
  handoff's "dim 51" is the seed-0 coordinate of the rank-1 dimension; in other seeds it is 46, 28,
  73, 28 and 24.)
- **Columns:**
  - share: fraction of total test variance.
  - $R^2$: variance-weighted over the group, $1-\sum \mathrm{SSE}/\sum \mathrm{SST}$.
  - unexplained: group SSE divided by the total SST of all 128 dimensions, so the column sums to the
    all-128 row.
  - Family $R^2$: the same regression with only one feature family. The families are periodogram
    peaks (9 `ls_*` features), band powers (8 bands plus spectral entropy and centroid, 10 features)
    and dispersion (skew, kurtosis, point-to-point scatter, 5–95 % depth, MAD, IQR; 6 features).
  - Leading features: the two features with the largest seed-mean $|r|$ against the dimension on
    test.
- **Spread:** ± is the standard deviation over the six pre-training seeds. Only the seed varied:
  stars, features, split and readout are fixed.

## Footing

The per-dimension $R^2$ and test variance reproduce
`experiments/exp10_forensics/fb_predictability/fb_per_dim_r2.csv` (population `subset`). The check
covers 768 (seed, dim) pairs: max $|\Delta R^2|$ is 1.1e-16 and max relative $|\Delta\,\mathrm{var}|$
is 5.6e-8. The script asserts this before writing.

## Result (`paper/tables/table_interp.tex`)

| $\mu$ dims (by variance) | share | $R^2$, 25 features | unexplained | peaks | bands | dispersion | leading features ($|r|$) |
|---|---|---|---|---|---|---|---|
| largest | 0.82 ± 0.02 | 0.99 ± 0.00 | 0.009 | 0.60 | 0.98 | 0.96 | point-to-point scatter 0.97, spectral entropy 0.95 |
| 2nd | 0.03 ± 0.01 | 0.49 ± 0.07 | 0.017 | 0.10 | 0.15 | 0.38 | 5–95 % depth 0.53, skewness 0.49 |
| 3rd–10th | 0.03 ± 0.01 | 0.46 ± 0.08 | 0.017 | 0.12 | 0.34 | 0.17 | — |
| 11th–128th | 0.11 ± 0.01 | 0.53 ± 0.04 | 0.053 | 0.15 | 0.42 | 0.13 | — |
| **all 128** | 1 | **0.90 ± 0.00** | **0.096** | 0.52 | 0.87 | 0.82 | — |
| untrained (1 init): largest | 0.06 | 0.95 | 0.003 | 0.38 | 0.68 | 0.86 | — |
| untrained (1 init): all 128 | 1 | 0.82 | 0.183 | 0.40 | 0.67 | 0.74 | — |

Reading it:

1. **Where the variance sits.** In every seed, one direction holds 82 % of $\mu$'s variance. The
   features predict it almost exactly ($R^2$ 0.99). It tracks the high-frequency share of the
   variability, and three collinear features measure that share about equally: point-to-point
   scatter, spectral entropy and spectral centroid ($|r|$ 0.97 / 0.95 / 0.95). The table therefore
   names two features, not a single winner.
2. **The second direction** (3.5 %) tracks amplitude and asymmetry: 5–95 % depth and skewness, a
   near-tie at $|r| \approx 0.5$. The features predict about half of it.
3. The rank-3 dimension is not a stable object across seeds. Its $R^2$ is 0.24 ± 0.20, and its top
   feature agrees in only 3 of 6 seeds. It is pooled into the 3rd–10th row instead of getting a
   label.
4. **Residual.** The features predict 90 % of $\mu$'s test variance. The 9.6 % they cannot predict
   sits mostly in the 118 low-variance dimensions (5.3 of the 9.6 points).
5. **Untrained comparison (one initialisation).** The untrained encoder spreads its variance (6 %
   in the largest direction), and the features predict less of it overall (82 %). Training
   concentrates variance onto the scatter direction and makes $\mu$ *more* feature-predictable
   overall. Caveat: this rests on one init. X1 (six inits) was dropped for the camera-ready.

## Draft paper text (NOT yet in `paper/main.tex`; for review)

**New appendix, "What the latent dimensions track"** (table + three sentences):

> We regress each of the 128 dimensions of the shipped encoder's star-level $\mu$ on the 25
> engineered features (ridge regression fitted on the first pool's training stars, $R^2$ on its test
> stars; Table X). In every pre-training seed one direction carries $82 \pm 2\,\%$ of $\mu$'s
> variance; the features predict it almost exactly ($R^2 = 0.99$), and it tracks the share of the
> variability at high frequency (point-to-point scatter, spectral entropy; $|r| = 0.97$, $0.95$).
> Over all 128 dimensions the features predict $90\,\%$ of $\mu$'s variance; under a linear readout
> $\mu$ can add only what is not a linear function of the features, so every gain in
> Table~\ref{tab:scorecard} comes from the remaining tenth, most of which is spread over the
> low-variance dimensions.

**Caption:**

> What the latent dimensions of the shipped encoder track. Dimensions are ranked within each
> pre-training seed by their share of $\mu$'s test variance; $R^2$ is the variance-weighted test
> $R^2$ of a ridge regression on all 25 features or on one feature family (periodogram peaks: 9;
> band powers: 10; dispersion: 6); "unexplained" is the variance the features cannot predict, as a
> fraction of $\mu$'s total, so the column sums to the last trained row. Mean $\pm$ standard
> deviation over six pre-training seeds. The untrained rows are one initialisation.

**Discussion pointer (one sentence):**

> Most of $\mu$'s variance lies along one direction that the engineered features already predict
> (Appendix X); what $\mu$ adds under a linear readout is the tenth of its variance they cannot.

Notes for the edit:

- The "every gain comes from the remaining tenth" clause is a property of linear fusion, not a
  measurement. Fusion on [features, $\mu$] can use only the part of $\mu$ outside the features'
  linear span. Ridge $R^2$ approximates that span on test. Keep the wording "can add only".
- The untrained row is optional (stretch S2). Drop it from the table if page or appendix space is
  tight, or if one init is judged too weak after the X1 decision.

## Reproduce

```bash
# swm env, repo root
PYTHONPATH=src PYTHONUNBUFFERED=1 python experiments/plot_ml4ps_interp_appendix.py
```

Outputs: `interp_per_dim.csv` (every seed × dim, all columns), `interp_groups_per_seed.csv`,
`interp_rows.csv` (the table's numbers with sd), `paper/tables/table_interp.tex`, `run.log`.
