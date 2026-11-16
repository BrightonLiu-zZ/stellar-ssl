# X2 — recipe-selection gates re-scored on pool 1's validation split

Camera-ready roadmap row **X2** (decisions D29, D32; reviewer LrHa item B9). Run 2026-11-15.
Manifest: `experiments/configs/exp09_val_recheck.yaml`. Script: `experiments/analyze_exp09_val_recheck.py`.

## Question

The recipe was selected on pool 1's **test** split (D29). Do the same rules, scored on pool 1's
**validation** split, reach the same decision?

Only the scoring split changed. Everything else is frozen: checkpoints, seeds, first-segment bag,
StandardScaler + balanced logistic fitted on train, the `mean` and `mean_std` readouts, gate
thresholds, the VOID rule and the paired-by-seed estimator.

- **Scope:** 40 cells, 226 checkpoints. That is exp05 (10 cells × 4 seeds), exp07 (10 cells × 4–12
  seeds) and exp09 (20 cells × 6–12 seeds).
- **Validation split:** 2,021 stars (pool 1 val, from the shared w256 pack). Test is also 2,021.

## Footing (rule 8) — passed

| check | rows | max abs diff |
|---|---|---|
| probe, test re-run vs every published summary (exp05 `readout_sweep`, `exp07_aux_gap*`, every `exp09_diag_*_ro`, `w6_ref`) | 1,920 | 9.7e-17 |
| artifact severity, test re-run vs every `exp09_impulse*_runs.csv` | 150 | 7.1e-15 |
| Appendix C, test re-run | — | pooled-29 loss-optimal ranks 10 / 27 / 19, cost 0.011 / 0.031 / 0.019 (paper: 10th, 27th, 19th; 0.011–0.031) |

Two estimator details had to be matched before the footing held. Both are recorded in the manifest.

1. **Encoder batching.** exp05's numbers came from `readout_sweep`, which encodes one star per
   forward pass. Batching 4,096 windows moves $\mu$ by up to 3.4e-4 and PR-AUC by up to 5.2e-4.
2. **Probe input memory layout.** The exp07/exp09 numbers went through `DataFrame.to_numpy()`, which
   returns a Fortran-ordered array. The `mean_std` fit (256 collinear columns) is ill-conditioned
   enough that C- vs F-order moves PR-AUC by up to 8.4e-4. Example: `exp07_comb0p3_fbwd` s1 eb gives
   0.767147 with F-order (the published value) and 0.766587 with C-order.

   So the `mean_std` readout has a numerical noise floor of about 1e-3 PR-AUC. The noregress gate's
   2·SE is typically about 0.01.

## Result

### exp09 ladder: the switch rule and both gates (vs `exp07_hann0p3_fbwd`)

| | test (published split) | val (re-check) |
|---|---|---|
| switch fires (G17 ≤ 2.0 ∧ noregress at both readouts ∧ ≥ 2/4 tasks > 2 SE ∧ not VOID) | **no cell** | **no cell** |
| both G9 gates (artifact ≤ 1.5 ∧ noregress, not VOID) | **no cell** | **no cell** |
| VOID | w0p0125, w0p02, w0p025 (12 seeds) | same (read from training curves, split-independent) |
| noregress passes | 9 cells | 6 cells (4 in common: impulse_pen, w0p10, hinge f5p23, hann w0p10) |

**Decision: same.** Pre-registered D32 branch: "selected on test; re-checked on validation, same
recipe". Artifact severity moves by ≤ 0.15 between splits on every cell. No cell crosses either bar.

What does change:

- **The nearest miss.** `dpss_impulse_w0p025_fb0p01` (the floored candidate): severity 1.546 test /
  1.554 val.
  - On test it regresses on pulsators at both readouts, as Appendix D says.
  - On val it **passes** noregress at both readouts.
  - On val it still fails G9-artifact (1.554 > 1.5), and it clears 2 SE on only one task (eb), so
    the switch would not fire.
  - Its stated failure reason therefore holds on test only.
- **The noregress verdict is split-sensitive.** 9 cells pass on test and 6 on val, with 4 in common.
  This fits the record that 6-seed passes did not replicate at 12 seeds three times. The gate sits
  close to its noise level.

### exp05 / exp07 rounds: per-task ranking at `mean` (Appendix C tasks)

| round | task | shipped lineage, test rank | val rank | best on test | best on val |
|---|---|---|---|---|---|
| exp07 | pulsating | hann0p3_fbwd 2/10 | **1/10** | hann0p3_off | hann0p3_fbwd |
| exp07 | eb | 5/10 | 5/10 | comb0p1_fbwd | lpsd0p3_fbwd |
| exp07 | transit | 3/10 | 3/10 | lpsd0p1_fbwd | lpsd0p1_fbwd |
| exp05 | pulsating / eb / transit | best fbwd cell 2 / 1 / 2 | 2 / 2 / 2 | fwd_c1p0 / fbwd_c0p3 / fwd_c1p0 | fwd_c0p3 / fwd_c1p0 / fwd_c1p0 |

The shipped lineage's rank is stable. Spearman correlation of the cell ordering, test vs val:

- eb: +0.90 (exp05) and +0.88 (exp07)
- transit: +0.76 and +0.77
- pulsating: **+0.08 and +0.32**. The pulsator ordering between recipes does not reproduce across
  splits.

### Appendix C on val: loss-optimal recipe vs probe

| scope | split | ρ(−loss, PR-AUC) puls / eb / transit | loss-optimal rank puls / eb / transit | cost of trusting the loss |
|---|---|---|---|---|
| exp05 (10) | test | −0.84 / −0.70 / −0.58 | 8 / 10 / 9 | 0.021–0.035 |
| exp05 (10) | val | −0.18 / −0.82 / −0.75 | 8 / 9 / 10 | 0.022–0.041 |
| exp09 family (19) | test | −0.51 / +0.37 / −0.53 | 17 / 10 / 16 | 0.011–0.031 |
| exp09 family (19) | val | +0.32 / +0.12 / −0.30 | **1** / 3 / 10 | **0.000**–0.013 |
| pooled 29 | test | | 27 / 10 / 19 | 0.011–0.031 |
| pooled 29 | val | | **1** / 3 / 10 | **0.000**–0.013 |

- **exp05 round:** the anti-correlation survives on val. The dynamics-free recipe is among the three
  worst probes on every task.
- **Final-recipe family:** it does not survive. On val the loss-optimal recipe (`aux_none`, no
  spectral term) is the **best** pulsator probe and 3rd on eb.
- The Discussion sentence "On none of the three did the variant with the lowest validation
  reconstruction loss score best; choosing it would have cost 0.011–0.031 PR-AUC" is therefore
  true on test only.
- `aux_none` stays ineligible under the gates on both splits: it regresses on rotation.

## Paper consequences (for the R9 / Discussion / App. C / App. D edits)

1. **R9 sentence (same-recipe branch):** "The pre-training recipe was chosen on the first pool's test
   split for its three tasks; re-scoring every gate of that choice on the validation split leaves
   the decision unchanged — no variant clears both acceptance criteria on either split."
2. **Discussion "Pick the encoder by a held-out probe" + App. C.** The paper must not keep "on none
   of the three … cost 0.011–0.031" as a claim that holds on any split. **Decided 2026-11-15 (user):
   cut the numbers from the body.**
   - The body keeps only the qualitative message, scoped to the earlier round, where it holds on
     both splits: there, the lowest-loss recipe is among the three worst probes on every task.
   - Every number, for both splits, moves to App. C. That includes the final-recipe family, where
     the loss pick ranks 17 / 10 / 16 of 19 on test and 1 / 3 / 10 on val.
3. **App. D nearest-miss sentence** ("regressed on pulsators at both readouts"): add "on the test
   split; on validation it does not regress, but stays above the artifact gate (1.55)".

## Files

- `runs/<cell>__seed<N>.json`: one per checkpoint, the resume marker.
- `probe.csv`, `artifact.csv`: long tables, val and test.
- `footing.csv`: every re-run row against its published counterpart.
- `gates_exp09.csv`, `ranks_exp05_exp07.csv`, `valloss_appc.csv`.
- `run.log`: first pass. Its exp07/exp09 records were deleted after the layout fix; only the exp05
  records from it are kept.
- `smoke_layout.log`, `run2.log` (exp07/exp09 re-run + gates), `gates.log` (final gates).

## Reproduce

```bash
# swm env, repo root; resumable (re-run skips checkpoints whose JSON exists)
PYTHONPATH=src PYTHONUNBUFFERED=1 python experiments/analyze_exp09_val_recheck.py --stages run gates 2>&1 | tee experiments/exp09_val_recheck/run.log
```
