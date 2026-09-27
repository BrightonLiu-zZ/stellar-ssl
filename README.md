# stellar-ssl

Code for the paper *What does self-supervision add to engineered features for stellar light
curves?* (source `paper/main.tex`, compiled `paper/main.pdf`).

We pre-train a convolutional variational encoder with a GRU latent-dynamics objective on unlabelled
TESS 2-minute PDCSAP light curves, freeze it, append its per-star latent mean µ to 25 engineered
periodogram and time-domain features, and read the combination out with one fixed linear probe on
ten tasks (five variability classes, four asteroseismic targets, rotation period). The paper's
Table 1 compares that fusion against the features alone, µ alone, the same fusion built on an
untrained encoder, and two networks trained end-to-end on the same labelled stars.

The tag [`arxiv-v1`](https://github.com/BrightonLiu-zZ/stellar-ssl/tree/arxiv-v1) is the state of
the code and score tables the arXiv version reports.

## What you can regenerate from a fresh clone

| Paper artefact | Command (`swm` env, repo root) | Reads |
|---|---|---|
| Table 1, Figure 1, Appendix D and I tables | `python experiments/plot_ml4ps_scorecard.py` | `experiments/{f1_fusion_scorecard,f1_xgb_control,c1c2_supervised,unseen_pretraining}/*.csv` |
| Appendix J table (rotation pool) | `python experiments/plot_rotation_pool_appendix.py` | `experiments/rotation_pool/**/*.csv`, `paper/build/table1_data.csv` |

Both write into `paper/tables/`, `paper/figures/` and `paper/build/`, and reproduce the committed
files exactly (the figure differs only in its PDF creation date). The first script also prints the
headline counts, e.g. `fusion beats features beyond 2*SE on 7 of 10`.

The other printed numbers come with their plotted or tabulated values but not with the inputs
needed to recompute them:

| Paper artefact | Published values | Needs (not in the repository) |
|---|---|---|
| Appendix G paired bootstrap | `experiments/paper_bootstrap/paired_bootstrap.csv` | per-star prediction parquets |
| Appendix J bootstrap sentence | `experiments/rotation_pool/bootstrap/paired_bootstrap.csv` | per-star prediction parquets |
| Validation loss vs probe score (Figure 2, Appendix E) | `paper/build/figB_data.csv`, `figB_exp09_data.csv` | training-curve dumps of the exp05 and exp09 pre-training runs |
| Reconstruction figure (Appendix F) | `paper/build/figD_recon_data.csv` | checkpoints and the light-curve corpus |

The score tables hold every arm and seed: `*_probe.csv` is one row per (task, arm, seed, readout),
`*_absolute.csv` the per-arm score (mean, SD and 2 SE over seeds), `*_summary.csv` the per-task
contrasts such as fusion minus features, with their 2 SE band and the per-seed deltas.

## Environments

| Env | File | Used for |
|---|---|---|
| `astro` | `environment.yml` | Stage 0: download, windowing, label cross-match (CPU) |
| `swm` | `environment-swm.yml` | everything under `src/swm/`, the `experiments/` analysis scripts, the paper's tables and figures; PyTorch 2.5.1 + CUDA 12.1 |

```bash
conda env create -f environment.yml
conda env create -f environment-swm.yml
conda activate swm
export PYTHONPATH=src          # PowerShell: $env:PYTHONPATH = "src"
python -m pytest src/swm/tests -q
```

On a fresh clone the tests that need the corpus, labels or cached µ skip themselves.

## Full pipeline, from raw data

This is the order the committed code runs in. It is a record, not a one-command pipeline: it needs
the TESS corpus (downloaded by step 2), a CUDA GPU for pre-training, and the edits listed under
*Before running* below.

| # | Step | Command | Env |
|---|---|---|---|
| 1 | SPOC sector map, Tmag < 10 | `python src/download_cdpp.py` | astro |
| 2 | Download PDCSAP light curves, segment at time gaps, cut NaN-free windows | `python src/build_sequences_bulk.py --sectors 1-101 --workers 16` | astro |
| 3 | Variability labels (TOI, eclipsing binaries, pulsators, rotation, flares) | `python src/build_variability_labels.py` | astro |
| 4 | Asteroseismic and other new-task labels | `python src/labels/build_new_task_labels.py` | astro |
| 5 | Pool 1: label-enriched subset and 70/15/15 split | `python -m swm.data.subset` | swm |
| 6 | Pack pool 1 into 256-cadence windows | `python -m swm.data.pack +experiment=exp01_window256_seq16` | swm |
| 7 | Pool 2 (the five new-task probes) | `python -m swm.eval.new_task_pool` | swm |
| 8 | Pre-train the shipped recipe, seeds 0-5, plus its no-dynamics twin | `python -m swm.train +experiment=exp07/exp07_hann0p3_fbwd variant=B seed=<s>` (and `exp07/exp07_hann0p3_off`); queue: `experiments/run_exp07_aux_factorization.ps1` | swm |
| 9 | Frozen-encoder µ, pool 2 and pool 1 | `python -m swm.eval.new_task_extract ...`, `python experiments/build_subset_mu_cache.py ...` | swm |
| 10 | Linear fusion scorecard (features / µ / fusion / untrained) | `python experiments/analyze_f1_fusion_scorecard.py` | swm |
| 11 | Supervised baselines C1, C2 | `experiments/run_c1c2_supervised_baselines.ps1`, then `python experiments/analyze_c1c2_supervised.py` | swm |
| 12 | Stars unseen in pre-training | `python experiments/dump_paper_linear_scores.py`, then `python experiments/analyze_unseen_rescore.py` | swm |
| 13 | Rotation pool (pool 3) | `python -m swm.eval.rotation_pool`, then `python experiments/analyze_rotation_pool.py`, `python experiments/report_rotation_pool.py` | swm |
| 14 | Tables and figures | the two commands at the top of this page | swm |

The untrained-encoder control is not trained; it is a seeded random initialisation built inside the
extraction step. The 25 engineered features are computed by `swm.eval.features` and cached on first
use. Checkpoint selection is `best_recon_aux` throughout. Per-experiment records, including the
pre-registered gates the shipped recipe had to pass, are the `experiments/*README.md` files and the
single-file manifests in `experiments/configs/`.

### Before running

- `src/swm/configs/paths/local.yaml` hard-codes `repo_root`; override it on the command line
  (`paths.repo_root=$PWD`) or edit the file.
- The `experiments/run_*.ps1` queues are generated for the author's machine and set `$py` and
  `$repo` to local paths in their first lines; edit those two lines.
- Two catalogues are not fetched by any script and must be placed by hand: the TARS rotation
  table (`data/tars_table_2.feather`, Zenodo 10.5281/zenodo.19917941) and the flare catalogue
  (`data/Table3_flare_catalog.csv`, Zenodo 10.5281/zenodo.14179313). Step 4 reads its catalogues from
  `labels/external/`, which you populate from the papers cited in the paper's §2.
- Four steps were run with arguments that no committed file records: the pool-2 µ extraction for
  the shipped cells, the XGBoost control (`analyze_f1_fusion_scorecard.py --families xgb`), and the
  rotation-pool bootstrap and supervised data cache. Their outputs are the published CSVs above.

## Not in the repository

| What | Size | Why |
|---|---|---|
| Light-curve corpus (`processed/sequences/`) and packed windows | 14.6 GB in 416,729 `.npz` files, before packing | regenerable from MAST with steps 1-2 and 6 |
| Label catalogues (`labels/`, `data/`) | 44 MB for `labels/` | third-party catalogues; regenerable with steps 3-4 |
| Checkpoints, µ caches, per-star predictions (`experiments/**`) | 113 GB in total | size |
| Training logs | — | Weights & Biases, not published |

## Layout

```
src/
  download_cdpp.py, build_sequences_bulk.py, build_variability_labels.py   Stage 0 (astro)
  labels/build_new_task_labels.py                                           new-task labels
  qc/                        label audits (not on the paper's path)
  swm/                       model, training, evaluation (Hydra configs in swm/configs/)
  notebooks/                 diagnostics notebooks, one per experiment
experiments/
  configs/*.yaml             one manifest per experiment
  analyze_*.py, plot_*.py    analysis and paper figures
  *README.md                 per-experiment records
  <score-table dirs>/*.csv   the published score tables
paper/
  main.tex, refs.bib, figures/, tables/    paper source (tables and figures generated)
  build/                     rebuild notes, reference checkers, provenance CSVs
```

## License

MIT, see `LICENSE`.
