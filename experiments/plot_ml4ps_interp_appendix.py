"""Camera-ready X3: what the 128 latent columns track, as one appendix table (roadmap D25, D33).

For the shipped encoder (exp07_hann0p3_fbwd, six pre-training seeds) on pool 1, each latent dimension
of the star-level mean-pooled mu is regressed on the 25 engineered features with the F-B instrument
(experiments/analyze_exp10_fb_mu_predictability.py: StandardScaler + RidgeCV with a per-target alpha,
fit on pool-1 train, scored as R^2 on pool-1 test). Per seed the dimensions are ranked by their share
of mu's total test variance, so rows are aligned by RANK, not by coordinate (mu coordinates are
arbitrary across seeds). Per row group:

  share        fraction of mu's total test variance held by those dims
  R^2 (all)    variance-weighted test R^2 from all 25 features: 1 - sum SSE / sum SST over the group
  unexplained  sum SSE of the group / sum SST of all 128 dims (the column adds up to the all-128 row)
  family R^2   the same R^2 with only one feature family as regressors:
                 peaks       ls_* (9: top-3 periodogram peak frequencies, powers, ratios, harmonic)
                 bands       psd_band0-7, spectral_entropy, spectral_centroid_log10 (10)
                 dispersion  skew, kurtosis, p2p_scatter_ratio, depth_5_95, mad, iqr (6)
  top feature  single-dim rows: the feature with the largest |Pearson r| with that dim on test, and in
               how many of the six seeds it is that feature

Spread is the standard deviation over the six encoder seeds (what varied: pre-training seed only; the
stars, features, split and readout are fixed). An untrained-encoder row (ONE initialisation, the
historical seed-0 init cached for exp08) is appended as a comparison: architecture vs training.

FOOTING: per-dim R^2 and test variance must reproduce experiments/exp10_forensics/fb_predictability/
fb_per_dim_r2.csv (population subset) before anything is written.

Run (repo root, swm env, PYTHONPATH=src; CPU, < 1 min):
    PYTHONUNBUFFERED=1 python experiments/plot_ml4ps_interp_appendix.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_exp08_menu_channel import align_features, stacked  # noqa: E402
from analyze_exp10_fb_mu_predictability import per_dim_r2, pooled_mu, ridge_predictions  # noqa: E402
from swm.eval.features import FEATURE_NAMES  # noqa: E402
from swm.eval.new_task_ceiling import cached_subset_features  # noqa: E402

log = logging.getLogger("interp_appendix")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "interp_table"
TABLE = ROOT / "paper" / "tables" / "table_interp.tex"
FB_REF = ROOT / "experiments" / "exp10_forensics" / "fb_predictability" / "fb_per_dim_r2.csv"
SEEDS = [0, 1, 2, 3, 4, 5]

FAMILIES = {
    "peaks": [f for f in FEATURE_NAMES if f.startswith("ls_")],
    "bands": [f for f in FEATURE_NAMES if f.startswith("psd_band") or f.startswith("spectral_")],
    "dispersion": ["skew", "kurtosis", "p2p_scatter_ratio", "depth_5_95", "mad", "iqr"],
}
# rank groups, 1-based inclusive: one row per dominant dim, then the tail
# The 3rd-ranked dim is not a stable object across seeds (R^2 0.24 +/- 0.20, top feature agrees in 3/6),
# so it is pooled with the next seven rather than given a feature label it does not hold.
GROUPS = [("largest", 1, 1), ("2nd", 2, 2), ("3rd--10th", 3, 10), ("11th--128th", 11, 128),
          ("all 128", 1, 128)]
FEATURE_LABEL = {"p2p_scatter_ratio": "point-to-point scatter", "skew": "skewness", "kurtosis": "kurtosis",
                 "depth_5_95": "5--95\\% depth", "mad": "MAD", "iqr": "IQR",
                 "spectral_entropy": "spectral entropy", "spectral_centroid_log10": "spectral centroid",
                 **{f"psd_band{k}": f"band-{k} power" for k in range(8)}}


def one_arm(arm: str, feats: dict) -> pd.DataFrame:
    """Per-dim test variance, R^2 from all features and from each family, and the top single feature."""
    mu = pooled_mu("subset", arm)
    aligned = align_features(feats, mu)
    x_train, x_test = stacked(aligned, "train"), stacked(aligned, "test")   # (n_train, 25), (n_test, 25)
    y_train, y_test = stacked(mu, "train"), stacked(mu, "test")             # (n_train, 128), (n_test, 128)
    r2, sse, sst = per_dim_r2(y_test, ridge_predictions(x_train, y_train, x_test))
    dims = pd.DataFrame({"dim": np.arange(y_test.shape[1]), "mu_test_var": sst / len(y_test),
                         "r2_all": r2, "sse_all": sse, "sst": sst})
    for family, names in FAMILIES.items():
        cols = [FEATURE_NAMES.index(n) for n in names]
        r2_f, sse_f, _ = per_dim_r2(y_test, ridge_predictions(x_train[:, cols], y_train, x_test[:, cols]))
        dims[f"r2_{family}"] = r2_f
        dims[f"sse_{family}"] = sse_f
    xz = (x_test - x_test.mean(0)) / x_test.std(0)
    yz = (y_test - y_test.mean(0)) / np.where(y_test.std(0) > 0, y_test.std(0), np.nan)
    corr = np.abs(xz.T @ yz / len(y_test))                                  # (25, 128)
    dims["top_feature"] = [FEATURE_NAMES[i] for i in np.nanargmax(np.nan_to_num(corr, nan=-1), axis=0)]
    dims["top_abs_r"] = np.nanmax(corr, axis=0)
    for i, name in enumerate(FEATURE_NAMES):   # kept so single-dim rows can average |r| over seeds
        dims[f"absr_{name}"] = corr[i]
    dims["share"] = dims.sst / dims.sst.sum()
    dims["rank"] = dims.share.rank(ascending=False, method="first").astype(int)
    return dims.sort_values("rank").reset_index(drop=True)


def footing(per_dim: pd.DataFrame) -> None:
    ref = pd.read_csv(FB_REF)
    ref = ref[ref.population == "subset"]
    merged = per_dim[per_dim.arm == "trained"].merge(ref, on=["seed", "dim"], suffixes=("", "_ref"))
    d_r2 = float((merged.r2_all - merged.r2_ridge).abs().max())
    d_var = float(((merged.mu_test_var - merged.mu_test_var_ref).abs() / merged.mu_test_var_ref).max())
    log.info(f"FOOTING vs F-B fb_per_dim_r2.csv: {len(merged)} dims, max |dR2| {d_r2:.2e}, "
             f"max relative |dvar| {d_var:.2e}")
    assert len(merged) == 128 * len(SEEDS), "F-B reference does not cover every (seed, dim)"
    assert d_r2 < 1e-6 and d_var < 1e-5, "X3 does not reproduce the published F-B per-dim numbers"


def group_rows(per_dim: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each rank group per seed (variance-weighted), keeping one row per (arm, seed, group)."""
    rows = []
    for (arm, seed), d in per_dim.groupby(["arm", "seed"]):
        total = d.sst.sum()
        for name, lo, hi in GROUPS:
            g = d[(d["rank"] >= lo) & (d["rank"] <= hi)]
            row = {"arm": arm, "seed": seed, "row": name, "first_rank": lo, "last_rank": hi,
                   "share": g.sst.sum() / total, "r2_all": 1 - g.sse_all.sum() / g.sst.sum(),
                   "unexplained": g.sse_all.sum() / total,
                   "top_feature": g.top_feature.iloc[0] if lo == hi else "",
                   "top_abs_r": g.top_abs_r.iloc[0] if lo == hi else np.nan}
            for family in FAMILIES:
                row[f"r2_{family}"] = 1 - g[f"sse_{family}"].sum() / g.sst.sum()
            if lo == hi:
                for name_f in FEATURE_NAMES:
                    row[f"absr_{name_f}"] = float(g[f"absr_{name_f}"].iloc[0])
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(groups: pd.DataFrame) -> pd.DataFrame:
    """Mean and seed sd per (arm, group); top feature = modal feature with its seed count."""
    num = ["share", "r2_all", "unexplained", "r2_peaks", "r2_bands", "r2_dispersion", "top_abs_r"]
    out = []
    for (arm, name), g in groups.groupby(["arm", "row"], sort=False):
        row = {"arm": arm, "row": name, "n_seeds": len(g)}
        for col in num:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_sd"] = float(g[col].std(ddof=1)) if len(g) > 1 else np.nan
        if g.top_feature.iloc[0]:
            counts = g.top_feature.value_counts()
            row["top_feature"], row["top_feature_seeds"] = counts.index[0], int(counts.iloc[0])
            row["top_feature_all_seeds"] = ";".join(g.top_feature)
            # several features are collinear (scatter / entropy / centroid all measure the high-frequency
            # share), so the table names the two leading features by seed-mean |r|, not one winner
            mean_r = pd.Series({f: g[f"absr_{f}"].mean() for f in FEATURE_NAMES}).sort_values(ascending=False)
            for k in (1, 2, 3):
                row[f"lead{k}_feature"], row[f"lead{k}_abs_r"] = mean_r.index[k - 1], float(mean_r.iloc[k - 1])
        out.append(row)
    return pd.DataFrame(out)


def cell(mean: float, sd: float, digits: int = 2) -> str:
    return f"${mean:.{digits}f}$" if np.isnan(sd) else f"${mean:.{digits}f} \\pm {sd:.{digits}f}$"


def write_table(summary: pd.DataFrame) -> None:
    trained = summary[summary.arm == "trained"].set_index("row")
    lines = ["% GENERATED by experiments/plot_ml4ps_interp_appendix.py -- do not hand-edit",
             "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lrrrrrrl}",
             "\\toprule",
             " & & \\multicolumn{2}{c}{all 25 features} & \\multicolumn{3}{c}{one family, $R^2$} & \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}",
             "$\\mu$ dims (by variance) & share & $R^2$ & unexplained & peaks & bands & dispersion & leading features, $|r|$ \\\\",
             "\\midrule"]
    for name, _, _ in GROUPS:
        if name == "all 128":
            lines.append("\\midrule")
        r = trained.loc[name]
        top = "--"
        if isinstance(r.get("lead1_feature"), str):
            parts = []
            for k in (1, 2):
                f = r[f"lead{k}_feature"]
                parts.append(f"{FEATURE_LABEL.get(f, f.replace('_', ' '))} {r[f'lead{k}_abs_r']:.2f}")
            top = ", ".join(parts)
        share = "$1$" if name == "all 128" else cell(r.share_mean, r.share_sd)
        lines.append(" & ".join([name, share, cell(r.r2_all_mean, r.r2_all_sd),
                                 cell(r.unexplained_mean, r.unexplained_sd, 3),
                                 cell(r.r2_peaks_mean, r.r2_peaks_sd), cell(r.r2_bands_mean, r.r2_bands_sd),
                                 cell(r.r2_dispersion_mean, r.r2_dispersion_sd), top]) + " \\\\")
    u = summary[summary.arm == "untrained"].set_index("row")
    if len(u):
        lines.append("\\midrule")
        for name, label in (("largest", "untrained: largest"), ("all 128", "untrained: all 128")):
            r = u.loc[name]
            share = "$1$" if name == "all 128" else cell(r.share_mean, np.nan)
            lines.append(" & ".join([label, share, cell(r.r2_all_mean, np.nan), cell(r.unexplained_mean, np.nan, 3),
                                     cell(r.r2_peaks_mean, np.nan), cell(r.r2_bands_mean, np.nan),
                                     cell(r.r2_dispersion_mean, np.nan), "--"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    TABLE.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    feats = cached_subset_features(ROOT / "experiments" / "exp01_window256_seq16" / "packed")
    arms = [("trained", s, f"hann0p3_fbwd_s{s}") for s in SEEDS] + [("untrained", 0, "untrained")]
    frames = []
    for arm, seed, name in tqdm(arms, desc="arms", total=len(arms)):
        frames.append(one_arm(name, feats).assign(arm=arm, seed=seed))
    per_dim = pd.concat(frames, ignore_index=True)
    footing(per_dim)

    groups = group_rows(per_dim)
    summary = summarise(groups)
    OUT.mkdir(parents=True, exist_ok=True)
    per_dim.to_csv(OUT / "interp_per_dim.csv", index=False)
    groups.to_csv(OUT / "interp_groups_per_seed.csv", index=False)
    summary.to_csv(OUT / "interp_rows.csv", index=False)
    write_table(summary)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.drop(columns=[c for c in summary if c.endswith("_sd")] + ["top_feature_all_seeds"],
                           errors="ignore").round(3).to_string(index=False))
    log.info(f"wrote {OUT}/interp_{{per_dim,groups_per_seed,rows}}.csv and {TABLE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
