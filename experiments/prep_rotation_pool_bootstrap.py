"""Shape the pool-3 outputs into what analyze_paper_bootstrap.py reads, so the same bootstrap runs unchanged.

Writes into experiments/rotation_pool/bootstrap/:
    linear_star_scores.parquet   the linear-family per-star test scores (block pool3, readout `mean`),
                                 with the `shape` column the bootstrap expects; untrained inits are
                                 seeds, and the bootstrap's "min seed" rule picks init 0, the paper's arm
    table1_data.csv              the paired-by-seed deltas + 2 SE the bootstrap prints beside its CIs
    dynamics_ablation.csv        fusion fbwd - off, same role

Run after analyze_rotation_pool.py (linear) and analyze_c1c2_supervised.py on the pool-3 manifest:
    python experiments/prep_rotation_pool_bootstrap.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1] / "experiments" / "rotation_pool"
out = root / "bootstrap"
SHAPE = {"rotation": "detection", "rotation_period": "regression"}


def main() -> int:
    out.mkdir(parents=True, exist_ok=True)
    stars = pd.read_parquet(root / "star_scores.parquet")
    stars = stars[(stars["block"] == "pool3") & (stars["readout_family"] == "linear")].copy()
    stars["shape"] = stars["task"].map(SHAPE)
    stars[["arm", "family", "seed", "arm_set", "task", "shape", "tic_id", "y", "score"]].to_parquet(
        out / "linear_star_scores.parquet", index=False)

    summary = pd.read_csv(root / "summary.csv")
    head = summary[(summary["block"] == "pool3") & (summary["readout"] == "mean")
                   & (summary["readout_family"] == "linear")]
    # analyze_c1c2_supervised.py differences against the POOL-1 fusion score (its reference is hardwired to
    # f1_fusion_scorecard), so its delta/recovery columns are not usable here; the pool-3 fusion absolute is
    # taken from absolute.csv instead. Unpaired contrast: SEs add in quadrature (D-C1.7).
    c1c2 = pd.read_csv(root / "c1c2" / "c1c2_absolute.csv")
    absolute = pd.read_csv(root / "absolute.csv")
    fusion = absolute[(absolute["block"] == "pool3") & (absolute["readout"] == "mean")
                      & (absolute["readout_family"] == "linear") & (absolute["arm_set"] == "features_plus_mu")
                      & (absolute["family"] == "hann0p3_fbwd")].set_index("task")
    rows, ablation = [], []
    for task in SHAPE:
        sel = head[head["task"] == task]
        gain = sel[(sel["contrast"] == "fusion_minus_features") & (sel["family"] == "hann0p3_fbwd")].iloc[0]
        untr = sel[(sel["contrast"] == "fusion_minus_features") & (sel["family"] == "untrained")].iloc[0]
        off = sel[sel["contrast"] == "fusion_fbwd_minus_off"].iloc[0]
        row = {"task": task, "d_features": gain["delta_mean"], "d_features_2se": gain["delta_2se"],
               "d_features_untrained": untr["delta_s0"]}  # init 0 = the paper's single untrained arm
        for arm, col in [("conv_supervised", "c1"), ("mlp_raw", "c2")]:
            c = c1c2[(c1c2["task"] == task) & (c1c2["arm"] == arm)].iloc[0]
            row[f"d_{col}"] = float(fusion.loc[task, "score_mean"] - c["score_mean"])
            row[f"d_{col}_2se"] = float(np.hypot(fusion.loc[task, "score_2se"], c["score_2se"]))
        rows.append(row)
        ablation.append({"task": task, "drop": off["delta_mean"], "drop_2se": off["delta_2se"]})
    pd.DataFrame(rows).to_csv(out / "table1_data.csv", index=False)
    pd.DataFrame(ablation).to_csv(out / "dynamics_ablation.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
