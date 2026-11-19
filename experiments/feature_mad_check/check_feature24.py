"""
Camera-ready row R6 (reviewer LrHa, item B6): is engineered feature 24 (MAD) constant on MAD-normalised flux?
Feature 24 is the median absolute deviation of a star's first stored segment, taken without the 1.4826 factor that the normalisation divides by.
If the stored flux were the whole normalised segment, the feature would equal 1/1.4826 = 0.6745 for every star.
This script reports its spread over the cached raw feature vectors of the rotation pool, and recomputes it from the stored windows for a random sample of stars as a footing check.
The pool-1 cache (exp10_features/subset_features25.parquet) holds standardised columns, so it cannot answer this question and is not read.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from swm.eval.features import FEATURE_NAMES, extract_features
from swm.eval.new_task_extract import index_pool_npz, replay_first_segment

repo = Path(__file__).resolve().parents[2]
out_dir = Path(__file__).resolve().parent
footing_n = 200 # stars re-extracted from npz to confirm the cached column is the feature

# 1. Cached raw feature vectors of the rotation pool (106,284 corpus stars outside pool 1)
shards = sorted((repo / "experiments" / "r8_fullpool" / "features25").glob("shard_*.parquet"))
frames = []
for shard in shards:
    frames.append(pd.read_parquet(shard, columns=["tic_id", "mad"]))
rot_pool = pd.concat(frames, ignore_index=True)

# 2. Footing: feature 24 recomputed from the stored windows of random rotation-pool stars
rng = np.random.default_rng(0)
sample = rng.choice(rot_pool["tic_id"].to_numpy(), size=footing_n, replace=False)
npz_index = index_pool_npz(repo / "processed" / "sequences", set(int(t) for t in sample))
mad_col = FEATURE_NAMES.index("mad")
diffs = []
for tic in sample:
    block = replay_first_segment(npz_index[int(tic)][2], 1024)
    recomputed = extract_features(block.reshape(-1))[mad_col]
    cached = float(rot_pool.loc[rot_pool["tic_id"] == tic, "mad"].iloc[0])
    diffs.append(abs(recomputed - cached))
print("footing: max |recomputed - cached| over", footing_n, "stars =", max(diffs))

# 3. Spread of feature 24
summary_rows = []
for name, values in [("rotation_pool", rot_pool["mad"])]:
    summary_rows.append({
        "population": name,
        "n_stars": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "p01": float(values.quantile(0.01)),
        "p99": float(values.quantile(0.99)),
        "min": float(values.min()),
        "max": float(values.max()),
        "share_equal_0p6745": float((values.sub(1 / 1.4826).abs().lt(1e-6)).mean()),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(out_dir / "feature24_summary.csv", index=False)
print(summary.to_string(index=False))
