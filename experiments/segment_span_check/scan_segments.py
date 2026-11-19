"""
Camera-ready row R7 (reviewer LrHa, item B7): how many 1,024-point blocks does a stored segment hold?
np.savez writes uncompressed arrays, so a segment file's size is fixed by its number of blocks; one directory listing therefore classifies all files without opening them.
The size --> block-count map is read off a random sample of opened files and must be one-to-one before anything else is reported.
Outputs: segments.parquet (one row per stored segment) and block_counts.csv (the shares quoted in the paper).
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

repo = Path(__file__).resolve().parents[2]
seq_dir = repo / "processed" / "sequences"
out_dir = Path(__file__).resolve().parent
name_re = re.compile(r"^TIC(\d+)_s(\d+)_seg(\d+)_run(\d+)\.npz$") # format confirmed on listed files, e.g. TIC0000000358_s38_seg00_run00.npz
sample_n = 2000 # files opened to calibrate size --> block count

# 1. One directory listing: star, sector, segment, run and file size
rows = []
with os.scandir(seq_dir) as entries:
    for entry in tqdm(entries, desc="list npz"):
        match = name_re.match(entry.name)
        if match is None:
            continue
        rows.append({
            "file": entry.name,
            "tic_id": int(match.group(1)),
            "sector": int(match.group(2)),
            "seg_idx": int(match.group(3)),
            "run_idx": int(match.group(4)),
            "size": entry.stat().st_size,
        })
segments = pd.DataFrame(rows)
print("stored segment files:", len(segments), "| stars:", segments["tic_id"].nunique())

# 2. Calibrate file size --> number of stored blocks on a random sample
rng = np.random.default_rng(0)
sample = segments.sample(n=sample_n, random_state=0)
size_to_blocks = {}
for row in tqdm(sample.itertuples(index=False), desc="calibrate", total=sample_n):
    with np.load(seq_dir / row.file) as data:
        n_blocks = int(data["n_windows"])
    size_to_blocks.setdefault(row.size, set()).add(n_blocks)
for size, blocks in size_to_blocks.items():
    assert len(blocks) == 1, f"file size {size} maps to several block counts {blocks}"
calibration = pd.DataFrame({"size": list(size_to_blocks), "n_blocks": [min(b) for b in size_to_blocks.values()]})
print(calibration.sort_values("size").to_string(index=False))

# 3. Every size seen in the full listing must be one the sample calibrated
unknown = set(segments["size"]) - set(calibration["size"])
assert not unknown, f"{len(unknown)} file sizes not covered by the calibration sample: {sorted(unknown)[:5]}"
segments = segments.merge(calibration, on="size", how="left")
segments.to_parquet(out_dir / "segments.parquet", index=False)

# 4. Block counts: every stored segment, and each star's first segment (lowest sector, then segment index)
first = segments.sort_values(["tic_id", "sector", "seg_idx"]).drop_duplicates("tic_id")
count_rows = []
for scope, frame in [("all stored segments", segments), ("first segment per star", first)]:
    shares = frame["n_blocks"].value_counts().sort_index()
    for n_blocks, n in shares.items():
        count_rows.append({"scope": scope, "n_blocks": n_blocks, "model_windows": 4 * n_blocks,
                           "n": int(n), "share": n / len(frame)})
counts = pd.DataFrame(count_rows)
counts.to_csv(out_dir / "block_counts.csv", index=False)
print(counts.to_string(index=False))
print("files with run_idx > 0:", int((segments["run_idx"] > 0).sum()))
