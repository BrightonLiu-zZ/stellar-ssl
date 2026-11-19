# R7: how many blocks does a stored segment hold?

Camera-ready roadmap row **R7** (reviewer LrHa, item B7). Run 2026-11-16. Script: `scan_segments.py` (astro env).

## Method

`np.savez` writes uncompressed arrays, so a stored segment's file size is fixed by its number of
1,024-point blocks. One directory listing of `processed/sequences/` therefore classifies every file
without opening it.

The size → block-count map was read off 2,000 randomly chosen files that were opened. It is
one-to-one: 34,548 bytes means 4 blocks, 42,740 bytes means 5. The script asserts that every size in
the full listing is covered by that map.

## Result (`block_counts.csv`)

| scope | 4 blocks (16 windows, 5.69 d) | 5 blocks (20 windows, 7.11 d) |
|---|---|---|
| all 416,729 stored segments | 94.0 % | 6.0 % |
| first segment of each of 119,754 stars | **94.8 %** | 5.2 % |

- No stored segment holds 6 or more blocks.
- No file has `run_idx > 0`. This matches `gap_threshold = 1`: a segment ends at every missing
  cadence, so it contains no NaN and is never split into runs.
- The pool-1 pack alone gives 95.1 % four (12,811 of 13,470). The roadmap's 95.2 % was not
  reproduced.

## Not established

**Why no segment reaches six blocks.** The roadmap guessed that a segment is bounded by one
spacecraft orbit between downlinks. That was not tested: the FITS files are gone and the paper gives
the descriptive answer only (user's choice, 2026-11-16).

One counter-indication: TIC 358, sector 38. Its stored `seg00` is followed directly by `seg16`, so
15 short segments in between were never stored. Quality-flagged cadences therefore also break
segments, not only downlinks.

## Reproduce

```bash
PYTHONUNBUFFERED=1 /c/Users/user1/.venvs/astro/Scripts/python.exe experiments/segment_span_check/scan_segments.py 2>&1 | tee experiments/segment_span_check/scan.log
```
