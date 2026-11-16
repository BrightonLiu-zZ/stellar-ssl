"""Camera-ready X2: re-score every recipe-selection gate on pool 1's VALIDATION split.

The recipe was selected on pool 1's TEST split (roadmap D29). This script re-runs the same selection
rules with the scoring split moved to validation, and nothing else changed: same checkpoints, same
seeds, same first-segment bag, same StandardScaler + balanced logistic probe fitted on train, same
gate thresholds, same VOID rule. The manifest experiments/configs/exp09_val_recheck.yaml holds every
cell, seed and threshold; the pre-registered outcome branches are written there too.

Stages:
  run    one pass per checkpoint: encode the first-segment bags of train / val / test, fit the probe on
         train and score val AND test (test is the footing), and measure the within-window artifact
         profile on every val and test window. One JSON per checkpoint under runs/ is both the result
         and the resume marker, so an interrupted run restarts where it stopped.
  gates  footing first (the test re-run must reproduce the published probe and artifact numbers to
         1e-4); then the exp09 gates on val for every cell, the exp05/exp07 per-task rankings, and the
         Appendix C loss-vs-probe statistics on val.

Run (repo root, swm env, PYTHONPATH=src):
    PYTHONUNBUFFERED=1 python experiments/analyze_exp09_val_recheck.py --stages run gates 2>&1 | tee experiments/exp09_val_recheck/run.log
    python experiments/analyze_exp09_val_recheck.py --stages run --cells exp07_hann0p3_fbwd \
        --max-seeds 1 --out-dir <scratch>                                   # smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from analyze_exp07_diagnostics import ckpt_path_of, encode_blocks_batched, load_model, pool_blocks
from plot_ml4ps_appendix_b import CURVES, CURVES09, VOID09, WARMUP, load_curve
from swm.eval.skyline import load_first_segment_blocks

log = logging.getLogger("exp09_val_recheck")
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
MANIFEST = EXP / "configs" / "exp09_val_recheck.yaml"

INTERIOR = (16, 240)   # analyze_exp09_artifact.py, unchanged
CENTRE = (112, 145)
ARTIFACT_BATCH = 512   # same batch as the published severity runs, so float summation order matches


def load_manifest() -> dict:
    with open(MANIFEST, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def job_list(man: dict) -> list[dict]:
    """Every (round, cell, seed, ckpt) the manifest names, in manifest order."""
    jobs = []
    for rnd, spec in man["rounds"].items():
        cells = spec["cells"]
        if isinstance(cells, list):
            cells = {c: spec["seeds"] for c in cells}
        for cell, seeds in cells.items():
            for seed in seeds:
                jobs.append({"round": rnd, "cell": cell, "seed": int(seed), "ckpt": spec["ckpt"],
                             "encode": spec["encode"], "probe_layout": spec["probe_layout"]})
    return jobs


def load_labels(man: dict) -> pd.DataFrame:
    """v1 labels for all three splits, checked against the feature cache the published probes used."""
    tasks = man["data"]["tasks"]
    labels = pd.read_parquet(ROOT / man["data"]["labels"])[["tic_id", "split", *tasks]]
    labels = labels.drop_duplicates("tic_id").set_index("tic_id")
    published = pd.read_parquet(EXP / "exp06_features_cache.parquet")[["tic_id", *tasks]]
    published = published.drop_duplicates("tic_id").set_index("tic_id")
    joined = published.join(labels[tasks], rsuffix="_new", how="left")
    for task in tasks:
        mismatch = int((joined[task] != joined[f"{task}_new"]).sum())
        assert mismatch == 0, f"{task}: {mismatch} train/test labels differ from the published feature cache"
    log.info(f"labels: {len(labels)} stars; train+test labels identical to the published cache on all tasks")
    return labels


# ----------------------------------------------------------------------------------------------------
# stage: run
# ----------------------------------------------------------------------------------------------------
def load_split_windows(packed: Path, split: str, window: int, limit: int) -> np.ndarray:
    """Every packed window of one split in stored order (analyze_exp09_artifact.load_test_windows)."""
    index = pd.read_parquet(packed / f"{split}_index.parquet")
    total = int(index["n_win"].sum())
    memmap = np.memmap(packed / f"{split}_windows.dat", dtype=np.float32, mode="r", shape=(total, window))
    if limit:
        index = index[index["tic_id"].isin(sorted(index["tic_id"].unique())[:limit])]
    rows = np.concatenate([np.arange(int(a), int(a) + int(n)) for a, n in zip(index["row_start"], index["n_win"])])
    return np.array(memmap[rows], dtype=np.float32)


@torch.no_grad()
def encode_per_star(model, blocks: list[np.ndarray], device: str) -> list[np.ndarray]:
    """One star per forward pass: swm.eval.readout_sweep.encode_blocks, the exp05 published estimator."""
    out = []
    for block in blocks:
        mu, _ = model.encoder(torch.from_numpy(block).unsqueeze(-1).to(device))  # (n_win, z)
        out.append(mu.float().cpu().numpy())
    return out


@torch.no_grad()
def position_profile(model, x: np.ndarray, device: str) -> np.ndarray:
    """Mean squared reconstruction error at each within-window position (published estimator)."""
    window = x.shape[1]
    total = np.zeros(window, dtype=np.float64)
    for start in range(0, x.shape[0], ARTIFACT_BATCH):
        chunk = torch.from_numpy(x[start:start + ARTIFACT_BATCH]).unsqueeze(-1).to(device)  # (b, W, 1)
        mu, _ = model.encoder(chunk)                                                        # (b, z)
        recon = model.decoder(mu)[:, :, 0]                                                  # (b, W)
        err = (recon - chunk[:, :, 0]) ** 2                                                 # (b, W)
        total += err.sum(dim=0).double().cpu().numpy()
    return total / max(x.shape[0], 1)


def artifact_ratios(profile: np.ndarray) -> dict:
    interior = float(np.median(profile[INTERIOR[0]:INTERIOR[1]]))
    return {"max_ratio": float(profile.max() / interior), "max_pos": int(profile.argmax()),
            "centre_ratio": float(profile[CENTRE[0]:CENTRE[1]].max() / interior),
            "edge_ratio": float(max(profile[0], profile[-1]) / interior)}


def probe_rows(pooled: dict, tics: dict, labels: pd.DataFrame, tasks: list[str],
               poolings: list[str], layout: str) -> list[dict]:
    """
    Fit the published probe once per (pooling, task) on train and score val and test from that fit.
    Same estimator as skyline.logistic_scores: StandardScaler on train, balanced logistic, max_iter 2000.
    `layout` reproduces the memory order the published numbers were computed in (see the manifest):
    the mean_std fit is ill-conditioned enough that C vs Fortran order moves PR-AUC by up to 8e-4.
    """
    order = np.asfortranarray if layout == "F" else np.ascontiguousarray
    rows = []
    for pooling in poolings:
        x = {}
        for split in ("train", "val", "test"):
            mean, std = pooled[split]
            x[split] = order(mean if pooling == "mean" else np.concatenate([mean, std], axis=1))  # (n, z) | (n, 2z)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x["train"])
        for task in tasks:
            y_train = labels.loc[tics["train"], task].to_numpy()
            clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(x_train, y_train)
            for split in ("val", "test"):
                y = labels.loc[tics[split], task].to_numpy()
                scores = clf.predict_proba(scaler.transform(x[split]))[:, 1]
                rows.append({"pooling": pooling, "task": task, "split": split,
                             "pr_auc": float(average_precision_score(y, scores)),
                             "n": int(len(y)), "n_pos": int(y.sum())})
    return rows


def run_stage(man: dict, jobs: list[dict], out_dir: Path, limit: int, device: str) -> None:
    data = man["data"]
    packed, window = ROOT / data["packed"], int(data["window"])
    labels = load_labels(man)

    bags_tics, bags = {}, {}
    for split in ("train", "val", "test"):
        t, b = load_first_segment_blocks(packed, split, window)
        if limit:
            t, b = t[:limit], b[:limit]
        bags_tics[split], bags[split] = t, b
    log.info("first-segment bags: " + ", ".join(f"{s} {len(bags_tics[s])}" for s in bags))
    windows = {s: load_split_windows(packed, s, window, limit) for s in ("val", "test")}
    log.info("artifact windows: " + ", ".join(f"{s} {w.shape[0]}" for s, w in windows.items()))

    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (runs_dir / f"{j['cell']}__seed{j['seed']}.json").exists()]
    log.info(f"{len(jobs)} checkpoints in scope, {len(jobs) - len(todo)} already done, {len(todo)} to run")

    for job in tqdm(todo, desc="checkpoints", total=len(todo)):
        path = ckpt_path_of(job["cell"], job["seed"], "B", job["ckpt"])
        assert path.exists(), f"missing checkpoint {path}"
        model, cfg = load_model(path, device)
        encode = encode_per_star if job["encode"] == "per_star" else encode_blocks_batched
        pooled = {}
        for split in ("train", "val", "test"):
            pooled[split] = pool_blocks(encode(model, bags[split], device))
        artifact = {s: artifact_ratios(position_profile(model, w, device)) for s, w in windows.items()}
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        record = {**job, "lambda_dyn": float(cfg["train"].get("lambda_dyn", 0.0)),
                  "probe": probe_rows(pooled, bags_tics, labels, data["tasks"], data["poolings"],
                                      job["probe_layout"]),
                  "artifact": artifact}
        final = runs_dir / f"{job['cell']}__seed{job['seed']}.json"
        tmp = final.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        os.replace(tmp, final)   # atomic: a kill mid-write never leaves a half file that looks done
        # progress shows TEST numbers only: no val number is read before the footing passes (rule 8)
        test_mean = {r["task"]: r["pr_auc"] for r in record["probe"] if r["split"] == "test" and r["pooling"] == "mean"}
        tqdm.write(f"{job['cell']} s{job['seed']}: test mean " +
                   " ".join(f"{t} {v:.3f}" for t, v in test_mean.items()) +
                   f" | artifact test {artifact['test']['max_ratio']:.2f}")


# ----------------------------------------------------------------------------------------------------
# stage: gates
# ----------------------------------------------------------------------------------------------------
def collect(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probe, artifact, meta = [], [], []
    for path in sorted((out_dir / "runs").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        keys = {"round": rec["round"], "cell": rec["cell"], "seed": rec["seed"], "ckpt": rec["ckpt"]}
        meta.append({**keys, "lambda_dyn": rec["lambda_dyn"]})
        for row in rec["probe"]:
            probe.append({**keys, **row})
        for split, r in rec["artifact"].items():
            artifact.append({**keys, "split": split, **r})
    return pd.DataFrame(probe), pd.DataFrame(artifact), pd.DataFrame(meta)


def published_probe(man: dict) -> pd.DataFrame:
    """The published test PR-AUCs each re-run row must reproduce, keyed (cell, seed, pooling, task)."""
    frames = []
    for cell in man["rounds"]["exp05"]["cells"]:
        s = pd.read_csv(EXP / cell / "results" / "readout_sweep.csv")
        s = s[(s.readout == "logistic") & (s.pooling == "mean") & (s.ckpt == "best_recon_aux")]
        frames.append(s.assign(cell=cell, source="readout_sweep")[["cell", "seed", "pooling", "task", "pr_auc", "source"]])
    for name in ("exp07_aux_gap.csv", "exp07_aux_gap_6seed.csv"):
        g = pd.read_csv(EXP / name)
        g = g[g.arm == "trained"]
        frames.append(g.assign(source=name)[["cell", "seed", "pooling", "task", "pr_auc", "source"]])
    for path in sorted(EXP.glob("exp09_diag_*_probe_summary.csv")):
        tag = path.name.replace("exp09_diag_", "").replace("_probe_summary.csv", "")
        if tag == "ax":   # best_recon_aux arm of three cells; the gates and this re-run read best_recon_only
            continue
        p = pd.read_csv(path)
        p = p[p.cell != "untrained_w256"]
        frames.append(p.assign(source=path.name)[["cell", "seed", "pooling", "task", "pr_auc", "source"]])
    out = pd.concat(frames, ignore_index=True)
    return out[out.pooling.isin(man["data"]["poolings"])]


def footing(man: dict, probe: pd.DataFrame, artifact: pd.DataFrame, out_dir: Path) -> bool:
    print("\n=== FOOTING (test re-run vs published; must hold before any val number is read)")
    tol_p, tol_a = man["footing"]["probe_tolerance"], man["footing"]["artifact_tolerance"]
    test = probe[probe.split == "test"][["round", "cell", "seed", "pooling", "task", "pr_auc"]]
    merged = test.merge(published_probe(man), on=["cell", "seed", "pooling", "task"], suffixes=("", "_pub"))
    merged["abs_diff"] = (merged.pr_auc - merged.pr_auc_pub).abs()
    by_source = merged.groupby(["round", "source"]).agg(rows=("abs_diff", "size"), max_abs_diff=("abs_diff", "max"))
    print(by_source.to_string(float_format=lambda v: f"{v:.2e}"))
    uncovered = test.merge(merged[["cell", "seed", "pooling", "task"]].drop_duplicates(), how="left",
                           on=["cell", "seed", "pooling", "task"], indicator=True)
    uncovered = uncovered[uncovered._merge == "left_only"]
    print(f"  re-run test rows with no published counterpart: {len(uncovered)}"
          + (f" ({sorted(set(uncovered.cell + ':' + uncovered.pooling))[:8]} ...)" if len(uncovered) else ""))
    probe_ok = bool(len(merged)) and float(merged.abs_diff.max()) <= tol_p

    pub_art = pd.concat([pd.read_csv(p).assign(source=p.name) for p in sorted(EXP.glob("exp09_impulse*_runs.csv"))],
                        ignore_index=True)
    pub_art = pub_art.drop_duplicates(["cell", "seed", "ckpt", "max_ratio"])
    art = artifact[artifact.split == "test"].merge(pub_art[["cell", "seed", "ckpt", "max_ratio", "source"]],
                                                    on=["cell", "seed", "ckpt"], suffixes=("", "_pub"))
    art["abs_diff"] = (art.max_ratio - art.max_ratio_pub).abs()
    print(art.groupby("source").agg(rows=("abs_diff", "size"), max_abs_diff=("abs_diff", "max"))
          .to_string(float_format=lambda v: f"{v:.2e}"))
    art_ok = bool(len(art)) and float(art.abs_diff.max()) <= tol_a

    pd.concat([merged.assign(kind="probe"), art.assign(kind="artifact")], ignore_index=True).to_csv(
        out_dir / "footing.csv", index=False)
    print(f"  probe    max |diff| {merged.abs_diff.max():.2e} (tol {tol_p:g}) -> {'OK' if probe_ok else 'FAIL'}")
    print(f"  artifact max |diff| {art.abs_diff.max():.2e} (tol {tol_a:g}) -> {'OK' if art_ok else 'FAIL'}")
    return probe_ok and art_ok


def void_of(cell: str, seeds: list[int], lam: float, band: list[float]) -> dict:
    """G9-dose and the VOID rule from the training curves (analyze_exp09_wave6_gates.dose_and_activity)."""
    if lam <= 0:
        return {"dose_min": np.nan, "n_act_min": np.nan, "G9_dose": "n/a (lambda 0)", "void": False}
    dose, n_active = [], []
    for seed in seeds:
        c = pd.read_csv(CURVES09 / f"{cell}_B_seed{seed}.csv")
        dose.append(lam * c.tail(10)["train/dyn"].mean() / c.tail(10)["train/recon"].mean())
        n_active.append(c.tail(10)["val/n_active_units"].mean())
    return {"dose_min": float(min(dose)), "n_act_min": float(min(n_active)),
            "G9_dose": "PASS" if all(band[0] <= d <= band[1] for d in dose) else "FAIL",
            "void": bool(min(dose) <= 0.0 or min(n_active) <= 0.0)}


def paired(probe: pd.DataFrame, cell: str, ref: str, task: str, pooling: str, split: str) -> dict:
    """Cell minus reference, paired within seed over the seeds both carry (published estimator)."""
    def series(c: str) -> pd.Series:
        rows = probe[(probe.cell == c) & (probe.task == task) & (probe.pooling == pooling) & (probe.split == split)]
        return rows.set_index("seed").pr_auc
    d = (series(cell) - series(ref)).dropna()
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"delta": float(d.mean()), "se": se, "n": int(len(d))}


def exp09_gates(man: dict, probe: pd.DataFrame, artifact: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    spec, gates = man["rounds"]["exp09"], man["gates"]
    ref, tasks = spec["reference"], man["data"]["tasks"]
    rows = []
    for cell, seeds in spec["cells"].items():
        lam = float(meta[meta.cell == cell].lambda_dyn.iloc[0])
        v = void_of(cell, seeds, lam, gates["dose_band"])
        for split in ("test", "val"):
            sev = artifact[(artifact.cell == cell) & (artifact.split == split)].max_ratio
            row = {"cell": cell, "split": split, "n_seeds": len(seeds), "severity": float(sev.mean()),
                   "G9_artifact": bool(sev.mean() <= gates["artifact"]["G9_bar"]),
                   "G17_artifact": bool(sev.mean() <= gates["artifact"]["G17_bar"]), **v}
            for pooling in man["data"]["poolings"]:
                bad = [t for t in tasks if (r := paired(probe, cell, ref, t, pooling, split))["delta"] < -2 * r["se"]]
                row[f"regress_{pooling}"] = ",".join(bad)
            row["noregress"] = all(not row[f"regress_{p}"] for p in man["data"]["poolings"])
            clears = [t for t in tasks if (r := paired(probe, cell, ref, t, "mean", split))["delta"] > 2 * r["se"]]
            row["clause4_clears"] = ",".join(clears)
            row["clause4"] = len(clears) >= 2
            row["both_gates_G9"] = row["G9_artifact"] and row["noregress"] and not row["void"]
            row["switch_fires"] = row["G17_artifact"] and row["noregress"] and row["clause4"] and not row["void"]
            rows.append(row)
    return pd.DataFrame(rows)


def round_ranks(man: dict, probe: pd.DataFrame) -> pd.DataFrame:
    """exp05 / exp07: every cell's seed-mean PR-AUC at mean pooling, ranked per task, on test and val."""
    rows = []
    for rnd in ("exp05", "exp07"):
        p = probe[(probe["round"] == rnd) & (probe.pooling == "mean") & probe.task.isin(man["data"]["app_c_tasks"])]
        means = p.groupby(["cell", "task", "split"]).pr_auc.agg(["mean", "std", "size"]).reset_index()
        for (task, split), g in means.groupby(["task", "split"]):
            g = g.sort_values("mean", ascending=False).reset_index(drop=True)
            for rank, r in enumerate(g.itertuples(), start=1):
                rows.append({"round": rnd, "task": task, "split": split, "cell": r.cell, "rank": rank,
                             "of": len(g), "pr_auc": r.mean, "sd": r.std, "n_seeds": r.size})
    return pd.DataFrame(rows)


def appc_stats(man: dict, probe: pd.DataFrame, gates: pd.DataFrame) -> pd.DataFrame:
    """
    Appendix C's loss-vs-probe statistics, on test (footing against the published figure data) and val:
    the 29 recipes = the ten exp05 cells + the exp09-family cells that are not VOID (incl. the reference).
    """
    # Appendix C's own exclusion set (plot_ml4ps_appendix_b.VOID09: the two cells VOID at 6 seeds), not
    # this script's VOID column -- w0p025 is VOID only at 12 seeds and IS one of the paper's 29 points
    void = set(VOID09)
    fam = ["exp07_hann0p3_fbwd", *man["rounds"]["exp09"]["cells"]]
    cells = {"exp05": man["rounds"]["exp05"]["cells"], "exp09": [c for c in fam if c not in void]}
    recon = {}
    for rnd, names in cells.items():
        curves = CURVES if rnd == "exp05" else CURVES09
        seeds_of = man["rounds"]["exp05"]["seeds"] if rnd == "exp05" else None
        for cell in names:
            seeds = seeds_of or sorted(probe[probe.cell == cell].seed.unique())
            per_seed = []
            for seed in seeds:
                post = load_curve(cell, int(seed), curves)
                per_seed.append(float(post[post["epoch"] >= WARMUP]["val/recon"].min()))
            recon[cell] = float(np.mean(per_seed))
    rows = []
    for scope, names in (("exp05", cells["exp05"]), ("exp09", cells["exp09"]),
                         ("pooled29", cells["exp05"] + cells["exp09"])):
        for split in ("test", "val"):
            for task in man["data"]["app_c_tasks"]:
                p = probe[(probe.pooling == "mean") & (probe.split == split) & (probe.task == task) & probe.cell.isin(names)]
                score = p.groupby("cell").pr_auc.mean()
                loss = pd.Series({c: recon[c] for c in score.index})
                loss_pick, probe_pick = loss.idxmin(), score.idxmax()
                ranks = score.rank(ascending=False, method="min")
                rows.append({"scope": scope, "split": split, "task": task, "n_cells": len(score),
                             "rho": float(spearmanr(-loss, score).statistic),
                             "loss_optimal": loss_pick, "loss_optimal_rank": int(ranks[loss_pick]),
                             "probe_optimal": probe_pick,
                             "cost_of_trusting_loss": float(score[probe_pick] - score[loss_pick]),
                             "shipped_rank": int(ranks.get("exp07_hann0p3_fbwd", -1)) if scope != "exp05" else -1})
    return pd.DataFrame(rows)


def gates_stage(man: dict, out_dir: Path, force: bool) -> int:
    probe, artifact, meta = collect(out_dir)
    expected = len(job_list(man))
    print(f"loaded {len(meta)} of {expected} checkpoint records")
    probe.to_csv(out_dir / "probe.csv", index=False)
    artifact.to_csv(out_dir / "artifact.csv", index=False)
    ok = footing(man, probe, artifact, out_dir)
    if not ok and not force:
        print("\nFOOTING FAILED -- val numbers are NOT read. Fix the drift, or pass --force to inspect.")
        return 1
    if len(meta) < expected:
        print(f"\n{expected - len(meta)} checkpoints missing -- run stage `run` to completion before the gates.")
        return 1

    gates = exp09_gates(man, probe, artifact, meta)
    gates.to_csv(out_dir / "gates_exp09.csv", index=False)
    show = ["severity", "G17_artifact", "void", "regress_mean", "regress_mean_std", "clause4_clears", "switch_fires"]
    wide = gates.set_index(["cell", "split"])[show].unstack("split")
    print("\n=== exp09 GATES vs exp07_hann0p3_fbwd, test (published split) and val (re-check)")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(wide.to_string())
    for split in ("test", "val"):
        g = gates[gates.split == split]
        fired = list(g[g.switch_fires].cell)
        both = list(g[g.both_gates_G9].cell)
        print(f"  {split}: switch fires on {fired or 'no cell'}; both G9 gates pass on {both or 'no cell'}")

    ranks = round_ranks(man, probe)
    ranks.to_csv(out_dir / "ranks_exp05_exp07.csv", index=False)
    print("\n=== exp05 / exp07: best cell per task, and the shipped lineage's rank")
    for (rnd, task), g in ranks.groupby(["round", "task"]):
        line = []
        for split in ("test", "val"):
            s = g[g.split == split]
            top = s.iloc[0]
            lineage = s[s.cell == "exp07_hann0p3_fbwd"] if rnd == "exp07" else s[s.cell.str.contains("fbwd")].head(1)
            line.append(f"{split}: best {top.cell} {top.pr_auc:.3f}, lineage {lineage.cell.iloc[0]} "
                        f"rank {int(lineage['rank'].iloc[0])}/{int(top.of)}")
        print(f"  {rnd} {task:9s} | " + " | ".join(line))
    for rnd in ("exp05", "exp07"):
        for task in man["data"]["app_c_tasks"]:
            g = ranks[(ranks["round"] == rnd) & (ranks.task == task)].pivot(index="cell", columns="split", values="pr_auc")
            print(f"  {rnd} {task:9s} Spearman(test, val) across cells = {spearmanr(g.test, g.val).statistic:+.2f}")

    appc = appc_stats(man, probe, gates)
    appc.to_csv(out_dir / "valloss_appc.csv", index=False)
    print("\n=== Appendix C statistics, test re-run vs val")
    with pd.option_context("display.width", 250):
        print(appc.round(3).to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="X2: every recipe-selection gate re-scored on pool 1's val split")
    ap.add_argument("--stages", nargs="+", default=["run", "gates"], choices=["run", "gates"])
    ap.add_argument("--cells", nargs="+", default=None, help="restrict the run stage to these cells (smoke)")
    ap.add_argument("--max-seeds", type=int, default=0, help="smoke: first N seeds of each cell")
    ap.add_argument("--limit-stars", type=int, default=0, help="smoke: first N stars per split")
    ap.add_argument("--out-dir", default=str(EXP / "exp09_val_recheck"))
    ap.add_argument("--force", action="store_true", help="read val numbers even if the footing fails")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    man = load_manifest()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = job_list(man)
    if args.cells:
        jobs = [j for j in jobs if j["cell"] in args.cells]
    if args.max_seeds:
        first = {}
        for j in jobs:
            first.setdefault(j["cell"], []).append(j)
        jobs = [j for js in first.values() for j in js[:args.max_seeds]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device {device} | out {out_dir}")
    if "run" in args.stages:
        run_stage(man, jobs, out_dir, args.limit_stars, device)
    if "gates" in args.stages:
        return gates_stage(man, out_dir, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
