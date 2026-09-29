#!/usr/bin/env python
"""Exploratory GOSDT frontier on the committed frontier run's folds (Phase 4).

Not pre-registered: added after K1/K2 were known, and reported as exploratory.
Runs :class:`GOSDTPathFrontier` on the first ``--folds`` outer folds of the same
``RepeatedStratifiedKFold(10, 3, random_state=42)`` split the committed run used,
then scores every method's frontier against one reference point per dataset —
recomputed from ``points.csv`` together with GOSDT's points, so the box is still
"max over all methods on that dataset".

    python scripts/gosdt_frontier.py --n-jobs 4
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ga_trees.benchmark.frontiers import REFERENCE_MARGIN, _score_candidates  # noqa: E402
from ga_trees.benchmark.gosdt_frontier import GOSDTPathFrontier  # noqa: E402

GOSDT_NAME = GOSDTPathFrontier.name
from ga_trees.evaluation.hypervolume import frontier, hypervolume  # noqa: E402
from ga_trees.reproducibility import derive_fold_seed  # noqa: E402

EVIDENCE = ROOT / "paper" / "evidence" / "frontier-2026-08-07"


def run_dataset(name, folds, max_depth, time_limit, out, memory_gb):
    import resource
    import time

    cached = out / f"points-{name}.csv"
    if cached.exists():  # resumable: each finished dataset is written as it completes
        return pd.read_csv(cached).to_dict("records")
    if memory_gb:
        # GOSDT's search queue can outgrow the machine within its time limit. A cap
        # turns that into a MemoryError, which the adapter counts as a failed fit.
        limit = int(memory_gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    from experiment import load_dataset

    X, y = load_dataset(name)
    outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(outer.split(X, y), 1):
        if fold > folds:
            break
        method = GOSDTPathFrontier(max_depth=max_depth, time_limit=time_limit)
        seed = derive_fold_seed(42, name, fold, method.name)
        started = time.time()
        models, _ = method.build(X[tr], y[tr], seed)
        elapsed = time.time() - started
        points = frontier(_score_candidates(models, X[tr], y[tr], X[te], y[te])).points
        for accuracy, nodes in np.asarray(points).tolist():
            rows.append(
                {
                    "dataset": name,
                    "method": method.name,
                    "fold": fold,
                    "accuracy": accuracy,
                    "nodes": nodes,
                    "fit_seconds": elapsed,
                    "timeouts": method.n_timeouts,
                    "failures": method.n_failures,
                }
            )
    columns = [
        "dataset",
        "method",
        "fold",
        "accuracy",
        "nodes",
        "fit_seconds",
        "timeouts",
        "failures",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(cached, index=False)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--folds", type=int, default=10, help="First N outer folds (repeat 1)")
    parser.add_argument("--time-limit", type=int, default=30, help="Seconds per GOSDT fit")
    parser.add_argument("--datasets", help="Comma-separated (default: all in the committed run)")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", default="results/gosdt")
    parser.add_argument("--memory-gb", type=float, default=6.0, help="Per-worker address-space cap")
    args = parser.parse_args()

    config = yaml.safe_load(open(EVIDENCE / "config.yaml"))
    committed = pd.read_csv(EVIDENCE / "points.csv")
    datasets = args.datasets.split(",") if args.datasets else list(config["experiment"]["datasets"])
    depth = config["tree"]["max_depth"]

    from joblib import Parallel, delayed

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    chunks = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_dataset)(name, args.folds, depth, args.time_limit, out, args.memory_gb)
        for name in datasets
    )
    gosdt = pd.DataFrame([row for chunk in chunks for row in chunk])

    gosdt.to_csv(out / "gosdt-points.csv", index=False)

    # Hypervolume for every method on the same folds, one reference per dataset.
    points = pd.concat(
        [committed[committed.fold <= args.folds], gosdt[committed.columns.tolist()]],
        ignore_index=True,
    )
    rows = []
    for name, block in points.groupby("dataset"):
        reference = float(block.nodes.max()) + REFERENCE_MARGIN
        for (method, fold), cell in block.groupby(["method", "fold"]):
            front = frontier(list(zip(cell.accuracy, cell.nodes)))
            rows.append(
                {
                    "dataset": name,
                    "method": method,
                    "fold": fold,
                    "hypervolume": hypervolume(front, reference_nodes=reference),
                    "reference_nodes": reference,
                }
            )
    # A fold where every GOSDT fit failed has no points; it scores zero, not "missing".
    for name in datasets:
        present = {r["fold"] for r in rows if r["dataset"] == name and r["method"] == GOSDT_NAME}
        reference = next(r["reference_nodes"] for r in rows if r["dataset"] == name)
        for fold in range(1, args.folds + 1):
            if fold not in present:
                rows.append(
                    {
                        "dataset": name,
                        "method": GOSDT_NAME,
                        "fold": fold,
                        "hypervolume": 0.0,
                        "reference_nodes": reference,
                    }
                )
    with open(out / "gosdt-folds.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    timing = gosdt.groupby(["dataset", "fold"]).agg(
        t=("fit_seconds", "first"), to=("timeouts", "first"), fail=("failures", "first")
    )
    print(
        f"\nGOSDT fits: mean {timing.t.mean():.1f}s per fold, {int(timing.to.sum())} timed out, "
        f"{int(timing.fail.sum())} failed and dropped"
    )
    print(f"✓ {out / 'gosdt-points.csv'}\n✓ {out / 'gosdt-folds.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
