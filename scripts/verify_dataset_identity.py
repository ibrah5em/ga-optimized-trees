#!/usr/bin/env python
"""Check that a locally sourced dataset is the one the committed frontier run used.

The frontier run (``paper/evidence/frontier-2026-08-07``) loaded every dataset
from OpenML. When OpenML is unreachable a mirror can stand in, but only if it is
the *same* data in the *same* row order — fold assignment depends on both. The
CART pruning path is deterministic given (data, fold, seed), so re-deriving its
test-fold frontier points and comparing them to ``points.csv`` exactly is a
strong identity check: a different encoding, row order or row set moves them.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ga_trees.benchmark.frontiers import CARTPathFrontier, _score_candidates  # noqa: E402
from ga_trees.evaluation.hypervolume import frontier  # noqa: E402
from ga_trees.reproducibility import derive_fold_seed  # noqa: E402

EVIDENCE = ROOT / "paper" / "evidence" / "frontier-2026-08-07"


def cart_points_match(name, X, y, base_seed=42, splits=10, repeats=3, max_folds=None):
    config = yaml.safe_load(open(EVIDENCE / "config.yaml"))
    points = pd.read_csv(EVIDENCE / "points.csv")
    points = points[(points.dataset == name) & (points.method == CARTPathFrontier.name)]
    method = CARTPathFrontier(config["tree"])
    outer = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=base_seed)
    mismatched = []
    for fold, (tr, te) in enumerate(outer.split(X, y), 1):
        if max_folds and fold > max_folds:
            break
        seed = derive_fold_seed(base_seed, name, fold, method.name)
        models, _ = method.build(X[tr], y[tr], seed)
        front = frontier(_score_candidates(models, X[tr], y[tr], X[te], y[te]))
        got = np.array(sorted(map(tuple, np.asarray(front.points).tolist())), dtype=float)
        want = np.array(
            sorted(map(tuple, points[points.fold == fold][["accuracy", "nodes"]].values)),
            dtype=float,
        )
        if got.shape != want.shape or not np.allclose(got, want):
            mismatched.append((fold, got, want))
    return mismatched


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--max-folds", type=int, default=None)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    from experiment import load_dataset

    failed = 0
    for name in args.datasets:
        X, y = load_dataset(name)
        bad = cart_points_match(name, X, y, max_folds=args.max_folds)
        status = "IDENTICAL" if not bad else f"DIFFERS on {len(bad)} folds"
        print(f"{name:<20} n={len(y):>5} p={X.shape[1]:>3} {status}")
        if bad:
            failed += 1
            fold, got, want = bad[0]
            print(f"    fold {fold}: got {got[:3].tolist()} want {want[:3].tolist()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
