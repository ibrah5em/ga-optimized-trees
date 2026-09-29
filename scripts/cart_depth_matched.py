#!/usr/bin/env python
"""CART's pruning path with the GA's depth cap, on the committed run's folds.

The pre-registered comparator (``CARTPathFrontier``) is not depth-limited, while
the GA cannot grow past ``tree.max_depth``. This re-derives CART's path with
``max_depth`` set to the same cap, on the same outer folds and seeds, and
rescores all methods against one per-dataset reference, so K2 can be read both
ways. Exploratory: added after K2 was known.

    python scripts/cart_depth_matched.py --output-dir paper/evidence/cart-depth6-2026-09-29
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ga_trees.benchmark.frontiers import (  # noqa: E402
    REFERENCE_MARGIN,
    CARTPathFrontier,
    _score_candidates,
)
from ga_trees.evaluation.hypervolume import frontier, hypervolume  # noqa: E402
from ga_trees.reproducibility import derive_fold_seed  # noqa: E402

EVIDENCE = ROOT / "paper" / "evidence" / "frontier-2026-08-07"
NAME = "CART (ccp path, depth-capped)"


def capped_path(X, y, seed, tree_config, max_alphas=12):
    kwargs = dict(
        min_samples_split=tree_config["min_samples_split"],
        min_samples_leaf=tree_config["min_samples_leaf"],
        max_depth=tree_config["max_depth"],
    )
    probe = DecisionTreeClassifier(random_state=seed, **kwargs)
    alphas = np.unique(probe.cost_complexity_pruning_path(X, y).ccp_alphas)
    alphas = alphas[alphas >= 0]
    if len(alphas) > max_alphas:
        alphas = alphas[np.linspace(0, len(alphas) - 1, max_alphas).astype(int)]
    return [
        DecisionTreeClassifier(ccp_alpha=float(a), random_state=seed, **kwargs).fit(X, y)
        for a in alphas
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    from experiment import load_dataset

    config = yaml.safe_load(open(EVIDENCE / "config.yaml"))
    committed = pd.read_csv(EVIDENCE / "points.csv")
    rows = []
    for name in config["experiment"]["datasets"]:
        X, y = load_dataset(name)
        outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        for fold, (tr, te) in enumerate(outer.split(X, y), 1):
            # Same seed as the pre-registered CART path on this fold.
            seed = derive_fold_seed(42, name, fold, CARTPathFrontier.name)
            models = capped_path(X[tr], y[tr], seed, config["tree"])
            front = frontier(_score_candidates(models, X[tr], y[tr], X[te], y[te]))
            for accuracy, nodes in np.asarray(front.points).tolist():
                rows.append(
                    {
                        "dataset": name,
                        "method": NAME,
                        "fold": fold,
                        "accuracy": accuracy,
                        "nodes": nodes,
                    }
                )
        print(f"{name} done", flush=True)
    capped = pd.DataFrame(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    capped.to_csv(out / "points.csv", index=False)

    points = pd.concat([committed, capped], ignore_index=True)
    records = []
    for name, block in points.groupby("dataset"):
        reference = float(block.nodes.max()) + REFERENCE_MARGIN
        for (method, fold), cell in block.groupby(["method", "fold"]):
            front = frontier(list(zip(cell.accuracy, cell.nodes)))
            records.append(
                {
                    "dataset": name,
                    "method": method,
                    "fold": fold,
                    "hypervolume": hypervolume(front, reference_nodes=reference),
                    "reference_nodes": reference,
                }
            )
    pd.DataFrame(records).to_csv(out / "folds.csv", index=False)
    print(f"✓ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
