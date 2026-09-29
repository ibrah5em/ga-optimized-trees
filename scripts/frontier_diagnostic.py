#!/usr/bin/env python
"""Why are the GA's fronts truncated? An exploratory ablation (not pre-registered).

The paper's K2 analysis finds the GA's delivered fronts stop at a median of ~5
nodes while CART's pruning path reaches ~32, and that the hypervolume gap tracks
the accuracy only larger trees reach. Three candidate causes are separated here,
each by changing one thing on the committed run's folds:

* ``resubstitution`` — no validation split (``validation_fraction: 0``). If the
  small validation split is what stops larger trees surviving selection, fronts
  should lengthen.
* ``grow-bias`` — ``growth_stop_prob: 0`` and the expand/prune mutation weights
  swapped. If the initial and mutational bias toward small trees is the cause,
  fronts should lengthen.
* ``2x-budget`` — population and generations doubled. If the search is simply
  under-budgeted, fronts should improve.

Added after K2 was known. It decides nothing and must not be used to re-open K2;
it is reported only to test the mechanism the paper proposes.

    python scripts/frontier_diagnostic.py --datasets vowel,eucalyptus,vehicle,tic_tac_toe
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ga_trees.benchmark.frontiers import (  # noqa: E402
    REFERENCE_MARGIN,
    ParetoGAFrontier,
    _score_candidates,
)
from ga_trees.evaluation.hypervolume import frontier, hypervolume  # noqa: E402
from ga_trees.reproducibility import derive_fold_seed  # noqa: E402

EVIDENCE = ROOT / "paper" / "evidence" / "frontier-2026-08-07"
GA = "GA (NSGA-II)"


def variants(config):
    base = copy.deepcopy(config)
    resub = copy.deepcopy(config)
    resub["fitness"]["validation_fraction"] = 0.0
    grow = copy.deepcopy(config)
    grow["tree"]["growth_stop_prob"] = 0.0
    weights = grow["ga"]["mutation_types"]
    weights["expand_leaf"], weights["prune_subtree"] = (
        weights["prune_subtree"],
        weights["expand_leaf"],
    )
    budget = copy.deepcopy(config)
    budget["ga"]["population_size"] *= 2
    budget["ga"]["n_generations"] *= 2
    return {"base": base, "resubstitution": resub, "grow-bias": grow, "2x-budget": budget}


def run_dataset(name, folds, config):
    from experiment import load_dataset

    X, y = load_dataset(name)
    outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(outer.split(X, y), 1):
        if fold > folds:
            break
        # The committed GA's seed for this fold, so "base" reproduces it.
        seed = derive_fold_seed(42, name, fold, GA)
        for label, cfg in variants(config).items():
            method = ParetoGAFrontier(cfg["ga"], cfg["tree"], cfg["fitness"])
            models, evaluations = method.build(X[tr], y[tr], seed)
            front = frontier(_score_candidates(models, X[tr], y[tr], X[te], y[te]))
            for accuracy, nodes in np.asarray(front.points).tolist():
                rows.append(
                    {
                        "dataset": name,
                        "method": f"GA [{label}]",
                        "fold": fold,
                        "accuracy": accuracy,
                        "nodes": nodes,
                        "evaluations": evaluations,
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    # configs/paper.yaml, not the evidence copy: that copy was written with sorted
    # keys, and mutation_types order decides which operator a given random draw
    # selects, so it does not reproduce the committed run. This file does.
    config = yaml.safe_load(open(ROOT / "configs" / "paper.yaml"))
    datasets = args.datasets.split(",")

    from joblib import Parallel, delayed

    chunks = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_dataset)(name, args.folds, config) for name in datasets
    )
    new = pd.DataFrame([row for chunk in chunks for row in chunk])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    new.to_csv(out / "points.csv", index=False)

    committed = pd.read_csv(EVIDENCE / "points.csv")
    committed = committed[committed.dataset.isin(datasets) & (committed.fold <= args.folds)]
    points = pd.concat([committed, new[committed.columns]], ignore_index=True)
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
                    "hv_normalised": hypervolume(front, reference_nodes=reference) / reference,
                    "largest_nodes": float(cell.nodes.max()),
                    "best_accuracy": float(cell.accuracy.max()),
                }
            )
    table = pd.DataFrame(records)
    table.to_csv(out / "folds.csv", index=False)
    summary = table.groupby(["dataset", "method"])[
        ["hv_normalised", "largest_nodes", "best_accuracy"]
    ].mean()
    print(summary.round(3).to_string())
    print(f"✓ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
