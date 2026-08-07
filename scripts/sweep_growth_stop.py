#!/usr/bin/env python
"""Sweep ``tree.growth_stop_prob`` — the one GA parameter never tuned.

``growth_stop_prob`` is the per-node probability that ``TreeInitializer`` emits a
leaf regardless of the depth and sample criteria. It sets how bushy the seed
population is, and it has carried the value 0.3 since the first commit without
anyone measuring what that buys.

Two things are reported, because they answer different questions:

* **Seed shape** — what the initial population actually looks like at each
  setting: how many individuals are stumps, and how many nodes the median
  individual has. This is cheap, needs no evolution, and is the part that says
  whether 0.3 is a defensible default.
* **Held-out accuracy** — a flat stratified CV of the full GA at each setting.
  This is a *screening* measurement, deliberately not the nested protocol: its
  job is to choose a sensible default before the pre-registered run, and no
  number from it belongs in the paper.

Examples
--------
    python scripts/sweep_growth_stop.py --config configs/fast.yaml --seed-only
    python scripts/sweep_growth_stop.py --config configs/fast.yaml \\
        --datasets banknote,wdbc --folds 3
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ga_trees.benchmark.methods import GATreeMethod  # noqa: E402
from ga_trees.ga.engine import TreeInitializer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import load_dataset  # noqa: E402

#: A stump tests once and predicts: three nodes, root plus two leaves. Anything
#: at or below this has no structure for crossover to recombine.
STUMP_NODES = 3

DEFAULT_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7)


def seed_shape(config, dataset_name, growth_stop_prob, n_samples, seed):
    """Describe the initial population at one setting, without evolving it."""
    X, y = load_dataset(dataset_name)
    import random

    random.seed(seed)
    np.random.seed(seed)

    initializer = TreeInitializer(
        n_features=X.shape[1],
        n_classes=len(np.unique(y)),
        max_depth=config["tree"]["max_depth"],
        min_samples_split=config["tree"]["min_samples_split"],
        min_samples_leaf=config["tree"]["min_samples_leaf"],
        growth_stop_prob=growth_stop_prob,
        split_strategy=config["tree"].get("split_strategy", "midpoint"),
    )
    trees = [initializer.create_random_tree(X, y) for _ in range(n_samples)]
    nodes = np.array([t.get_num_nodes() for t in trees], dtype=float)
    depths = np.array([t.get_depth() for t in trees], dtype=float)

    return {
        "dataset": dataset_name,
        "growth_stop_prob": growth_stop_prob,
        "stump_fraction": float(np.mean(nodes <= STUMP_NODES)),
        "median_nodes": float(np.median(nodes)),
        "mean_nodes": float(np.mean(nodes)),
        "mean_depth": float(np.mean(depths)),
        "distinct_sizes": int(len(np.unique(nodes))),
    }


def screening_accuracy(config, dataset_name, growth_stop_prob, folds, seed):
    """Flat-CV held-out accuracy of the GA at one setting."""
    X, y = load_dataset(dataset_name)
    tree_config = dict(config["tree"], growth_stop_prob=growth_stop_prob)
    method = GATreeMethod(config["ga"], tree_config, config["fitness"], tune=False)

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accuracies, leaves = [], []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        model = method.fit(X[train_idx], y[train_idx], {}, seed + fold)
        accuracies.append(float(np.mean(model.predict(X[test_idx]) == y[test_idx])))
        leaves.append(model.n_leaves)

    return {
        "dataset": dataset_name,
        "growth_stop_prob": growth_stop_prob,
        "test_accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        "leaves": float(np.mean(leaves)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/fast.yaml")
    parser.add_argument("--datasets", default="banknote,wdbc,tic_tac_toe")
    parser.add_argument(
        "--grid",
        default=",".join(str(v) for v in DEFAULT_GRID),
        help="Comma-separated growth_stop_prob values",
    )
    parser.add_argument("--folds", type=int, default=3, help="Flat CV folds for screening")
    parser.add_argument("--population", type=int, default=200, help="Trees sampled for seed shape")
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Report seed-population shape only; skip the GA runs",
    )
    parser.add_argument("--output-dir", default="results/sweeps")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    datasets = [d.strip() for d in args.datasets.split(",")]
    grid = [float(v) for v in args.grid.split(",")]
    seed = config["experiment"]["random_state"]

    print(f"growth_stop_prob sweep over {grid}")
    print(f"datasets: {', '.join(datasets)}\n")

    shape_rows = []
    print(f"{'dataset':<14}{'p':>6}{'stumps':>9}{'median':>9}{'mean':>8}{'depth':>8}")
    print("-" * 54)
    for dataset in datasets:
        for value in grid:
            row = seed_shape(config, dataset, value, args.population, seed)
            shape_rows.append(row)
            print(
                f"{row['dataset']:<14}{value:>6.2f}{row['stump_fraction']:>8.0%}"
                f"{row['median_nodes']:>9.0f}{row['mean_nodes']:>8.1f}{row['mean_depth']:>8.2f}"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "growth-stop-seed-shape.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(shape_rows[0].keys()))
        writer.writeheader()
        writer.writerows(shape_rows)
    print(f"\nwrote {output_dir / 'growth-stop-seed-shape.csv'}")

    if args.seed_only:
        print("\n--seed-only: skipping the GA runs.")
        return 0

    accuracy_rows = []
    print(f"\n{'dataset':<14}{'p':>6}{'accuracy':>11}{'sd':>8}{'leaves':>9}")
    print("-" * 48)
    for dataset in datasets:
        for value in grid:
            row = screening_accuracy(config, dataset, value, args.folds, seed)
            accuracy_rows.append(row)
            print(
                f"{row['dataset']:<14}{value:>6.2f}{row['test_accuracy']:>11.4f}"
                f"{row['accuracy_std']:>8.4f}{row['leaves']:>9.1f}"
            )

    with open(output_dir / "growth-stop-accuracy.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(accuracy_rows[0].keys()))
        writer.writeheader()
        writer.writerows(accuracy_rows)
    print(f"\nwrote {output_dir / 'growth-stop-accuracy.csv'}")

    best = {}
    for row in accuracy_rows:
        current = best.get(row["dataset"])
        if current is None or row["test_accuracy"] > current["test_accuracy"]:
            best[row["dataset"]] = row
    print("\nBest setting per dataset (screening only — not a paper number):")
    for dataset, row in best.items():
        print(f"  {dataset:<14} p={row['growth_stop_prob']:.2f}  acc={row['test_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
