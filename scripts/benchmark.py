#!/usr/bin/env python
"""Nested cross-validation benchmark — the harness that produces paper numbers.

Runs the protocol fixed in ``paper/PREREGISTRATION.md``: outer 10-fold × 3
repeats for reporting, inner 5-fold for all hyperparameter selection, applied
identically to every method.

``scripts/experiment.py`` remains for quick flat-CV screening. Nothing from it
belongs in the paper.

Examples
--------
    # Full pre-registered protocol on the pre-registered dataset list
    python scripts/benchmark.py --config configs/paper.yaml

    # Cheap smoke run
    python scripts/benchmark.py --config configs/fast.yaml \\
        --datasets iris,wine --outer-splits 3 --outer-repeats 1 --inner-splits 3 --no-tune

Cost warning: the default protocol is 30 outer folds × 5 inner folds × grid size
fits per method per dataset. On ~20 datasets that is many CPU-hours. Start with
--dry-run to see the fit count before committing to it.
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ga_trees.benchmark import (  # noqa: E402
    GATreeMethod,
    PrunedCARTMethod,
    RandomForestMethod,
    RandomTreeSearch,
    UnconstrainedCARTMethod,
    results_to_nested_dict,
    run_nested_cv,
    verify_budget_match,
)
from ga_trees.benchmark.methods import ga_evaluation_budget  # noqa: E402
from ga_trees.evaluation.statistics import per_dataset_means  # noqa: E402
from ga_trees.reproducibility import build_seed_manifest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import load_dataset, run_statistical_analysis  # noqa: E402

#: Methods that must spend the same number of candidate evaluations (K1).
BUDGET_MATCHED = ("GA-Optimized", "Random Search")


def build_methods(config, tune=True, include_forest=True, tune_depth=True):
    """Instantiate the benchmark methods from a config dict."""
    ga_config = config["ga"]
    tree_config = config["tree"]
    fitness_config = config["fitness"]

    methods = [
        GATreeMethod(ga_config, tree_config, fitness_config, tune=tune, tune_depth=tune_depth),
        RandomTreeSearch(ga_config, tree_config, fitness_config, tune=tune, tune_depth=tune_depth),
        PrunedCARTMethod(tree_config),
        UnconstrainedCARTMethod(),
    ]
    if include_forest:
        methods.append(RandomForestMethod())
    return methods


def estimate_fits(methods, datasets, outer_splits, outer_repeats, inner_splits):
    """Rough fit count for the whole run, for the --dry-run estimate.

    Grid sizes are probed on a small synthetic sample because the real grids for
    CART are data-dependent and the point here is an order of magnitude, not an
    exact number.
    """
    probe_X = np.random.RandomState(0).rand(60, 4)
    probe_y = np.array([0, 1, 2] * 20)

    total = 0
    per_method = {}
    for method in methods:
        grid_size = len(method.param_grid(probe_X, probe_y))
        outer_fits = outer_splits * outer_repeats
        # inner selection fits + one refit per outer fold
        fits = outer_fits * (grid_size * inner_splits + 1) if grid_size > 1 else outer_fits
        per_method[method.name] = fits * len(datasets)
        total += per_method[method.name]
    return total, per_method


def main():
    parser = argparse.ArgumentParser(
        description="Nested CV benchmark", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Path to configuration YAML")
    parser.add_argument("--datasets", help="Comma-separated dataset names (overrides config)")
    parser.add_argument("--outer-splits", type=int, default=10, help="Outer CV folds")
    parser.add_argument("--outer-repeats", type=int, default=3, help="Outer CV repeats")
    parser.add_argument("--inner-splits", type=int, default=5, help="Inner CV folds")
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip inner tuning for the GA and random search (screening only, not for the paper)",
    )
    parser.add_argument("--no-forest", action="store_true", help="Skip the random forest reference")
    parser.add_argument(
        "--no-depth-tuning",
        action="store_true",
        help=(
            "Tune the GA and random search over accuracy weight only, at the configured "
            "depth (a third of the fits). CART is still tuned over depth and ccp_alpha. "
            "Recorded as a deviation for the K3 run."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report the fit count and exit without running"
    )
    parser.add_argument("--output-dir", default="results/nested", help="Where to write artifacts")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Datasets to process in parallel. Parallelism is across datasets only: "
            "seeds come from derive_fold_seed(base, dataset, fold, method), so results are "
            "identical to a serial run. Uses processes, never threads — GAConfig.random_state "
            "seeds the global RNG, which threads would share."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    datasets = (
        [d.strip() for d in args.datasets.split(",")]
        if args.datasets
        else config["experiment"]["datasets"]
    )
    methods = build_methods(
        config,
        tune=not args.no_tune,
        include_forest=not args.no_forest,
        tune_depth=not args.no_depth_tuning,
    )

    total_fits, per_method = estimate_fits(
        methods, datasets, args.outer_splits, args.outer_repeats, args.inner_splits
    )
    ga_budget = ga_evaluation_budget(config["ga"])

    print("=" * 70)
    print("NESTED CROSS-VALIDATION BENCHMARK")
    print("=" * 70)
    print(f"Datasets      : {len(datasets)} — {', '.join(datasets)}")
    print(f"Outer         : {args.outer_splits}-fold x {args.outer_repeats} repeat(s)")
    print(f"Inner         : {args.inner_splits}-fold")
    print(f"Methods       : {', '.join(m.name for m in methods)}")
    print(f"Tuning        : {'off (screening)' if args.no_tune else 'on'}")
    if args.no_depth_tuning:
        print("Depth tuning  : off for GA and random search (weight only; CART unchanged)")
    print(f"GA budget     : {ga_budget} evaluations per fit (random search matched)")
    print(f"Estimated fits: {total_fits:,}")
    for name, count in per_method.items():
        print(f"  {name:24s} {count:>9,}")

    if args.dry_run:
        print("\nDry run — nothing executed.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    config_name = Path(args.config).stem

    checkpoints = output_dir / "partial"
    checkpoints.mkdir(exist_ok=True)

    def run_one(name):
        """Full nested CV for one dataset. Self-contained so it can be forked.

        Each finished dataset is pickled under ``partial/`` and reused on a
        re-run with the same output directory, so an interrupted run loses only
        the datasets in progress. Seeds depend on (dataset, fold, method) alone,
        so a resumed run is identical to an uninterrupted one.
        """
        import pickle

        cached = checkpoints / f"{name}.pkl"
        if cached.exists():
            # Only ever reads a checkpoint this same script wrote into its own
            # output directory, never external input.
            with open(cached, "rb") as handle:
                return pickle.load(handle)  # nosec B301
        X, y = load_dataset(name)
        results = run_nested_cv(
            X,
            y,
            methods,
            dataset_name=name,
            base_seed=config["experiment"]["random_state"],
            outer_splits=args.outer_splits,
            outer_repeats=args.outer_repeats,
            inner_splits=args.inner_splits,
            progress=print if args.n_jobs == 1 else None,
        )
        with open(cached, "wb") as handle:
            pickle.dump(results, handle)
        return results

    all_results = []
    if args.n_jobs == 1:
        for name in datasets:
            print(f"\n{'-' * 70}\n{name}\n{'-' * 70}")
            all_results.extend(run_one(name))
    else:
        from joblib import Parallel, delayed

        print(f"\nRunning {len(datasets)} datasets across {args.n_jobs} processes.")
        print("Per-fold progress is suppressed; output would interleave unreadably.\n")
        # Datasets are independent and every seed is derived from the dataset
        # name, so this reorders work without changing any result.
        for chunk in Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(run_one)(name) for name in datasets
        ):
            all_results.extend(chunk)

    rows = [r.as_row() for r in all_results]
    folds_file = output_dir / f"folds-{config_name}-{stamp}.csv"
    with open(folds_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    nested = results_to_nested_dict(all_results)

    # K1 depends on the two searching methods having spent equal effort.
    budget = verify_budget_match(all_results, BUDGET_MATCHED)
    print(f"\n{'=' * 70}\nBudget match ({' vs '.join(BUDGET_MATCHED)})\n{'=' * 70}")
    for name, mean in budget["mean_evaluations"].items():
        print(f"  {name:24s} {mean:,.1f} evaluations/fit (mean)")
    print(f"  matched={budget['matched']} (relative spread {budget['spread']:.3%})")
    if not budget["matched"]:
        print("  ⚠ Budgets differ — the K1 comparison is not valid until they match.")

    stats_rows = run_statistical_analysis(
        nested, reference="GA-Optimized", equivalence_baseline=PrunedCARTMethod.name
    )

    stats_file = output_dir / f"stats-{config_name}-{stamp}.csv"
    if stats_rows:
        with open(stats_file, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stats_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stats_rows)

    manifest = build_seed_manifest(
        base_seed=config["experiment"]["random_state"],
        dataset_names=datasets,
        n_folds=args.outer_splits * args.outer_repeats,
        methods=tuple(m.name for m in methods),
    )
    manifest["protocol"] = {
        "outer_splits": args.outer_splits,
        "outer_repeats": args.outer_repeats,
        "inner_splits": args.inner_splits,
        "tuning": not args.no_tune,
        "depth_tuning": not args.no_depth_tuning,
        "ga_evaluation_budget": ga_budget,
        "budget_match": budget,
    }
    seeds_file = output_dir / f"seeds-{config_name}-{stamp}.json"
    with open(seeds_file, "w") as handle:
        json.dump(manifest, handle, indent=2)

    config_file = output_dir / f"config-{config_name}-{stamp}.yaml"
    with open(config_file, "w") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)

    datasets_seen, _ = per_dataset_means(nested)
    print(f"\n{'=' * 70}")
    print(f"✓ Fold results : {folds_file}  ({len(rows)} rows over {len(datasets_seen)} datasets)")
    if stats_rows:
        print(f"✓ Statistics   : {stats_file}")
    print(f"✓ Seeds        : {seeds_file}")
    print(f"✓ Config       : {config_file}")


if __name__ == "__main__":
    main()
