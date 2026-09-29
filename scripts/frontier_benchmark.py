#!/usr/bin/env python
"""Frontier benchmark — the run that answers K1 and H1 as pre-registered.

``scripts/benchmark.py`` reports one operating point per method and answers the
accuracy questions. Both K1 and H1 are stated on **hypervolume**, which needs a
set of models per fold, so they need this run instead:

    K1 — if budget-matched random search matches the GA on hypervolume
         (Wilcoxon across datasets, alpha = 0.05), there is no paper.
    K2 — if the GA frontier fails to dominate CART's ccp_alpha frontier on
         >= 60% of datasets, H1 is rejected.

Examples
--------
    python scripts/frontier_benchmark.py --config configs/paper.yaml --dry-run
    python scripts/frontier_benchmark.py --config configs/paper.yaml --n-jobs 8
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

from ga_trees.benchmark.frontiers import (  # noqa: E402
    CARTPathFrontier,
    ParetoGAFrontier,
    RandomSearchFrontier,
    dominance_rate,
    hypervolume_by_dataset,
    run_frontier_cv,
)
from ga_trees.evaluation.statistics import (  # noqa: E402
    MIN_DATASETS_FOR_INFERENCE,
    compare_all_to_reference,
    friedman_nemenyi,
)
from ga_trees.reproducibility import build_seed_manifest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import load_dataset  # noqa: E402

GA_NAME = "GA (NSGA-II)"
GA_ARCHIVED_NAME = "GA (NSGA-II, archived)"
RANDOM_NAME = RandomSearchFrontier.name
CART_NAME = CARTPathFrontier.name

#: K2 rejects H1 below this dominance rate over CART's pruning path.
K2_DOMINANCE_THRESHOLD = 0.60


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", help="Comma-separated names (overrides config)")
    parser.add_argument("--outer-splits", type=int, default=10)
    parser.add_argument("--outer-repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1, help="Datasets processed in parallel")
    parser.add_argument(
        "--no-archived-ga",
        action="store_true",
        help=(
            "Skip the archived-GA row. It is on by default because random search "
            "necessarily delivers an archive, and comparing it against NSGA-II's "
            "final population alone confounds the search with the bookkeeping."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="results/frontiers")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    datasets = (
        [d.strip() for d in args.datasets.split(",")]
        if args.datasets
        else config["experiment"]["datasets"]
    )

    folds = args.outer_splits * args.outer_repeats
    print("=" * 70)
    print("FRONTIER BENCHMARK — hypervolume (K1, H1/K2)")
    print("=" * 70)
    print(f"Datasets  : {len(datasets)} — {', '.join(datasets)}")
    print(f"Outer     : {args.outer_splits}-fold x {args.outer_repeats} repeat(s) = {folds} folds")
    print(f"Methods   : {GA_NAME}, {RANDOM_NAME}, {CART_NAME}")
    print("Objectives: (accuracy, -node count)   [K4: not the composite score]")
    print(f"NSGA-II   : pop={config['ga']['population_size']} gen={config['ga']['n_generations']}")
    print(f"Runs      : {len(datasets) * folds} folds x 3 methods")
    if len(datasets) < MIN_DATASETS_FOR_INFERENCE:
        print(
            f"\n  ! {len(datasets)} datasets is below the {MIN_DATASETS_FOR_INFERENCE} needed "
            "for a signed-rank test to reach alpha=0.05. Results will be descriptive."
        )
    if args.dry_run:
        print("\nDry run — nothing executed.")
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    config_name = Path(args.config).stem

    def run_one(name):
        X, y = load_dataset(name)
        others = [CARTPathFrontier(config["tree"])]
        if not args.no_archived_ga:
            others.insert(
                0, ParetoGAFrontier(config["ga"], config["tree"], config["fitness"], archive=True)
            )
        return run_frontier_cv(
            X,
            y,
            ParetoGAFrontier(config["ga"], config["tree"], config["fitness"]),
            RandomSearchFrontier(config["ga"], config["tree"], config["fitness"]),
            others,
            dataset_name=name,
            base_seed=config["experiment"]["random_state"],
            outer_splits=args.outer_splits,
            outer_repeats=args.outer_repeats,
            progress=print if args.n_jobs == 1 else None,
        )

    results = []
    if args.n_jobs == 1:
        for name in datasets:
            print(f"\n{'-' * 70}\n{name}\n{'-' * 70}")
            results.extend(run_one(name))
    else:
        from joblib import Parallel, delayed

        print(f"\nRunning {len(datasets)} datasets across {args.n_jobs} processes.\n")
        for chunk in Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(run_one)(name) for name in datasets
        ):
            results.extend(chunk)

    rows = [r.as_row() for r in results]
    folds_file = output_dir / f"frontier-folds-{config_name}-{stamp}.csv"
    with open(folds_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    points_file = output_dir / f"frontier-points-{config_name}-{stamp}.csv"
    with open(points_file, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "method", "fold", "accuracy", "nodes"])
        for result in results:
            for accuracy, nodes in result.points:
                writer.writerow([result.dataset, result.method, result.fold, accuracy, nodes])

    nested = hypervolume_by_dataset(results)

    # Budget match is what makes K1 mean anything; verify per fold, not on average.
    mismatches = 0
    by_fold = {}
    for result in results:
        by_fold.setdefault((result.dataset, result.fold), {})[result.method] = result.n_evaluations
    for key, spend in by_fold.items():
        if spend.get(GA_NAME) != spend.get(RANDOM_NAME):
            mismatches += 1
    print(f"\n{'=' * 70}\nBudget match ({GA_NAME} vs {RANDOM_NAME})\n{'=' * 70}")
    print(f"  folds with unequal evaluation counts: {mismatches} of {len(by_fold)}")
    if mismatches:
        print("  ! K1 is not valid until every fold matches.")

    present = [
        name
        for name in (GA_NAME, GA_ARCHIVED_NAME, RANDOM_NAME, CART_NAME)
        if any(name in methods for methods in nested.values())
    ]

    print(f"\n{'=' * 70}\nHypervolume per dataset (mean over folds)\n{'=' * 70}")
    header = f"{'dataset':<22}" + "".join(f"{name:>24}" for name in present)
    print(header)
    print("-" * len(header))
    for dataset in sorted(nested):
        methods = nested[dataset]
        row = f"{dataset:<22}"
        for name in present:
            row += f"{np.mean(methods.get(name, [np.nan])):>24.3f}"
        print(row)

    means = {
        method: [float(np.mean(nested[d][method])) for d in sorted(nested) if method in nested[d]]
        for method in present
    }

    print(f"\n{'=' * 70}\nK1 — GA vs budget-matched random search, on hypervolume\n{'=' * 70}")
    comparisons = compare_all_to_reference({m: v for m, v in means.items()}, reference=GA_NAME)
    for comparison in comparisons:
        p_value = comparison.p_value if comparison.p_value is not None else float("nan")
        p_adjusted = comparison.p_adjusted if comparison.p_adjusted is not None else float("nan")
        effect = comparison.effect_size if comparison.effect_size is not None else float("nan")
        flag = "SIGNIFICANT" if comparison.significant else "(ns)"
        if comparison.underpowered:
            flag = "(underpowered)"
        print(
            f"  vs {comparison.method_b:<16} mean diff={comparison.mean_difference:+.4f} "
            f"p={p_value:.4f} p_holm={p_adjusted:.4f} {flag} d_z={effect:+.3f}"
        )

    random_comparison = next((c for c in comparisons if c.method_b == RANDOM_NAME), None)
    if random_comparison is not None:
        if random_comparison.underpowered:
            # A null result from an underpowered run is not evidence of no
            # effect, and reading it as one would be the exact failure the
            # pre-registration exists to prevent.
            verdict = (
                f"UNDECIDED — {random_comparison.n_datasets} datasets cannot reach alpha=0.05. "
                f"K1 needs at least {MIN_DATASETS_FOR_INFERENCE}. This run decides nothing."
            )
        elif random_comparison.significant and random_comparison.mean_difference > 0:
            verdict = "K1 NOT triggered — the GA beats budget-matched random search."
        elif random_comparison.significant:
            verdict = "K1 TRIGGERED, and worse: random search significantly beats the GA."
        else:
            verdict = (
                "K1 TRIGGERED — no significant hypervolume difference from random search. "
                "Per the pre-registration: publish as software (JOSS) and stop."
            )
        print(f"\n  => {verdict}")

    rate = dominance_rate(results, GA_NAME, CART_NAME)
    print(f"\n{'=' * 70}\nK2 — GA frontier vs CART ccp path\n{'=' * 70}")
    print(
        f"  GA has the larger hypervolume on {rate:.0%} of datasets "
        f"(threshold {K2_DOMINANCE_THRESHOLD:.0%})"
    )
    print(
        f"  => {'H1 survives' if rate >= K2_DOMINANCE_THRESHOLD else 'K2 TRIGGERED — H1 rejected'}"
    )

    friedman = friedman_nemenyi(means)
    print(f"\n{'=' * 70}\nFriedman + Nemenyi on hypervolume\n{'=' * 70}")
    ranks = ", ".join(
        f"{m}={r:.2f}" for m, r in sorted(friedman.average_ranks.items(), key=lambda kv: kv[1])
    )
    print(f"  Average ranks (1 = best): {ranks}")
    if friedman.p_value is not None:
        print(
            f"  chi2={friedman.statistic:.3f}, p={friedman.p_value:.4f}, "
            f"CD={friedman.critical_difference:.3f}"
        )
    else:
        print(f"  {friedman.note}")

    manifest = build_seed_manifest(
        base_seed=config["experiment"]["random_state"],
        dataset_names=datasets,
        n_folds=folds,
        methods=(GA_NAME, RANDOM_NAME, CART_NAME),
    )
    manifest["protocol"] = {
        "outer_splits": args.outer_splits,
        "outer_repeats": args.outer_repeats,
        "objectives": "(accuracy, -node_count)",
        "budget_mismatched_folds": mismatches,
        "k2_dominance_rate": rate,
    }
    with open(output_dir / f"frontier-seeds-{config_name}-{stamp}.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    with open(output_dir / f"frontier-config-{config_name}-{stamp}.yaml", "w") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)

    print(f"\n{'=' * 70}")
    print(f"✓ Fold hypervolumes : {folds_file}  ({len(rows)} rows)")
    print(f"✓ Frontier points   : {points_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
