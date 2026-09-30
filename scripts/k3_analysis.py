#!/usr/bin/env python
"""K3 / H2 verdict from a ``scripts/benchmark.py`` fold CSV.

Applies the decision rule fixed in ``paper/PREREGISTRATION.md`` (2026-09-29)
before the run was started:

* Primary — a dataset counts against the GA when its mean outer-fold accuracy
  is more than the 0.02 margin below tuned CART's. K3 fires above 30% of
  datasets.
* Secondary — per-dataset TOST over outer folds with the Nadeau–Bengio
  corrected variance. Reported, not relied on.
* H2 — across-dataset TOST on dataset means.

Complexity is reported with the Phase 3 proxies only (K4): nodes, leaves, mean
decision-path length and distinct features used.

    python scripts/k3_analysis.py paper/evidence/k3-2026-09-29/folds.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ga_trees.evaluation.statistics import (  # noqa: E402
    DEFAULT_EQUIVALENCE_MARGIN,
    compare_all_to_reference,
    corrected_fold_equivalence,
    equivalence_test,
)

GA = "GA-Optimized"
CART = "CART (pruned)"
K3_SHARE = 0.30
COMPLEXITY = ("nodes", "leaves", "mean_path_length", "features_used")


def per_dataset(folds: pd.DataFrame, outer_splits: int) -> pd.DataFrame:
    """One row per dataset: accuracies, GA − CART, both K3 readings, complexity."""
    margin = DEFAULT_EQUIVALENCE_MARGIN
    ratio = 1.0 / (outer_splits - 1)  # n_test / n_train under k-fold
    rows = []
    for dataset, block in folds.groupby("dataset", sort=True):
        wide = block.pivot_table(index="fold", columns="method", values="test_accuracy")
        diff = (wide[GA] - wide[CART]).dropna()
        mean, low, high, equivalent = corrected_fold_equivalence(diff, ratio, margin)
        row = {"dataset": dataset}
        for method in wide.columns:
            row[f"acc[{method}]"] = float(wide[method].mean())
        row.update(
            {
                "ga_minus_cart": mean,
                "ci90_low": low,
                "ci90_high": high,
                "k3_primary_loss": mean < -margin,
                "k3_secondary_not_equivalent": not equivalent,
            }
        )
        for method in (GA, CART):
            sub = block[block.method == method]
            for column in COMPLEXITY:
                if column in sub:
                    row[f"{column}[{method}]"] = float(sub[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folds_csv")
    parser.add_argument("--outer-splits", type=int, default=10)
    parser.add_argument("--output", help="Per-dataset CSV to write (default: next to input)")
    args = parser.parse_args()

    folds = pd.read_csv(args.folds_csv)
    table = per_dataset(folds, args.outer_splits)
    n = len(table)

    pd.set_option("display.width", 200)
    shown = ["dataset", f"acc[{GA}]", f"acc[{CART}]", "ga_minus_cart", "ci90_low", "ci90_high"]
    print(table[shown].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    primary = int(table.k3_primary_loss.sum())
    secondary = int(table.k3_secondary_not_equivalent.sum())
    print(f"\nK3 primary   : GA > 2 points below tuned CART on {primary}/{n} datasets")
    print(f"               => {'K3 TRIGGERED' if primary / n > K3_SHARE else 'K3 not triggered'}")
    print(f"K3 secondary : corrected TOST fails on {secondary}/{n} datasets")
    print(f"               => {'K3 TRIGGERED' if secondary / n > K3_SHARE else 'K3 not triggered'}")

    means = {
        method: [
            float(folds[(folds.dataset == d) & (folds.method == method)].test_accuracy.mean())
            for d in table.dataset
        ]
        for method in sorted(folds.method.unique())
    }
    h2 = equivalence_test(means[GA], means[CART], method_a=GA, method_b=CART)
    print(
        f"\nH2 (TOST across {h2.n_datasets} datasets): mean diff={h2.mean_difference:+.4f}, "
        f"90% CI [{h2.ci_low:+.4f}, {h2.ci_high:+.4f}], p={h2.p_value:.4f} "
        f"=> {'EQUIVALENT' if h2.equivalent else 'not equivalent'} within ±0.02"
    )

    print("\nWilcoxon, GA minus each method (Holm), accuracy:")
    for c in compare_all_to_reference(means, reference=GA):
        print(
            f"  {c.method_b:22s} diff={c.mean_difference:+.4f} p_holm={c.p_adjusted:.4f} "
            f"{'SIGNIFICANT' if c.significant else 'ns'}"
        )

    print("\nComplexity (mean over datasets of per-dataset means):")
    for column in COMPLEXITY:
        values = [
            f"{method}={np.nanmean(table[f'{column}[{method}]']):.2f}"
            for method in (GA, CART)
            if f"{column}[{method}]" in table
        ]
        print(f"  {column:18s} " + "  ".join(values))

    output = Path(args.output) if args.output else Path(args.folds_csv).with_name("k3-table.csv")
    table.to_csv(output, index=False)
    print(f"\n✓ {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
