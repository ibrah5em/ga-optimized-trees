#!/usr/bin/env python
"""Publication figures, generated from committed benchmark output.

Reads the fold-level CSV written by ``scripts/benchmark.py`` and draws only
figures the pre-registration licenses. There are no numbers in this file. Given
no result file it exits with an error rather than drawing anything.

The previous version of this script carried two module-level dicts, ``RESULTS``
and ``PAPER_RESULTS``, and rendered them into figures captioned "GA Achieves
46-82% Tree Size Reduction" and "Statistical Equivalence to CART (All p > 0.05 =
No Significant Difference)". Those numbers were never produced by any run
(since withdrawn), and that second caption is the cross-fold paired t-test
the project retired in ``0d446e7`` as invalid. It was the last thing in the repo
able to regenerate the withdrawn claims.

Examples
--------
    python scripts/visualize_comprehensive.py
    python scripts/visualize_comprehensive.py --results results/nested/folds-paper-2026-08-07.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ga_trees.evaluation.figures import (  # noqa: E402
    accuracy_complexity_frontier,
    accuracy_delta_bars,
    critical_difference_diagram,
    load_fold_results,
    load_frontier_results,
    method_scores_by_dataset,
    summary_table,
)
from ga_trees.evaluation.statistics import friedman_nemenyi  # noqa: E402

#: The three methods the frontier scatter draws, in palette order. Capped at
#: three because the categorical palette only validates all-pairs separation at
#: three slots — see ga_trees.evaluation.figures.SERIES_COLORS.
FRONTIER_METHODS = ("GA-Optimized", "Random Search", "CART (pruned)")

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "axes.facecolor": "#fcfcfb",
        "figure.facecolor": "#fcfcfb",
    }
)


def _save(fig, output_dir: Path, stem: str) -> None:
    """Write a figure as PNG and PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_dir / stem}.png / .pdf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--results",
        default="results/nested",
        help="folds-*.csv from scripts/benchmark.py, or a directory holding one",
    )
    parser.add_argument("--output-dir", default="results/figures", help="Where to write figures")
    parser.add_argument(
        "--reference",
        default="CART (pruned)",
        help="Baseline for the per-dataset accuracy difference figure",
    )
    parser.add_argument(
        "--frontier-results",
        default="results/frontiers",
        help=(
            "frontier-folds-*.csv from scripts/frontier_benchmark.py, or a directory. "
            "Drives the hypervolume figures that K1 and H1 are stated on."
        ),
    )
    args = parser.parse_args()

    try:
        frame = load_fold_results(args.results)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    datasets = sorted(frame["dataset"].unique())
    methods = sorted(frame["method"].unique())
    print(f"Source   : {frame.attrs['source']}")
    print(f"Datasets : {len(datasets)} — {', '.join(datasets)}")
    print(f"Methods  : {', '.join(methods)}\n")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    accuracy_complexity_frontier(frame, ax, methods=FRONTIER_METHODS)
    _save(fig, output_dir, "fig1_accuracy_complexity")

    if args.reference in methods and "GA-Optimized" in methods:
        fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.42 * len(datasets) + 1.6)))
        accuracy_delta_bars(frame, ax, method="GA-Optimized", reference=args.reference)
        _save(fig, output_dir, "fig2_accuracy_delta")
    else:
        print(f"  skipped fig2: needs 'GA-Optimized' and '{args.reference}' in the results")

    scores = method_scores_by_dataset(frame)
    friedman = friedman_nemenyi(scores)
    if friedman.critical_difference is None:
        print(
            f"  skipped fig3 (critical difference): {friedman.note or 'omnibus test did not run'}"
        )
    else:
        fig, ax = plt.subplots(figsize=(8.0, 1.1 + 0.42 * len(scores)))
        critical_difference_diagram(friedman, ax)
        _save(fig, output_dir, "fig3_critical_difference")

    table = summary_table(frame)
    table_path = output_dir / "summary-table.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path)
    print(f"  wrote {table_path}\n")
    print(table.round(4).to_string())

    _frontier_figures(args.frontier_results, output_dir)
    return 0


def _frontier_figures(source: str, output_dir: Path) -> None:
    """Hypervolume figures — the quantity K1 and H1 are actually stated on."""
    try:
        frame = load_frontier_results(source)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  skipped frontier figures: {exc}")
        return

    print(f"\nFrontier source: {frame.attrs['source']}")
    methods = sorted(frame["method"].unique())
    ga = next((m for m in methods if m.startswith("GA") and "archived" not in m), None)
    if ga is None:
        print("  skipped: no GA method in the frontier results")
        return

    for reference, stem in (
        ("Random Search", "fig4_hypervolume_vs_random"),
        ("CART (ccp path)", "fig5_hypervolume_vs_cart"),
    ):
        if reference not in methods:
            continue
        datasets = frame["dataset"].nunique()
        fig, ax = plt.subplots(figsize=(7.6, max(3.0, 0.42 * datasets + 1.6)))
        accuracy_delta_bars(
            frame, ax, method=ga, reference=reference, column="hypervolume", label="Hypervolume"
        )
        _save(fig, output_dir, stem)

    scores = method_scores_by_dataset(frame, column="hypervolume")
    friedman = friedman_nemenyi(scores)
    if friedman.critical_difference is None:
        print(f"  skipped fig6: {friedman.note or 'omnibus test did not run'}")
        return
    fig, ax = plt.subplots(figsize=(8.6, 1.1 + 0.42 * len(scores)))
    critical_difference_diagram(
        friedman,
        ax,
        title=(
            f"Hypervolume ranks over {friedman.n_datasets} datasets "
            f"(Friedman p = {friedman.p_value:.4f})"
        ),
    )
    _save(fig, output_dir, "fig6_hypervolume_critical_difference")


if __name__ == "__main__":
    raise SystemExit(main())
