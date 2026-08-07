"""Figures drawn from committed result files, and nothing else.

Every function here takes fold-level results produced by
``scripts/benchmark.py`` and returns a matplotlib axes. None of them accepts a
literal, and there is no fallback that invents data: given no results, the
loader raises. That is deliberate. The generator this module replaces carried
its numbers as module-level dicts and printed claims — "46-82% smaller",
"statistical equivalence to CART" — that ``paper/CLAIMS.md`` marks FABRICATED.
Deleting the figures while leaving that generator in place fixed nothing.

What may be plotted is constrained by ``paper/PREREGISTRATION.md``:

* K4 — reported interpretability is leaf count, mean weighted decision-path
  length and distinct features used. The composite interpretability score is a
  search heuristic and is not an outcome measure, so nothing here plots it.
* Cross-fold paired t-tests are gone (``0d446e7``). Folds of one CV are not
  independent, so per-dataset differences are drawn as descriptive only, with
  no significance annotation. Inference happens across datasets, via
  :mod:`ga_trees.evaluation.statistics`.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ga_trees.evaluation.statistics import FriedmanResult

logger = logging.getLogger(__name__)

#: Categorical slots, assigned in fixed order and never cycled. Validated for
#: all-pairs use (scatter, small multiples) at three slots: worst pair CVD
#: dE 9.2, normal-vision dE 24.0 on a #fcfcfb surface. A fourth slot puts
#: yellow beside orange and fails the all-pairs floor, so charts that need more
#: series fold the tail into a neutral rather than growing the palette.
SERIES_COLORS: Tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")

#: Everything that is not the subject of the chart.
NEUTRAL = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"

#: Diverging pair for above/below-baseline deltas, with a gray midpoint.
DIVERGING_POSITIVE = "#2a78d6"
DIVERGING_NEGATIVE = "#d03b3b"

#: Columns every fold-level result file carries.
REQUIRED_COLUMNS = frozenset({"dataset", "method", "fold", "test_accuracy"})

#: Columns a frontier-level result file carries (scripts/frontier_benchmark.py).
REQUIRED_FRONTIER_COLUMNS = frozenset({"dataset", "method", "fold", "hypervolume"})


def load_fold_results(source: str) -> pd.DataFrame:
    """Load fold-level benchmark results.

    Parameters
    ----------
    source : str
        A ``folds-*.csv`` written by ``scripts/benchmark.py``, or a directory
        containing one or more of them. Given a directory, the most recently
        modified file is used.

    Returns
    -------
    DataFrame
        One row per (dataset, method, outer fold).

    Raises
    -------
    FileNotFoundError
        If no result file exists. There is deliberately no default dataset to
        fall back on — a figure with no run behind it is the failure mode this
        module exists to prevent.
    ValueError
        If the file is missing columns the figures depend on.
    """
    path = Path(source)
    if path.is_dir():
        candidates = sorted(path.glob("folds-*.csv"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(
                f"No folds-*.csv in {path}. Run scripts/benchmark.py first — "
                "these figures are generated from run output, never from stored numbers."
            )
        path = candidates[-1]
    elif not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/benchmark.py first — these figures "
            "are generated from run output, never from stored numbers."
        )

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")

    logger.info("Loaded %d rows from %s", len(frame), path)
    frame.attrs["source"] = str(path)
    return frame


def load_frontier_results(source: str) -> pd.DataFrame:
    """Load fold-level hypervolumes from ``scripts/frontier_benchmark.py``.

    Same contract as :func:`load_fold_results`: no result file, no figure.
    """
    path = Path(source)
    if path.is_dir():
        candidates = sorted(path.glob("frontier-folds-*.csv"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(
                f"No frontier-folds-*.csv in {path}. Run scripts/frontier_benchmark.py first."
            )
        path = candidates[-1]
    elif not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run scripts/frontier_benchmark.py first.")

    frame = pd.read_csv(path)
    missing = REQUIRED_FRONTIER_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")

    frame.attrs["source"] = str(path)
    return frame


def dataset_method_means(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean of *column* per (dataset, method), as a dataset x method table."""
    if column not in frame.columns:
        raise ValueError(f"Column '{column}' not in results; have {sorted(frame.columns)}.")
    return frame.pivot_table(index="dataset", columns="method", values=column, aggfunc="mean")


def critical_difference_diagram(
    result: FriedmanResult,
    ax,
    title: Optional[str] = None,
) -> "object":
    """Draw a Demsar critical-difference diagram.

    Methods sit on a rank axis with 1 (best) at the left. Any set of methods
    whose average ranks span no more than the Nemenyi critical difference is
    joined by a heavy bar, meaning the post-hoc test cannot separate them.

    The bars are the honest part of this plot: they show how much of the ranking
    is noise. With few datasets the critical difference is wide enough to join
    everything, which is the correct reading, not a drawing error.

    Parameters
    ----------
    result : FriedmanResult
        From :func:`ga_trees.evaluation.statistics.friedman_nemenyi`.
    ax : matplotlib axes
        Target axes.
    title : str, optional
        Overrides the default title.

    Raises
    ------
    ValueError
        If *result* has no critical difference, which means the omnibus test did
        not run. Drawing rank positions without it would imply a comparison the
        data does not license.
    """
    if result.critical_difference is None:
        raise ValueError(
            "FriedmanResult has no critical difference "
            f"({result.note or 'omnibus test did not run'}); nothing to draw."
        )

    ordered = sorted(result.average_ranks.items(), key=lambda kv: kv[1])
    names = [name for name, _ in ordered]
    ranks = [rank for _, rank in ordered]
    n_methods = len(names)
    cd = result.critical_difference

    low, high = 1, n_methods
    span = high - low

    # Vertical layout in data coordinates, anchored top-down so the axes can be
    # trimmed to the content instead of leaving dead space under short diagrams.
    ruler_y = 1.0
    axis_y = 0.86
    clique_step = 0.05
    label_top = axis_y - 0.16
    label_step = 0.13
    rows = (n_methods + 1) // 2

    ax.set_xlim(low - 0.35, high + 0.35)  # rank 1, the best, on the left
    ax.set_ylim(label_top - label_step * (rows - 1) - 0.10, ruler_y + 0.14)
    ax.axis("off")

    ax.plot([low, high], [axis_y, axis_y], color=BASELINE, linewidth=1.5, zorder=1)
    for tick in range(low, high + 1):
        ax.plot([tick, tick], [axis_y, axis_y + 0.035], color=BASELINE, linewidth=1.2, zorder=1)
        ax.text(
            tick,
            axis_y + 0.055,
            str(tick),
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK_SECONDARY,
        )

    # Better-ranked half exits left, worse half exits right, so leader lines
    # never cross and the label column stays outside the plotted range.
    left_count = (n_methods + 1) // 2
    for index, (name, rank) in enumerate(zip(names, ranks)):
        going_left = index < left_count
        row = index if going_left else n_methods - 1 - index
        label_y = label_top - label_step * row
        elbow = low - 0.30 if going_left else high + 0.30

        ax.plot([rank, rank], [axis_y, label_y], color=INK_SECONDARY, linewidth=1.4, zorder=2)
        ax.plot([rank, elbow], [label_y, label_y], color=INK_SECONDARY, linewidth=1.4, zorder=2)
        ax.text(
            elbow - 0.06 if going_left else elbow + 0.06,
            label_y,
            f"{name} ({rank:.2f})",
            ha="right" if going_left else "left",
            va="center",
            fontsize=10,
            color=INK_PRIMARY,
        )

    for offset, (start, end) in enumerate(_cliques(ranks, cd)):
        bar_y = axis_y - 0.04 - clique_step * offset
        ax.plot(
            [ranks[start] - 0.03, ranks[end] + 0.03],
            [bar_y, bar_y],
            color=INK_PRIMARY,
            linewidth=4.0,
            solid_capstyle="round",
            zorder=3,
        )

    # The CD ruler sits over the axis at the same scale, so its width can be
    # compared to any rank gap by eye.
    ruler_end = min(low + cd, high)
    ax.plot([low, ruler_end], [ruler_y, ruler_y], color=INK_PRIMARY, linewidth=2.0, zorder=3)
    for end in (low, ruler_end):
        ax.plot([end, end], [ruler_y - 0.025, ruler_y + 0.025], color=INK_PRIMARY, linewidth=2.0)
    ax.text(
        low + (ruler_end - low) / 2,
        ruler_y + 0.04,
        f"CD = {cd:.2f}" + (" (wider than the rank range)" if cd > span else ""),
        ha="center",
        va="bottom",
        fontsize=10,
        color=INK_PRIMARY,
    )

    default_title = (
        f"Average ranks over {result.n_datasets} datasets " f"(Friedman p = {result.p_value:.4f})"
        if result.p_value is not None
        else f"Average ranks over {result.n_datasets} datasets"
    )
    ax.set_title(title or default_title, fontsize=11, color=INK_PRIMARY, pad=14)
    return ax


def _cliques(ranks: Sequence[float], cd: float) -> List[Tuple[int, int]]:
    """Maximal runs of rank-adjacent methods spanning no more than *cd*.

    Runs contained in a longer run are dropped, and singletons are not drawn —
    a bar joining a method to itself asserts nothing.
    """
    spans: List[Tuple[int, int]] = []
    for start in range(len(ranks)):
        end = start
        while end + 1 < len(ranks) and ranks[end + 1] - ranks[start] <= cd:
            end += 1
        if end > start:
            spans.append((start, end))

    return [
        span
        for index, span in enumerate(spans)
        if not any(
            other[0] <= span[0] and span[1] <= other[1] and other != span
            for j, other in enumerate(spans)
            if j != index
        )
    ]


def accuracy_complexity_frontier(
    frame: pd.DataFrame,
    ax,
    methods: Optional[Sequence[str]] = None,
    complexity: str = "leaves",
) -> "object":
    """Held-out accuracy against model size, one marker per method per dataset.

    This is the figure the frontier claim lives or dies on: a method is better
    only if it sits up and to the left. Comparing accuracy alone across methods
    that settle at different sizes compares points on different parts of the
    trade-off, which is what the retired "equivalent accuracy at 46-82% smaller"
    framing did.

    Parameters
    ----------
    frame : DataFrame
        Fold-level results.
    ax : matplotlib axes
        Target axes.
    methods : sequence of str, optional
        Methods to draw, in palette order. Capped at three: the palette
        validates all-pairs at three slots and a fourth fails the separation
        floor for scatter.
    complexity : str
        Complexity column — ``leaves`` (K4's primary axis) or
        ``mean_path_length``.
    """
    available = list(frame["method"].unique())
    methods = list(methods or available)[: len(SERIES_COLORS)]

    accuracy = dataset_method_means(frame, "test_accuracy")
    size = dataset_method_means(frame, complexity)

    for slot, method in enumerate(methods):
        if method not in accuracy.columns:
            logger.warning("Method '%s' not in results; skipping.", method)
            continue
        ax.scatter(
            size[method],
            accuracy[method],
            s=70,
            color=SERIES_COLORS[slot],
            edgecolor="#fcfcfb",
            linewidth=1.5,
            label=method,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlabel(
        {"leaves": "Leaves (log scale) — fewer is simpler"}.get(
            complexity, f"{complexity} (log scale)"
        ),
        fontsize=10,
        color=INK_SECONDARY,
    )
    ax.set_ylabel("Held-out accuracy", fontsize=10, color=INK_SECONDARY)
    ax.set_title(
        "Accuracy against model size — upper left is better",
        fontsize=11,
        color=INK_PRIMARY,
        pad=10,
    )
    ax.grid(True, color=GRIDLINE, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    return ax


def accuracy_delta_bars(
    frame: pd.DataFrame,
    ax,
    method: str,
    reference: str,
    column: str = "test_accuracy",
    label: Optional[str] = None,
) -> "object":
    """Per-dataset difference between two methods on *column*.

    Descriptive only. There is no p-value here by design: the folds behind each
    mean come from one cross-validation and are not independent, so a per-dataset
    test on them is invalid. Inference is across datasets — see
    :func:`ga_trees.evaluation.statistics.compare_across_datasets`.
    """
    label = label or column.replace("_", " ").capitalize()
    accuracy = dataset_method_means(frame, column)
    for name in (method, reference):
        if name not in accuracy.columns:
            raise ValueError(f"Method '{name}' not in results; have {list(accuracy.columns)}.")

    delta = (accuracy[method] - accuracy[reference]).sort_values()
    colors = [DIVERGING_POSITIVE if value >= 0 else DIVERGING_NEGATIVE for value in delta]

    positions = np.arange(len(delta))
    ax.barh(positions, delta.values, color=colors, height=0.68, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(delta.index, fontsize=9, color=INK_SECONDARY)
    ax.axvline(0, color=BASELINE, linewidth=1.4, zorder=2)

    span = float(np.max(np.abs(delta.values))) if len(delta) else 0.01
    ax.set_xlim(-span * 1.35, span * 1.35)
    precision = 3 if span < 1 else 2
    for position, value in zip(positions, delta.values):
        offset = span * 0.05
        ax.text(
            value + (offset if value >= 0 else -offset),
            position,
            f"{value:+.{precision}f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.5,
            color=INK_SECONDARY,
        )

    wins = int(np.sum(delta.values > 0))
    ax.set_xlabel(f"{label}: {method} minus {reference}", fontsize=10, color=INK_SECONDARY)
    ax.set_title(
        f"{method} vs {reference}, per dataset — {method} ahead on {wins}/{len(delta)} "
        f"(descriptive)",
        fontsize=11,
        color=INK_PRIMARY,
        pad=10,
    )
    ax.grid(True, axis="x", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    return ax


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-method means over every dataset, for the table view a figure needs."""
    columns = [
        c
        for c in (
            "test_accuracy",
            "test_f1",
            "leaves",
            "nodes",
            "depth",
            "mean_path_length",
            "features_used",
            "fit_seconds",
        )
        if c in frame.columns
    ]
    table = frame.groupby("method")[columns].mean()
    return table.sort_values("test_accuracy", ascending=False)


def method_scores_by_dataset(frame: pd.DataFrame, column: str = "test_accuracy") -> Dict[str, list]:
    """Per-dataset means keyed by method, aligned for the statistics layer."""
    table = dataset_method_means(frame, column).dropna(axis=1, how="any")
    return {method: table[method].tolist() for method in table.columns}
