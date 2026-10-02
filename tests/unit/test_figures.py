"""Unit tests for the figure generators.

The point of most of these is negative: the module must refuse to draw anything
when there is no run behind it.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ga_trees.evaluation.figures import (  # noqa: E402
    SERIES_COLORS,
    _cliques,
    accuracy_complexity_frontier,
    accuracy_delta_bars,
    critical_difference_diagram,
    dataset_method_means,
    load_fold_results,
    load_frontier_results,
    method_scores_by_dataset,
    summary_table,
)
from ga_trees.evaluation.statistics import FriedmanResult, friedman_nemenyi  # noqa: E402

METHODS = ["GA-Optimized", "Random Search", "CART (pruned)"]
DATASETS = ["alpha", "beta", "gamma"]


@pytest.fixture
def folds(tmp_path):
    """A small but complete fold-level result file."""
    rng = np.random.RandomState(0)
    rows = []
    for dataset_index, dataset in enumerate(DATASETS):
        for method_index, method in enumerate(METHODS):
            for fold in range(1, 4):
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "fold": fold,
                        "seed": 100 + fold,
                        "test_accuracy": 0.7
                        + 0.05 * method_index
                        + 0.01 * dataset_index
                        + rng.rand() * 0.01,
                        "test_f1": 0.7 + 0.05 * method_index,
                        "fit_seconds": 0.1,
                        "nodes": 5 + 10 * method_index,
                        "leaves": 3 + 5 * method_index,
                        "depth": 2 + method_index,
                        "features_used": 1 + method_index,
                        "mean_path_length": 1.5 + method_index,
                        "evaluations": 100,
                    }
                )
    path = tmp_path / "folds-test-2026-08-07.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def axes():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


class TestLoadFoldResults:
    def test_missing_file_raises_rather_than_inventing_data(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Run scripts/benchmark.py first"):
            load_fold_results(str(tmp_path / "nope.csv"))

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No folds-"):
            load_fold_results(str(tmp_path))

    def test_reads_a_file(self, folds):
        frame = load_fold_results(str(folds))
        assert len(frame) == len(DATASETS) * len(METHODS) * 3
        assert frame.attrs["source"] == str(folds)

    def test_directory_picks_the_newest_file(self, tmp_path, folds):
        newer = tmp_path / "folds-newer-2026-08-08.csv"
        pd.read_csv(folds).head(3).to_csv(newer, index=False)
        import os
        import time

        os.utime(newer, (time.time() + 10, time.time() + 10))
        assert load_fold_results(str(tmp_path)).attrs["source"] == str(newer)

    def test_missing_required_column_raises(self, tmp_path, folds):
        frame = pd.read_csv(folds).drop(columns=["test_accuracy"])
        path = tmp_path / "folds-broken-2026-08-07.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match="test_accuracy"):
            load_fold_results(str(path))


@pytest.fixture
def frontier_folds(tmp_path):
    """A small frontier-level result file."""
    rows = []
    for dataset_index, dataset in enumerate(DATASETS):
        for method_index, method in enumerate(METHODS):
            for fold in range(1, 4):
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "fold": fold,
                        "hypervolume": 10.0 + method_index + dataset_index * 0.5,
                        "n_points": 3,
                        "n_distinct": 5,
                    }
                )
    path = tmp_path / "frontier-folds-test-2026-08-07.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestLoadFrontierResults:
    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="frontier_benchmark"):
            load_frontier_results(str(tmp_path))

    def test_reads_a_frontier_file(self, frontier_folds):
        frame = load_frontier_results(str(frontier_folds))
        assert "hypervolume" in frame.columns
        assert len(frame) == len(DATASETS) * len(METHODS) * 3

    def test_missing_hypervolume_column_raises(self, tmp_path, frontier_folds):
        frame = pd.read_csv(frontier_folds).drop(columns=["hypervolume"])
        path = tmp_path / "frontier-folds-broken-2026-08-07.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match="hypervolume"):
            load_frontier_results(str(path))

    def test_delta_bars_work_on_hypervolume(self, frontier_folds, axes):
        frame = load_frontier_results(str(frontier_folds))
        accuracy_delta_bars(
            frame,
            axes,
            method=METHODS[0],
            reference=METHODS[1],
            column="hypervolume",
            label="Hypervolume",
        )
        assert "Hypervolume" in axes.get_xlabel()
        assert len(axes.patches) == len(DATASETS)

    def test_title_reports_the_win_count(self, frontier_folds, axes):
        frame = load_frontier_results(str(frontier_folds))
        accuracy_delta_bars(
            frame, axes, method=METHODS[2], reference=METHODS[0], column="hypervolume"
        )
        # METHODS[2] has the larger hypervolume on every dataset by construction.
        assert f"ahead on {len(DATASETS)}/{len(DATASETS)}" in axes.get_title()


class TestReshaping:
    def test_dataset_method_means_shape(self, folds):
        table = dataset_method_means(load_fold_results(str(folds)), "test_accuracy")
        assert list(table.index) == DATASETS
        assert set(table.columns) == set(METHODS)

    def test_unknown_column_raises(self, folds):
        with pytest.raises(ValueError, match="not in results"):
            dataset_method_means(load_fold_results(str(folds)), "no_such_column")

    def test_method_scores_align_across_methods(self, folds):
        scores = method_scores_by_dataset(load_fold_results(str(folds)))
        assert set(scores) == set(METHODS)
        assert len({len(v) for v in scores.values()}) == 1

    def test_summary_table_is_sorted_best_first(self, folds):
        table = summary_table(load_fold_results(str(folds)))
        assert list(table["test_accuracy"]) == sorted(table["test_accuracy"], reverse=True)


class TestCliques:
    def test_all_within_cd_is_one_clique(self):
        assert _cliques([1.0, 1.5, 2.0], cd=2.0) == [(0, 2)]

    def test_separated_groups(self):
        # 1.0 and 1.5 join; 4.0 and 4.2 join; nothing spans the gap.
        assert _cliques([1.0, 1.5, 4.0, 4.2], cd=1.0) == [(0, 1), (2, 3)]

    def test_singletons_are_not_drawn(self):
        assert _cliques([1.0, 5.0, 9.0], cd=0.5) == []

    def test_contained_runs_are_dropped(self):
        spans = _cliques([1.0, 1.2, 1.4, 3.0], cd=1.0)
        assert (0, 2) in spans
        assert (0, 1) not in spans


class TestCriticalDifferenceDiagram:
    def _result(self, cd=1.0):
        return FriedmanResult(
            methods=METHODS,
            n_datasets=5,
            average_ranks={"GA-Optimized": 1.4, "Random Search": 2.2, "CART (pruned)": 2.4},
            statistic=6.0,
            p_value=0.03,
            critical_difference=cd,
        )

    def test_draws_without_error(self, axes):
        assert critical_difference_diagram(self._result(), axes) is axes

    def test_best_rank_is_on_the_left(self, axes):
        critical_difference_diagram(self._result(), axes)
        left, right = axes.get_xlim()
        assert left < right, "rank axis must ascend left to right"

    def test_every_method_is_labelled(self, axes):
        critical_difference_diagram(self._result(), axes)
        rendered = " ".join(t.get_text() for t in axes.texts)
        for method in METHODS:
            assert method in rendered

    def test_missing_critical_difference_refuses_to_draw(self, axes):
        result = self._result()
        result.critical_difference = None
        result.note = "Friedman needs at least 3 datasets"
        with pytest.raises(ValueError, match="no critical difference"):
            critical_difference_diagram(result, axes)

    def test_end_to_end_from_friedman_nemenyi(self, folds, axes):
        scores = method_scores_by_dataset(load_fold_results(str(folds)))
        result = friedman_nemenyi(scores)
        if result.critical_difference is None:
            pytest.skip(result.note)
        critical_difference_diagram(result, axes)


class TestOtherFigures:
    def test_frontier_caps_series_at_the_validated_palette_size(self, folds, axes):
        frame = load_fold_results(str(folds))
        accuracy_complexity_frontier(frame, axes, methods=METHODS + ["extra", "more"])
        # A fourth categorical hue would fail the all-pairs separation floor.
        assert len(axes.collections) <= len(SERIES_COLORS)

    def test_frontier_uses_a_log_size_axis(self, folds, axes):
        accuracy_complexity_frontier(load_fold_results(str(folds)), axes)
        assert axes.get_xscale() == "log"

    def test_delta_bars_one_per_dataset(self, folds, axes):
        accuracy_delta_bars(
            load_fold_results(str(folds)),
            axes,
            method="GA-Optimized",
            reference="CART (pruned)",
        )
        assert len(axes.patches) == len(DATASETS)

    def test_delta_bars_reject_an_unknown_method(self, folds, axes):
        with pytest.raises(ValueError, match="not in results"):
            accuracy_delta_bars(
                load_fold_results(str(folds)), axes, method="Nonexistent", reference="GA-Optimized"
            )

    def test_no_significance_annotation_on_per_dataset_deltas(self, folds, axes):
        # Cross-fold paired tests aren't valid (folds share training data); the
        # figure must not smuggle them back in as stars or p-values.
        accuracy_delta_bars(
            load_fold_results(str(folds)),
            axes,
            method="GA-Optimized",
            reference="CART (pruned)",
        )
        rendered = " ".join(t.get_text() for t in axes.texts)
        assert "*" not in rendered
        assert "p =" not in rendered and "p=" not in rendered
