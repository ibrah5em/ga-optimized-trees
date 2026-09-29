"""Unit tests for evaluation/statistics.py.

Covers:
- summarize (ddof=1 sample standard deviation)
- holm_adjust (step-down correction)
- cohens_dz (paired effect size)
- compare_across_datasets (Wilcoxon + underpowered flagging)
- equivalence_test (TOST)
- friedman_nemenyi (omnibus + critical difference)
- compare_all_to_reference (family-wise correction)
- per_dataset_means (fold collapse)
"""

import numpy as np
import pytest
from scipy import stats

from ga_trees.evaluation.statistics import (
    ALPHA,
    DEFAULT_EQUIVALENCE_MARGIN,
    MIN_DATASETS_FOR_INFERENCE,
    cohens_dz,
    compare_across_datasets,
    compare_all_to_reference,
    equivalence_test,
    friedman_nemenyi,
    holm_adjust,
    per_dataset_means,
    summarize,
)

# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_uses_sample_standard_deviation(self):
        values = [0.90, 0.92, 0.94, 0.96]
        result = summarize(values)
        assert result.std == pytest.approx(np.std(values, ddof=1))
        # The population form is the bug this replaced — it reports a smaller spread.
        assert result.std > np.std(values)

    def test_mean_and_count(self):
        result = summarize([1.0, 2.0, 3.0])
        assert result.mean == pytest.approx(2.0)
        assert result.n == 3

    def test_single_value_has_zero_std(self):
        """ddof=1 is undefined for n=1; report 0.0 rather than nan."""
        result = summarize([0.8])
        assert result.std == 0.0
        assert result.mean == pytest.approx(0.8)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one value"):
            summarize([])

    def test_str_formats_mean_and_std(self):
        assert str(summarize([1.0, 1.0])) == "1.0000 +/- 0.0000"


# ---------------------------------------------------------------------------
# holm_adjust
# ---------------------------------------------------------------------------


class TestHolmAdjust:
    def test_empty_input(self):
        assert holm_adjust([]) == []

    def test_single_p_value_unchanged(self):
        assert holm_adjust([0.03]) == [0.03]

    def test_step_down_multipliers(self):
        # Sorted: 0.01 * 3, 0.02 * 2, 0.03 * 1
        assert holm_adjust([0.01, 0.02, 0.03]) == pytest.approx([0.03, 0.04, 0.04])

    def test_preserves_input_order(self):
        assert holm_adjust([0.03, 0.01]) == pytest.approx([0.03, 0.02])

    def test_clipped_at_one(self):
        assert all(p <= 1.0 for p in holm_adjust([0.6, 0.7, 0.8]))

    def test_monotone_non_decreasing_in_sorted_order(self):
        raw = [0.001, 0.04, 0.005, 0.9]
        adjusted = holm_adjust(raw)
        by_raw = [adjusted[i] for i in sorted(range(len(raw)), key=lambda i: raw[i])]
        assert by_raw == sorted(by_raw)

    def test_at_least_as_powerful_as_bonferroni(self):
        raw = [0.001, 0.01, 0.04]
        bonferroni = [min(p * len(raw), 1.0) for p in raw]
        assert all(h <= b for h, b in zip(holm_adjust(raw), bonferroni))


# ---------------------------------------------------------------------------
# cohens_dz
# ---------------------------------------------------------------------------


class TestCohensDz:
    def test_zero_mean_difference(self):
        assert cohens_dz([-1.0, 1.0, -1.0, 1.0]) == pytest.approx(0.0)

    def test_matches_manual_formula(self):
        diffs = [0.02, 0.03, 0.01, 0.04]
        expected = np.mean(diffs) / np.std(diffs, ddof=1)
        assert cohens_dz(diffs) == pytest.approx(expected)

    def test_too_few_observations(self):
        assert cohens_dz([0.5]) == 0.0

    def test_constant_nonzero_difference_is_infinite(self):
        assert np.isinf(cohens_dz([0.02, 0.02, 0.02]))

    def test_constant_zero_difference_is_zero(self):
        assert cohens_dz([0.0, 0.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# compare_across_datasets
# ---------------------------------------------------------------------------


class TestCompareAcrossDatasets:
    def test_matches_scipy_wilcoxon(self):
        a = [0.91, 0.88, 0.95, 0.79, 0.85, 0.90, 0.93]
        b = [0.89, 0.87, 0.90, 0.80, 0.82, 0.88, 0.86]
        expected_stat, expected_p = stats.wilcoxon(a, b)
        result = compare_across_datasets(a, b, "GA", "CART")
        assert result.statistic == pytest.approx(expected_stat)
        assert result.p_value == pytest.approx(expected_p)

    def test_reports_mean_difference(self):
        result = compare_across_datasets([0.9, 0.8], [0.8, 0.7], "GA", "CART")
        assert result.mean_difference == pytest.approx(0.1)

    def test_flags_underpowered_below_threshold(self):
        n = MIN_DATASETS_FOR_INFERENCE - 1
        result = compare_across_datasets(
            [0.9 + i * 0.01 for i in range(n)], [0.8 + i * 0.01 for i in range(n)]
        )
        assert result.underpowered is True
        assert "below" in result.note

    def test_not_underpowered_at_threshold(self):
        n = MIN_DATASETS_FOR_INFERENCE
        result = compare_across_datasets(
            [0.9 + i * 0.01 for i in range(n)], [0.8 + i * 0.01 for i in range(n)]
        )
        assert result.underpowered is False

    def test_underpowered_is_never_significant(self):
        """Three consistent wins must not be reported as a real effect."""
        result = compare_across_datasets([0.9, 0.8, 0.7], [0.5, 0.4, 0.3])
        result.p_adjusted = 0.001  # even with a tiny adjusted p
        assert result.significant is False

    def test_significant_requires_adjusted_p(self):
        result = compare_across_datasets(
            [0.9 + i * 0.01 for i in range(8)], [0.8 + i * 0.01 for i in range(8)]
        )
        assert result.significant is False  # p_adjusted not set yet
        result.p_adjusted = 0.01
        assert result.significant is True

    def test_identical_scores_skip_the_test(self):
        result = compare_across_datasets([0.9, 0.8, 0.7], [0.9, 0.8, 0.7])
        assert result.p_value is None
        assert "zero" in result.note

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="align"):
            compare_across_datasets([0.9, 0.8], [0.9])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one dataset"):
            compare_across_datasets([], [])


# ---------------------------------------------------------------------------
# equivalence_test (TOST)
# ---------------------------------------------------------------------------


class TestEquivalenceTest:
    def test_tight_differences_are_equivalent(self):
        """Differences well inside ±2% should clear TOST."""
        a = [0.900, 0.901, 0.899, 0.902, 0.898, 0.900, 0.901, 0.899]
        b = [0.899, 0.900, 0.900, 0.901, 0.899, 0.900, 0.900, 0.900]
        result = equivalence_test(a, b, margin=DEFAULT_EQUIVALENCE_MARGIN)
        assert result.equivalent is True
        assert result.p_value < ALPHA

    def test_large_difference_is_not_equivalent(self):
        a = [0.95, 0.96, 0.94, 0.95, 0.96, 0.94, 0.95, 0.96]
        b = [0.80, 0.81, 0.79, 0.80, 0.81, 0.79, 0.80, 0.81]
        result = equivalence_test(a, b, margin=DEFAULT_EQUIVALENCE_MARGIN)
        assert result.equivalent is False

    def test_noisy_small_difference_is_not_equivalent(self):
        """A near-zero mean difference with wide spread is inconclusive, not equivalent.

        This is the failure mode the old ns-means-equivalent reading had.
        """
        a = [0.90, 0.70, 0.95, 0.65, 0.88, 0.72]
        b = [0.70, 0.90, 0.65, 0.95, 0.72, 0.88]
        result = equivalence_test(a, b, margin=DEFAULT_EQUIVALENCE_MARGIN)
        assert result.mean_difference == pytest.approx(0.0, abs=1e-9)
        assert result.equivalent is False

    def test_matches_manual_tost_computation(self):
        a = [0.90, 0.91, 0.89, 0.92, 0.88]
        b = [0.89, 0.90, 0.89, 0.91, 0.88]
        margin = 0.02
        diffs = np.array(a) - np.array(b)
        n = len(diffs)
        se = np.std(diffs, ddof=1) / np.sqrt(n)
        expected = max(
            stats.t.sf((diffs.mean() + margin) / se, n - 1),
            stats.t.cdf((diffs.mean() - margin) / se, n - 1),
        )
        assert equivalence_test(a, b, margin=margin).p_value == pytest.approx(expected)

    def test_confidence_interval_brackets_mean(self):
        a = [0.90, 0.91, 0.89, 0.92, 0.88]
        b = [0.89, 0.90, 0.89, 0.91, 0.88]
        result = equivalence_test(a, b)
        assert result.ci_low <= result.mean_difference <= result.ci_high

    def test_zero_variance_inside_margin(self):
        result = equivalence_test([0.90, 0.80, 0.70], [0.895, 0.795, 0.695], margin=0.02)
        assert result.equivalent is True
        assert "Zero variance" in result.note

    def test_zero_variance_outside_margin(self):
        result = equivalence_test([0.90, 0.80, 0.70], [0.85, 0.75, 0.65], margin=0.02)
        assert result.equivalent is False

    def test_single_dataset_cannot_be_tested(self):
        result = equivalence_test([0.9], [0.9])
        assert result.equivalent is False
        assert "at least 2" in result.note

    def test_non_positive_margin_raises(self):
        with pytest.raises(ValueError, match="margin"):
            equivalence_test([0.9, 0.8], [0.9, 0.8], margin=0.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="align"):
            equivalence_test([0.9, 0.8], [0.9])


# ---------------------------------------------------------------------------
# friedman_nemenyi
# ---------------------------------------------------------------------------


class TestFriedmanNemenyi:
    def _consistent_scores(self):
        return {
            "GA": [0.95, 0.94, 0.96, 0.93, 0.95, 0.94],
            "CART": [0.90, 0.89, 0.91, 0.88, 0.90, 0.89],
            "RF": [0.85, 0.84, 0.86, 0.83, 0.85, 0.84],
        }

    def test_average_ranks_order_methods(self):
        result = friedman_nemenyi(self._consistent_scores())
        assert result.average_ranks["GA"] == pytest.approx(1.0)
        assert result.average_ranks["CART"] == pytest.approx(2.0)
        assert result.average_ranks["RF"] == pytest.approx(3.0)

    def test_higher_scores_rank_better(self):
        result = friedman_nemenyi(self._consistent_scores())
        assert result.average_ranks["GA"] < result.average_ranks["RF"]

    def test_matches_scipy_friedman(self):
        scores = self._consistent_scores()
        expected = stats.friedmanchisquare(*[scores[m] for m in scores])
        result = friedman_nemenyi(scores)
        assert result.statistic == pytest.approx(expected.statistic)
        assert result.p_value == pytest.approx(expected.pvalue)

    def test_critical_difference_matches_demsar_formula(self):
        scores = self._consistent_scores()
        result = friedman_nemenyi(scores)
        k, n = 3, 6
        q = stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
        assert result.critical_difference == pytest.approx(q * np.sqrt(k * (k + 1) / (6.0 * n)))

    def test_critical_difference_shrinks_with_more_datasets(self):
        few = friedman_nemenyi({m: v[:4] for m, v in self._consistent_scores().items()})
        many = friedman_nemenyi(
            {m: list(v) * 3 for m, v in self._consistent_scores().items()},
        )
        assert many.critical_difference < few.critical_difference

    def test_two_methods_skips_omnibus(self):
        result = friedman_nemenyi({"GA": [0.9, 0.8, 0.7], "CART": [0.8, 0.7, 0.6]})
        assert result.p_value is None
        assert "at least 3 methods" in result.note
        assert result.average_ranks["GA"] == pytest.approx(1.0)

    def test_two_datasets_skips_omnibus(self):
        result = friedman_nemenyi({"GA": [0.9, 0.8], "CART": [0.8, 0.7], "RF": [0.7, 0.6]})
        assert result.p_value is None
        assert "at least 3 datasets" in result.note

    def test_differs_is_false_without_critical_difference(self):
        result = friedman_nemenyi({"GA": [0.9, 0.8, 0.7], "CART": [0.8, 0.7, 0.6]})
        assert result.differs("GA", "CART") is False

    def test_rank_gap(self):
        result = friedman_nemenyi(self._consistent_scores())
        assert result.rank_gap("GA", "RF") == pytest.approx(2.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one method"):
            friedman_nemenyi({})


# ---------------------------------------------------------------------------
# compare_all_to_reference
# ---------------------------------------------------------------------------


class TestCompareAllToReference:
    def _scores(self):
        return {
            "GA": [0.95, 0.94, 0.96, 0.93, 0.95, 0.94, 0.92],
            "CART": [0.90, 0.89, 0.91, 0.88, 0.90, 0.89, 0.87],
            "RF": [0.85, 0.84, 0.86, 0.83, 0.85, 0.84, 0.82],
        }

    def test_one_comparison_per_other_method(self):
        comparisons = compare_all_to_reference(self._scores(), "GA")
        assert {c.method_b for c in comparisons} == {"CART", "RF"}

    def test_reference_is_method_a(self):
        for comparison in compare_all_to_reference(self._scores(), "GA"):
            assert comparison.method_a == "GA"

    def test_adjusted_p_is_populated(self):
        for comparison in compare_all_to_reference(self._scores(), "GA"):
            assert comparison.p_adjusted is not None
            assert comparison.p_adjusted >= comparison.p_value

    def test_untestable_comparison_has_no_adjusted_p(self):
        scores = self._scores()
        scores["Clone"] = list(scores["GA"])
        comparisons = compare_all_to_reference(scores, "GA")
        clone = next(c for c in comparisons if c.method_b == "Clone")
        assert clone.p_value is None
        assert clone.p_adjusted is None

    def test_unknown_reference_raises(self):
        with pytest.raises(ValueError, match="not in"):
            compare_all_to_reference(self._scores(), "Missing")


# ---------------------------------------------------------------------------
# per_dataset_means
# ---------------------------------------------------------------------------


class TestPerDatasetMeans:
    def _results(self):
        return {
            "iris": {
                "GA": {"test_acc": [0.9, 1.0]},
                "CART": {"test_acc": [0.8, 0.9]},
            },
            "wine": {
                "GA": {"test_acc": [0.7, 0.9]},
                "CART": {"test_acc": [0.6, 0.8]},
            },
        }

    def test_returns_dataset_order(self):
        datasets, _ = per_dataset_means(self._results())
        assert datasets == ["iris", "wine"]

    def test_collapses_folds_to_means(self):
        _, scores = per_dataset_means(self._results())
        assert scores["GA"] == pytest.approx([0.95, 0.80])
        assert scores["CART"] == pytest.approx([0.85, 0.70])

    def test_drops_methods_missing_on_some_dataset(self, caplog):
        results = self._results()
        results["iris"]["XGBoost"] = {"test_acc": [0.99]}
        with caplog.at_level("WARNING"):
            _, scores = per_dataset_means(results)
        assert "XGBoost" not in scores
        assert "XGBoost" in caplog.text

    def test_empty_results(self):
        assert per_dataset_means({}) == ([], {})

    def test_alternate_metric(self):
        results = {"iris": {"GA": {"test_f1": [0.5, 0.7]}}}
        _, scores = per_dataset_means(results, metric="test_f1")
        assert scores["GA"] == pytest.approx([0.6])


class TestCorrectedFoldEquivalence:
    def test_correction_widens_the_interval(self):
        from ga_trees.evaluation.statistics import corrected_fold_equivalence

        diffs = [0.001, -0.004, 0.003, 0.0, 0.002, -0.001, 0.004, -0.002, 0.001, 0.0]
        _, lo_naive, hi_naive, _ = corrected_fold_equivalence(diffs, test_train_ratio=0.0)
        _, lo, hi, _ = corrected_fold_equivalence(diffs, test_train_ratio=1 / 9)
        assert lo < lo_naive and hi > hi_naive

    def test_tight_small_differences_are_equivalent(self):
        from ga_trees.evaluation.statistics import corrected_fold_equivalence

        diffs = [0.001, -0.002, 0.0, 0.001, -0.001, 0.002, 0.0, -0.001, 0.001, 0.0]
        assert corrected_fold_equivalence(diffs, test_train_ratio=1 / 9)[3]

    def test_a_large_loss_is_not_equivalent(self):
        from ga_trees.evaluation.statistics import corrected_fold_equivalence

        diffs = [-0.05, -0.04, -0.06, -0.05, -0.03, -0.05, -0.04, -0.06, -0.05, -0.04]
        mean, _, _, equivalent = corrected_fold_equivalence(diffs, test_train_ratio=1 / 9)
        assert mean < -0.02 and not equivalent
