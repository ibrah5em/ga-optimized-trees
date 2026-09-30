"""Statistical comparison of learners across datasets.

Implements the analysis fixed in ``paper/PREREGISTRATION.md``. The guiding
constraint is Dietterich (1998): cross-validation folds share training data, so
a paired test *across folds* violates the independence assumption and its
p-value means nothing. Inference therefore happens **across datasets**, with
each dataset contributing one paired observation (Demsar 2006).

References
----------
Dietterich (1998), *Approximate Statistical Tests for Comparing Supervised
Classification Learning Algorithms*, Neural Computation 10(7).

Demsar (2006), *Statistical Comparisons of Classifiers over Multiple Data
Sets*, JMLR 7.

Lakens (2017), *Equivalence Tests*, Social Psychological and Personality
Science 8(4) — TOST procedure.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

#: Significance level fixed in paper/PREREGISTRATION.md.
ALPHA = 0.05

#: TOST equivalence margin, in absolute accuracy, fixed in paper/PREREGISTRATION.md.
DEFAULT_EQUIVALENCE_MARGIN = 0.02

#: Below this many datasets, a signed-rank test cannot reach alpha=0.05 no
#: matter how consistent the differences are: the smallest attainable two-sided
#: p-value for n paired observations is 2 / 2**n, which only drops under 0.05
#: at n >= 6. Results below this are reported as descriptive, not inferential.
MIN_DATASETS_FOR_INFERENCE = 6


@dataclass
class Summary:
    """Descriptive statistics for one method on one dataset."""

    mean: float
    std: float
    n: int

    def __str__(self) -> str:
        return "{:.4f} +/- {:.4f}".format(self.mean, self.std)


@dataclass
class PairedComparison:
    """Wilcoxon signed-rank comparison of two methods across datasets."""

    method_a: str
    method_b: str
    n_datasets: int
    mean_difference: float
    statistic: Optional[float]
    p_value: Optional[float]
    p_adjusted: Optional[float] = None
    effect_size: Optional[float] = None
    underpowered: bool = False
    note: str = ""

    @property
    def significant(self) -> bool:
        """True when the Holm-adjusted p-value clears alpha and n is adequate."""
        if self.underpowered or self.p_adjusted is None:
            return False
        return self.p_adjusted < ALPHA


@dataclass
class EquivalenceResult:
    """Paired TOST equivalence test across datasets."""

    method_a: str
    method_b: str
    n_datasets: int
    margin: float
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    equivalent: bool
    note: str = ""


@dataclass
class FriedmanResult:
    """Friedman omnibus test plus Nemenyi critical difference."""

    methods: List[str]
    n_datasets: int
    average_ranks: Dict[str, float] = field(default_factory=dict)
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    critical_difference: Optional[float] = None
    note: str = ""

    def rank_gap(self, method_a: str, method_b: str) -> float:
        """Absolute average-rank distance between two methods."""
        return abs(self.average_ranks[method_a] - self.average_ranks[method_b])

    def differs(self, method_a: str, method_b: str) -> bool:
        """True when two methods are separated by more than the Nemenyi CD."""
        if self.critical_difference is None:
            return False
        return self.rank_gap(method_a, method_b) > self.critical_difference


def summarize(values: Sequence[float]) -> Summary:
    """Mean and *sample* standard deviation of fold scores.

    Uses ``ddof=1``. The population form understates the spread of a
    cross-validation estimate, which is a sample of a larger population of
    possible splits.

    Parameters
    ----------
    values : sequence of float
        Per-fold scores for one method on one dataset.

    Returns
    -------
    Summary
        Mean, sample standard deviation, and count. Standard deviation is 0.0
        for a single observation, where it is undefined.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("summarize() requires at least one value.")
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return Summary(mean=float(np.mean(arr)), std=std, n=int(arr.size))


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjustment.

    Uniformly more powerful than Bonferroni at the same family-wise error rate,
    which is why ``paper/PREREGISTRATION.md`` fixes Holm.

    Parameters
    ----------
    p_values : sequence of float
        Raw p-values from a family of tests.

    Returns
    -------
    list of float
        Adjusted p-values in the input order, each clipped to 1.0 and
        monotonically non-decreasing in the sorted order.
    """
    raw = list(p_values)
    m = len(raw)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: raw[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * raw[idx]
        # Step-down: an adjusted p-value can never drop below an earlier one.
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def cohens_dz(differences: Sequence[float]) -> float:
    """Paired effect size: mean difference over the SD of the differences.

    ``d_z`` is the correct paired-design effect size. It is not comparable to
    the between-groups ``d``, so it is labelled distinctly wherever reported.
    """
    diffs = np.asarray(list(differences), dtype=float)
    if diffs.size < 2:
        return 0.0
    sd = float(np.std(diffs, ddof=1))
    if sd == 0.0:
        return 0.0 if float(np.mean(diffs)) == 0.0 else float(np.inf * np.sign(np.mean(diffs)))
    return float(np.mean(diffs) / sd)


def compare_across_datasets(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    method_a: str = "A",
    method_b: str = "B",
) -> PairedComparison:
    """Wilcoxon signed-rank test on per-dataset scores.

    Each element is one dataset's aggregate score (typically the mean over
    outer folds), so the paired observations are independent — unlike the
    fold-level scores they summarise.

    Parameters
    ----------
    scores_a, scores_b : sequence of float
        Per-dataset scores, aligned by dataset.
    method_a, method_b : str
        Labels used in the returned record.

    Returns
    -------
    PairedComparison
        With ``underpowered=True`` when there are too few datasets for the test
        to reach alpha regardless of the data, in which case ``p_value`` is
        still reported but must not be read as evidence.
    """
    a = np.asarray(list(scores_a), dtype=float)
    b = np.asarray(list(scores_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Score vectors must align: got {a.shape} and {b.shape}.")
    if a.size == 0:
        raise ValueError("compare_across_datasets() requires at least one dataset.")

    diffs = a - b
    n = int(a.size)
    result = PairedComparison(
        method_a=method_a,
        method_b=method_b,
        n_datasets=n,
        mean_difference=float(np.mean(diffs)),
        statistic=None,
        p_value=None,
        effect_size=cohens_dz(diffs),
        underpowered=n < MIN_DATASETS_FOR_INFERENCE,
    )

    if np.allclose(diffs, 0.0):
        result.note = "All per-dataset differences are zero; no test performed."
        return result

    try:
        statistic, p_value = stats.wilcoxon(a, b)
    except ValueError as exc:  # e.g. every difference is zero after trimming
        result.note = f"Wilcoxon not computable: {exc}"
        return result

    result.statistic = float(statistic)
    result.p_value = float(p_value)
    if result.underpowered:
        result.note = (
            f"n={n} datasets is below the {MIN_DATASETS_FOR_INFERENCE} needed for a "
            "signed-rank test to reach alpha=0.05; descriptive only."
        )
    return result


def equivalence_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    margin: float = DEFAULT_EQUIVALENCE_MARGIN,
    method_a: str = "A",
    method_b: str = "B",
    alpha: float = ALPHA,
) -> EquivalenceResult:
    """Two one-sided tests (TOST) for equivalence within ``margin``.

    A non-significant difference test is *not* evidence of equivalence. TOST
    inverts the question: it rejects the null of a difference at least as large
    as ``margin``, which is what H2 in ``paper/PREREGISTRATION.md`` claims.

    Parameters
    ----------
    scores_a, scores_b : sequence of float
        Per-dataset scores, aligned by dataset.
    margin : float
        Equivalence bound in the units of the scores (0.02 = 2% absolute
        accuracy). Must be positive.
    alpha : float
        One-sided significance level for each of the two tests.

    Returns
    -------
    EquivalenceResult
        ``equivalent`` is True only when both one-sided tests reject, i.e. the
        (1 - 2*alpha) confidence interval of the mean difference lies entirely
        inside (-margin, +margin).
    """
    if margin <= 0:
        raise ValueError(f"margin must be > 0, got {margin}.")

    a = np.asarray(list(scores_a), dtype=float)
    b = np.asarray(list(scores_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Score vectors must align: got {a.shape} and {b.shape}.")

    diffs = a - b
    n = int(diffs.size)
    mean_diff = float(np.mean(diffs)) if n else 0.0

    if n < 2:
        return EquivalenceResult(
            method_a=method_a,
            method_b=method_b,
            n_datasets=n,
            margin=margin,
            mean_difference=mean_diff,
            ci_low=float("nan"),
            ci_high=float("nan"),
            p_value=1.0,
            equivalent=False,
            note="TOST needs at least 2 datasets.",
        )

    sd = float(np.std(diffs, ddof=1))
    df = n - 1

    if sd == 0.0:
        # Degenerate but well-defined: an exactly constant difference is
        # equivalent iff it sits inside the margin.
        equivalent = abs(mean_diff) < margin
        return EquivalenceResult(
            method_a=method_a,
            method_b=method_b,
            n_datasets=n,
            margin=margin,
            mean_difference=mean_diff,
            ci_low=mean_diff,
            ci_high=mean_diff,
            p_value=0.0 if equivalent else 1.0,
            equivalent=equivalent,
            note="Zero variance in per-dataset differences.",
        )

    se = sd / np.sqrt(n)
    # H0_lower: true difference <= -margin. H0_upper: true difference >= +margin.
    p_lower = float(stats.t.sf((mean_diff + margin) / se, df))
    p_upper = float(stats.t.cdf((mean_diff - margin) / se, df))
    p_tost = max(p_lower, p_upper)

    half_width = float(stats.t.ppf(1 - alpha, df)) * se
    ci_low = mean_diff - half_width
    ci_high = mean_diff + half_width

    return EquivalenceResult(
        method_a=method_a,
        method_b=method_b,
        n_datasets=n,
        margin=margin,
        mean_difference=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_tost,
        equivalent=p_tost < alpha,
    )


def corrected_fold_equivalence(
    fold_differences: Sequence[float],
    test_train_ratio: float,
    margin: float = DEFAULT_EQUIVALENCE_MARGIN,
    alpha: float = ALPHA,
) -> Tuple[float, float, float, bool]:
    """Per-dataset TOST over outer-fold differences, Nadeau–Bengio corrected.

    Outer folds share training rows, so the naive variance of their differences
    understates the true variance and a plain paired t-test over folds is
    anti-conservative (Dietterich 1998). Nadeau and Bengio (2003) inflate it by
    ``1/k + n_test/n_train``. This is the secondary K3 reading fixed in
    ``paper/PREREGISTRATION.md``; with ten folds it has little power.

    Returns:
        ``(mean_difference, ci_low, ci_high, equivalent)`` where the interval
        is the ``1 - 2*alpha`` interval that TOST inverts.
    """
    d = np.asarray(list(fold_differences), dtype=float)
    k = int(d.size)
    mean = float(d.mean()) if k else 0.0
    if k < 2:
        return mean, float("nan"), float("nan"), False
    variance = (1.0 / k + float(test_train_ratio)) * float(np.var(d, ddof=1))
    if variance == 0.0:
        return mean, mean, mean, abs(mean) < margin
    half = float(stats.t.ppf(1 - alpha, k - 1)) * float(np.sqrt(variance))
    low, high = mean - half, mean + half
    return mean, low, high, bool(low > -margin and high < margin)


def friedman_nemenyi(
    scores_by_method: Dict[str, Sequence[float]], alpha: float = ALPHA
) -> FriedmanResult:
    """Friedman omnibus test with Nemenyi critical difference.

    The standard Demsar (2006) protocol for comparing several methods over
    several datasets: rank the methods per dataset, test whether the average
    ranks differ, then read pairwise differences off the critical difference.

    Parameters
    ----------
    scores_by_method : dict of str to sequence of float
        Per-dataset scores for each method, all aligned to the same dataset
        order. Higher is better.
    alpha : float
        Significance level for the Nemenyi critical difference.

    Returns
    -------
    FriedmanResult
        ``statistic``/``p_value`` are None when the omnibus test is not
        applicable (fewer than 3 methods or fewer than 3 datasets); average
        ranks are still reported.
    """
    methods = list(scores_by_method.keys())
    if not methods:
        raise ValueError("friedman_nemenyi() requires at least one method.")

    matrix = np.asarray([list(scores_by_method[m]) for m in methods], dtype=float)
    if matrix.ndim != 2:
        raise ValueError("All methods must have the same number of per-dataset scores.")

    k, n = matrix.shape

    # Rank per dataset, 1 = best. Negate so that higher scores rank lower.
    ranks = np.apply_along_axis(stats.rankdata, 0, -matrix)
    average_ranks = {m: float(np.mean(ranks[i])) for i, m in enumerate(methods)}

    result = FriedmanResult(methods=methods, n_datasets=int(n), average_ranks=average_ranks)

    if k < 3:
        result.note = "Friedman needs at least 3 methods; average ranks reported only."
        return result
    if n < 3:
        result.note = "Friedman needs at least 3 datasets; average ranks reported only."
        return result

    try:
        statistic, p_value = stats.friedmanchisquare(*[matrix[i] for i in range(k)])
    except ValueError as exc:
        result.note = f"Friedman not computable: {exc}"
        return result

    result.statistic = float(statistic)
    result.p_value = float(p_value)
    # CD = q_alpha * sqrt(k(k+1) / 6N), with q_alpha the studentized range
    # statistic at infinite df divided by sqrt(2) (Demsar 2006, eq. 6).
    q_alpha = float(stats.studentized_range.ppf(1 - alpha, k, np.inf)) / np.sqrt(2)
    result.critical_difference = float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * n)))
    return result


def compare_all_to_reference(
    scores_by_method: Dict[str, Sequence[float]], reference: str
) -> List[PairedComparison]:
    """Wilcoxon-compare every method against ``reference``, Holm-corrected.

    Parameters
    ----------
    scores_by_method : dict of str to sequence of float
        Per-dataset scores per method, aligned by dataset.
    reference : str
        Key of the method every other method is compared against.

    Returns
    -------
    list of PairedComparison
        One entry per non-reference method, with ``p_adjusted`` filled in by
        Holm correction over the family of comparisons.
    """
    if reference not in scores_by_method:
        raise ValueError(f"Reference method '{reference}' not in {list(scores_by_method)}.")

    others = [m for m in scores_by_method if m != reference]
    comparisons = [
        compare_across_datasets(
            scores_by_method[reference], scores_by_method[m], method_a=reference, method_b=m
        )
        for m in others
    ]

    testable = [c for c in comparisons if c.p_value is not None]
    if testable:
        adjusted = holm_adjust([c.p_value for c in testable])
        for comparison, p_adj in zip(testable, adjusted):
            comparison.p_adjusted = p_adj
    return comparisons


def per_dataset_means(
    all_results: Dict[str, Dict[str, Dict[str, Sequence[float]]]],
    metric: str = "test_acc",
) -> Tuple[List[str], Dict[str, List[float]]]:
    """Collapse fold-level results to one score per (method, dataset).

    Parameters
    ----------
    all_results : nested dict
        ``{dataset: {method: {metric: [per-fold values]}}}``, the structure
        produced by ``scripts/experiment.py``.
    metric : str
        Key of the per-fold metric to average.

    Returns
    -------
    tuple
        ``(dataset_names, {method: [mean per dataset]})``, restricted to the
        methods present on *every* dataset so the pairing stays valid.
    """
    datasets = list(all_results.keys())
    if not datasets:
        return [], {}

    shared = set(all_results[datasets[0]].keys())
    seen = set(all_results[datasets[0]].keys())
    for dataset in datasets[1:]:
        methods = set(all_results[dataset].keys())
        shared &= methods
        seen |= methods

    dropped = seen - shared
    if dropped:
        logger.warning(
            "Excluding %s from across-dataset tests: not present on every dataset.",
            ", ".join(sorted(dropped)),
        )

    scores = {
        method: [float(np.mean(all_results[d][method][metric])) for d in datasets]
        for method in sorted(shared)
        if all(metric in all_results[d][method] for d in datasets)
    }
    return datasets, scores
