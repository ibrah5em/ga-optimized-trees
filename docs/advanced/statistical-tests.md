# Statistical Testing

Guide to statistical evaluation of GA-optimized trees.

> **⚠️ This page previously recommended an invalid procedure** — a paired t-test across
> cross-validation folds, with a non-significant result reported as equivalence. Both are
> wrong, and both shaped this project's withdrawn results. The page has been rewritten. See
> `paper/CLAIMS.md`.

## Two mistakes to avoid first

**Do not run a paired t-test across CV folds.** Folds of a single cross-validation share
training data, so fold scores are not independent draws. The test's assumptions fail and the
p-value is not interpretable — this is the classic result in Dietterich (1998). The fold count
also inflates apparent precision: 20 folds does not buy 20 independent observations. Compare
methods **across datasets**, where each dataset contributes one paired observation.

**Do not read `p > 0.05` as equivalence.** Failure to reject a null hypothesis is not evidence
for it; a non-significant result is equally consistent with a real difference the test lacked
power to detect. Claiming equivalence requires an equivalence test against a margin chosen in
advance.

## Comparing across datasets

Aggregate to one score per dataset per method, then test across datasets.

```python
import numpy as np
from scipy import stats

# One mean score per dataset, paired by dataset
ga = np.array([0.812, 0.774, 0.903, 0.865, 0.744])
cart = np.array([0.828, 0.767, 0.911, 0.858, 0.759])

stat, p = stats.wilcoxon(ga, cart)
print(f"Wilcoxon signed-rank: statistic={stat:.4f}, p={p:.4f}")
```

For three or more methods, use Friedman followed by a Nemenyi post-hoc test and report a
critical-difference diagram, per Demšar (2006). Correct for multiplicity across datasets —
Holm is a reasonable default.

## Testing for equivalence (TOST)

Two one-sided tests against a pre-specified margin. The margin is a modelling decision and
must be fixed before seeing results; this project uses 2% absolute accuracy.

```python
import numpy as np
from scipy import stats

MARGIN = 0.02  # pre-registered, absolute accuracy

diff = ga - cart
n = len(diff)
se = stats.sem(diff)

# H0a: diff <= -MARGIN   H0b: diff >= +MARGIN
t_lower = (diff.mean() + MARGIN) / se
t_upper = (diff.mean() - MARGIN) / se
p_lower = stats.t.sf(t_lower, df=n - 1)
p_upper = stats.t.cdf(t_upper, df=n - 1)

p_tost = max(p_lower, p_upper)
print(f"TOST p={p_tost:.4f} -> {'equivalent' if p_tost < 0.05 else 'not equivalent'}")
```

Rejecting *both* one-sided nulls supports equivalence within the margin. A Bayesian
correlated t-test with a region of practical equivalence (Benavoli et al. 2017) is a
reasonable alternative and reports directly on the probability of practical equivalence.

## Effect size (Cohen's d)

```python
def cohens_d(scores1, scores2):
    """Cohen's d. Note ddof=1 — the sample estimate, not the biased one."""
    mean_diff = np.mean(scores1) - np.mean(scores2)
    pooled_std = np.sqrt((np.var(scores1, ddof=1) + np.var(scores2, ddof=1)) / 2)
    return mean_diff / pooled_std if pooled_std > 0 else 0.0
```

Report effect size alongside every p-value. Conventional thresholds (0.2 / 0.5 / 0.8) are
rules of thumb, not decision rules.

Use `ddof=1` for every standard deviation and variance you report. NumPy defaults to `ddof=0`,
which biases the estimate low — the withdrawn results on this project were affected.

## Confidence intervals

```python
def confidence_interval(scores, confidence=0.95):
    """Two-sided t interval for the mean."""
    n = len(scores)
    margin = stats.sem(scores) * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (np.mean(scores) - margin, np.mean(scores) + margin)
```

Valid only when the inputs are independent — so across datasets, not across folds of one CV.

## Recommended approach

1. Nested CV: outer 10-fold × 3 repeats for reporting, inner 5-fold for **all** tuning,
   applied identically to every method
1. Aggregate to one score per dataset, then compare across datasets
1. Wilcoxon signed-rank for pairs; Friedman + Nemenyi with a critical-difference diagram for
   three or more methods
1. TOST against a pre-registered margin if the claim is equivalence
1. Holm correction across datasets; report effect sizes and `ddof=1` dispersion throughout
1. Fix hypotheses, margins, and decision thresholds before the run — see
   `paper/PREREGISTRATION.md`

## References

- Dietterich (1998), *Approximate Statistical Tests for Comparing Supervised Classification
  Learning Algorithms*
- Demšar (2006), *Statistical Comparisons of Classifiers over Multiple Data Sets*
- Benavoli et al. (2017), *Time for a Change: a Tutorial for Comparing Multiple Classifiers
  Through Bayesian Analysis*
