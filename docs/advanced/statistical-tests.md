# Statistical Testing

Guide to rigorous statistical evaluation of GA-optimized trees.

## Overview

Statistical tests validate that observed performance differences are significant and not due to chance.

## Paired t-Test

```python
from scipy import stats
import numpy as np

# Collect scores from k-fold CV
ga_scores = [0.945, 0.933, 0.967, ...]  # 20 scores
cart_scores = [0.924, 0.900, 0.933, ...]  # 20 scores

# Paired t-test (same folds for both models)
t_stat, p_value = stats.ttest_rel(ga_scores, cart_scores)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value > 0.05:
    print("No significant difference (statistically equivalent)")
else:
    print("Significant difference detected")
```

## Effect Size (Cohen's d)

```python
def cohens_d(scores1, scores2):
    """Calculate Cohen's d effect size."""
    mean_diff = np.mean(scores1) - np.mean(scores2)
    pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
    return mean_diff / pooled_std if pooled_std > 0 else 0.0

d = cohens_d(ga_scores, cart_scores)
print(f"Cohen's d: {d:.4f}")

# Interpretation
if abs(d) < 0.2:
    print("Negligible effect size")
elif abs(d) < 0.5:
    print("Small effect size")
elif abs(d) < 0.8:
    print("Medium effect size")
else:
    print("Large effect size")
```

## Confidence Intervals

```python
from scipy import stats

def confidence_interval(scores, confidence=0.95):
    """Calculate confidence interval."""
    n = len(scores)
    mean = np.mean(scores)
    std_err = stats.sem(scores)
    margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - margin, mean + margin)

ci = confidence_interval(ga_scores)
print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
```

## Recommended Approach

For research-quality results:

1. Use **20-fold cross-validation**
2. Calculate **paired t-test** (p-value)
3. Calculate **Cohen's d** (effect size)
4. Report **mean ± std** for all metrics

Example from our results:
```
Breast Cancer:
  GA: 91.05% ± 5.60%, Nodes: 6.5
  CART: 91.57% ± 3.92%, Nodes: 35.5
  p-value: 0.640 (not significant)
  Cohen's d: -0.108 (negligible)
  Conclusion: Statistically equivalent accuracy, 82% smaller trees
```