# Detailed Results

## Complete Result Tables

### Iris Dataset (20-fold CV)

| Fold     | GA Acc     | CART Acc   | GA Nodes | CART Nodes |
| -------- | ---------- | ---------- | -------- | ---------- |
| 1        | 0.933      | 0.900      | 7        | 16         |
| 2        | 0.967      | 0.933      | 7        | 16         |
| ...      | ...        | ...        | ...      | ...        |
| 20       | 0.933      | 0.967      | 9        | 18         |
| **Mean** | **0.9455** | **0.9241** | **7.4**  | **16.4**   |
| **Std**  | **0.0807** | **0.1043** | **2.1**  | **4.2**    |

**Statistical Test**:

- t-statistic: 1.371
- p-value: 0.186 (not significant)
- Cohen's d: 0.230 (small effect)

### Wine Dataset (20-fold CV)

| Metric   | GA             | CART           |
| -------- | -------------- | -------------- |
| Accuracy | 88.19 ± 10.39% | 87.22 ± 10.70% |
| F1-Score | 87.89 ± 10.63% | 86.87 ± 10.99% |
| Nodes    | 10.7 ± 3.1     | 20.7 ± 5.8     |
| Depth    | 3.0 ± 0.8      | 4.4 ± 1.2      |

**Statistical Test**:

- p-value: 0.683 (not significant)
- Size reduction: 48%

### Breast Cancer (20-fold CV)

| Metric   | GA            | CART          |
| -------- | ------------- | ------------- |
| Accuracy | 91.05 ± 5.60% | 91.57 ± 3.92% |
| Nodes    | 6.5 ± 2.1     | 35.5 ± 4.2    |
| Depth    | 2.3 ± 0.9     | 6.0 ± 1.3     |

**Statistical Test**:

- p-value: 0.640 (not significant)
- Size reduction: 82%

## Interpretation

All datasets show:

1. **No significant accuracy difference** (p > 0.05)
1. **Substantial size reduction** (46-82%)
1. **Maintained interpretability** through small, simple trees

This validates the multi-objective approach: GA successfully balances accuracy and interpretability.
