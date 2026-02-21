# Benchmark Results

Complete benchmark results across datasets.

## Summary

| Dataset | GA Acc | CART Acc | p-value | GA Nodes | CART Nodes | Reduction |
|---------|--------|----------|---------|----------|------------|-----------|
| Iris | 94.55% | 92.41% | 0.186 | 7.4 | 16.4 | 55% |
| Wine | 88.19% | 87.22% | 0.683 | 10.7 | 20.7 | 48% |
| Breast Cancer | 91.05% | 91.57% | 0.640 | 6.5 | 35.5 | 82% |

All results use **20-fold cross-validation** with `configs/paper.yaml`.

## Key Findings

1. **Statistical Equivalence**: All p-values > 0.05
2. **Size Reduction**: 46-82% smaller trees than CART
3. **Interpretability**: Explicit control via fitness weights