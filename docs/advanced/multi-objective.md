# Multi-Objective Optimization

Guide to Pareto-based optimization for accuracy-interpretability trade-offs.

## Overview

Multi-objective optimization finds a set of Pareto-optimal solutions representing different trade-offs between objectives (accuracy vs interpretability).

## Using the Pareto Optimizer

```bash
# Run Pareto optimization
python scripts/run_pareto_optimization.py --config configs/paper.yaml --dataset breast_cancer
```

## Understanding Pareto Fronts

A solution is Pareto-optimal if improving one objective requires sacrificing another.

```
    Accuracy ↑
       |
    95%|     ● (High acc, low interp)
       |
    90%|   ●   ● (Pareto front)
       |
    85%| ●       (High interp, lower acc)
       |_____________
            Interpretability →
```

## Example Results

The Pareto script explores different weight combinations:

| Solution | Accuracy | Nodes | Depth | Weight Ratio |
|----------|----------|-------|-------|--------------|
| 1 | 94.2% | 15 | 5 | 90/10 (accuracy focused) |
| 2 | 92.8% | 10 | 4 | 70/30 (balanced) |
| 3 | 90.1% | 6 | 3 | 50/50 (interpretability focused) |

## Next Steps

See `scripts/run_pareto_optimization.py` for implementation details.
