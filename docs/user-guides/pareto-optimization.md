# Pareto Optimization

Explore accuracy-interpretability trade-offs.

## Overview

Pareto optimization finds multiple solutions representing different trade-offs between objectives.

## Running Pareto Analysis

```bash
python scripts/run_pareto_optimization.py --config configs/custom.yaml --dataset breast_cancer
```

**Output:**
- Console: Summary of Pareto solutions
- Figure: Pareto front visualization

## Understanding Results

Example output:
```
Config   Acc Weight  Test Acc      Interp        Nodes  Depth
1        0.30        0.8823        0.7234        8      3
2        0.50        0.9015        0.6543        11     4
3        0.70        0.9234        0.5123        17     5
4        0.90        0.9456        0.3456        28     6

YOUR CONFIG (★): Acc Weight=0.68, Test Acc=0.9105, Nodes=6.5
```

## Interpretation

- **Config 1**: High interpretability, lower accuracy
- **Config 2-3**: Balanced solutions (★ your config)
- **Config 4**: High accuracy, lower interpretability

Choose based on your domain requirements!

See [Multi-Objective Guide](../advanced/multi-objective.md) for implementation details.