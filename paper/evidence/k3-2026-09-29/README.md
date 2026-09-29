# K3 / H2 point-estimate run — 2026-09-29

The run that decides **K3** and **H2** in `paper/PREREGISTRATION.md`.

```bash
python scripts/benchmark.py --config configs/paper.yaml --outer-repeats 1 --no-depth-tuning --n-jobs 4
python scripts/k3_analysis.py paper/evidence/k3-2026-09-29/folds.csv
```

20 pre-registered datasets × 10 outer folds (1 repeat), 5-fold inner CV for every method.
Methods: GA (single-objective, tuned over accuracy weight {0.5, 0.7, 0.9} at depth 6),
budget-matched random search (same grid, 2,920 evaluations per fit), CART with
cost-complexity pruning (tuned over depth {3, 4, 5, 6, 8} × `ccp_alpha` on its path),
unconstrained CART, random forest. Both deviations (1 outer repeat; weight-only GA/RS grid)
were recorded before the run.

| File          | Contents                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------- |
| `folds.csv`   | One row per (dataset, method, outer fold): accuracy, F1, selected params, nodes, leaves, depth, features used, mean path length, evaluations |
| `k3-table.csv`| Per-dataset GA − CART difference, Nadeau–Bengio 90% interval, both K3 readings           |
| `stats.csv`   | Across-dataset Wilcoxon and TOST from `benchmark.py`                                     |
| `seeds.json`  | Per-(dataset, fold, method) seeds and protocol block                                     |
| `config.yaml` | Resolved configuration                                                                   |
| `run.log`     | Console output                                                                           |

**Outcome:** K3 triggered (8/20 datasets lose > 2 points to tuned CART; the corrected TOST
fails on 18/20). H2 rejected: GA − CART = −0.0385, 90% CI [−0.0694, −0.0076].

A first attempt at the full weight × depth grid was lost to a container restart after 5 of
20 datasets (`benchmark.py` then wrote nothing until the end). It now checkpoints each
dataset under `partial/`; no result from the lost attempt was seen or used.
