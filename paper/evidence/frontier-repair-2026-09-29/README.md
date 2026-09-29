# Frontier run with constraint repair — 2026-09-29 (sensitivity analysis)

`configs/paper-repair.yaml`: identical to the pre-registered `configs/paper.yaml` except
`tree.repair_constraints: true` (`src/ga_trees/ga/repair.py`). Same datasets, folds, seeds and
budget rule as `../frontier-2026-08-07/`.

**Status: sensitivity only.** Repair was implemented after K1/K2 were observed
(`paper/PREREGISTRATION.md`, Deviations, 2026-09-29). It cannot replace or overturn the
pre-registered verdicts; it is reported to show whether they depend on the constraint bug.

```bash
python scripts/frontier_benchmark.py --config configs/paper-repair.yaml --n-jobs 4
```

## Result

| Test                                  | Pre-registered (repair off)  | This run (repair on)          |
| ------------------------------------- | ---------------------------- | ----------------------------- |
| K1: GA − random search                | +0.662, p_holm 0.032, 15/20  | +0.846, p_holm 0.034          |
| K2: GA dominance over CART ccp path   | 45% (triggered)              | 45% (triggered)               |
| GA − archived GA                      | +0.116, p_holm 0.286 (ns)    | +0.544, p_holm 0.025          |
| Friedman over four methods            | p = 0.202                    | p = 0.041                     |

Neither verdict changes. Budget match: 0 of 600 folds unequal. `run.log` is the full console
output.
