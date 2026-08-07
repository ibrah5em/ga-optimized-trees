# Frontier run — 2026-08-07

The run that decides **K1** and **K2** in `paper/PREREGISTRATION.md`. Committed here rather
than left in `results/` (which `.gitignore` excludes) because the outcome section cites
these numbers, and an uncited claim with no committed data behind it is the failure this
whole audit exists to correct — see `paper/CLAIMS.md`.

## Reproducing

```bash
python scripts/frontier_benchmark.py --config configs/paper.yaml --n-jobs 5
```

20 pre-registered CC-18 datasets × 10-fold × 3 repeats, 4 methods, 2400 rows.
Runtime ≈ 35 min on 5 workers. `config.yaml` is the resolved config the run actually used;
`seeds.json` is the per-(dataset, fold, method) seed manifest.

## Files

| File                                      | What it is                                                                                                                                               |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `folds.csv`                               | **The result.** One row per (dataset, method, outer fold): hypervolume, frontier size, delivered candidates, realised evaluation count, reference point. |
| `points.csv`                              | Every non-dominated (accuracy, nodes) point behind those hypervolumes. Enough to recompute the metric under any reference point.                         |
| `seeds.json`                              | Seed manifest plus the protocol block (`budget_mismatched_folds`, `k2_dominance_rate`).                                                                  |
| `config.yaml`                             | Resolved run configuration.                                                                                                                              |
| `folds-INVALID-selection-on-test.csv`     | **Do not use.** See below.                                                                                                                               |
| `folds-SUPERSEDED-per-fold-reference.csv` | **Do not use.** See below.                                                                                                                               |

## Headline

- **K1 — not triggered.** GA vs budget-matched random search: mean Δ hypervolume
  **+0.6616**, p = 0.0107, p_holm **0.0321**, d_z +0.501, ahead on **15/20** datasets.
  Budget match: 0 of 600 folds unequal.
- **K2 — triggered.** GA-over-CART dominance **45%**, below the 60% threshold. H1 rejected.
- Archived-GA control: +0.1156, ns — the result is not an artefact of comparing NSGA-II's
  final-population front against random search's archive.

## The two rejected result sets, and why they are kept

Both were produced by this same script before bugs in it were found. They are committed so
that the outcome section's audit trail points at real files rather than at a description of
files.

**`folds-INVALID-selection-on-test.csv`** — reported K1 TRIGGERED (random search ahead by
−1.585, p_holm \< 0.0001, 19/20). `RandomSearchFrontier` returned all ~2,545 candidates it
sampled and let the dominance filter run on their *test* scores: a maximum over thousands
of test evaluations, which nothing can deliver, since choosing among those candidates needs
the test labels. Random search delivered 2,545 models against the GA's 10. Check the
`n_candidates` column to see it.

**`folds-SUPERSEDED-per-fold-reference.csv`** — computed the hypervolume reference point per
*fold*; the pre-registration fixes it per *dataset*. Raising the reference adds
`max_accuracy × Δreference` to a method's area, so the smaller per-fold box favours whichever
method has the lower peak accuracy — the GA. It put K2's dominance rate at 70% instead of
45%, i.e. on the other side of the threshold. K1's conclusion was unaffected. Check the
`reference_nodes` column: it varies by fold here, and is constant per dataset in `folds.csv`.
