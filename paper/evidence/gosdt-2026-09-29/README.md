# GOSDT frontier — 2026-09-29 (exploratory)

`scripts/gosdt_frontier.py`: GOSDT (`gosdt` 1.0.4) over ten regularisation values
(0.002–0.12), depth budget 7 (= max depth 6), threshold-guessed binarisation from a
40-stump gradient-boosted ensemble fitted on the training fold, 30 s per fit. Run on the
**first 10 outer folds** (repeat 1) of the committed split. **Added after K2 was known;
decides nothing.**

| File               | Contents                                                                             |
| ------------------ | ------------------------------------------------------------------------------------ |
| `gosdt-points.csv` | GOSDT test fronts, with per-fold fit time, timed-out fits and failed fits            |
| `gosdt-folds.csv`  | Hypervolume for every method on those folds, one reference per dataset over all five |
| `incomplete.txt`   | Datasets on which GOSDT did not complete                                             |

Outcome: GOSDT completed 19 of 20 datasets. On `qsar_biodeg` (41 features) its search
queue outgrew memory within the time limit and the process segfaulted, under an 8 GB cap
and again under 11 GB; that dataset is excluded, not scored. Across the rest, 32 of 1,900
fits timed out (so are not certified optimal) and 10 raised "false convergence, no model
was found" and were dropped from their path, which can only lower GOSDT's hypervolume.

Reproduce:

```bash
python scripts/gosdt_frontier.py --n-jobs 2 --time-limit 30 --memory-gb 8 --output-dir results/gosdt
python scripts/gosdt_frontier.py --aggregate-only --output-dir results/gosdt
```
