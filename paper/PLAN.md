# Publication Plan — ga-optimized-trees

**Started:** 2026-08-04
**Targets:** JOSS (software, in parallel) + GECCO or Applied Soft Computing / SWEVO (research)
**Claim:** evolution beats budget-matched random search over the same tree space (H3).
Frontier dominance over CART was rejected by K2 on 2026-08-07; accuracy parity was retired
earlier. See below.

______________________________________________________________________

## Why this plan exists

Two mutually inconsistent result sets live in this repo, and the public-facing claim is
the older, smaller, unreproducible one. See `CLAIMS.md` for the audit. The plan below
rebuilds the experimental protocol, fixes the algorithm, and replaces the claim with one
the architecture can actually support.

**Retired claim:** "46–82% smaller trees with statistically equivalent accuracy."

**Target claim — also retired, 2026-08-07, by its own kill criterion:**

> ~~A multi-objective evolutionary search traces the accuracy–complexity frontier for
> decision trees in a single run, dominating (by hypervolume) the frontier obtainable
> from CART's cost-complexity pruning path~~, and admitting non-decomposable objectives
> that greedy and exact methods cannot express.

**K2 fired.** The GA's frontier has the larger hypervolume than CART's `ccp_alpha` path on
**45%** of the 20 pre-registered datasets, below the 60% threshold. H1 is rejected, and the
pre-registration forbids weakening it to "competitive on some datasets."

**What survives — and it is narrower than what this project set out to prove:**

> Over the same tree space and an exactly matched evaluation budget, evolutionary search
> recovers a better accuracy–complexity frontier than random sampling (+0.66 hypervolume,
> p = 0.032 Holm-corrected, 15/20 datasets, d_z = 0.50), and the representation admits
> non-decomposable objectives that greedy induction cannot express.

That is H3 — a mechanism claim about evolution versus random search, not a claim to beat
CART. It is publishable as a contribution to the evolutionary-tree-induction literature
(Barros et al. 2012 is full of methods that never established this much), but it is not
the frontier-dominance headline, and the paper must not be written as though it were.

Full outcome and the two harness errors found along the way: `paper/PREREGISTRATION.md`.

______________________________________________________________________

## Phase 0 — Stop the bleeding — **COMPLETE** (2026-08-06)

- [x] `git tag pre-paper-audit` on current `main`
- [x] Complete `paper/CLAIMS.md` audit — every public claim marked SUPPORTED, STALE,
  UNSUPPORTED, FALSE or FABRICATED, with the file and line it lives on
- [x] Strip unsupported numbers from `README.md`, `docs/research/benchmarks.md`,
  `docs/research/results.md`, `docs/research/methodology.md` (`eaea3c4`)
- [x] Delete `results/tables/paper-results.csv` and 15 other unattributable artifacts;
  `results/PROVENANCE.md` records what went and why (`70d3fdb`)
- [x] Remove the "Target: 24-77% smaller trees" header comment from `configs/paper.yaml`

**Exit criterion met:** no claim is public that the code on `main` cannot reproduce.

- [x] **`scripts/visualize_comprehensive.py` rebuilt** (2026-08-07) — the last item, closed.
  The `RESULTS`/`PAPER_RESULTS` dicts are gone along with every figure that consumed them:
  `create_statistical_equivalence` rendered a chart captioned "Statistical Equivalence to
  CART (All p > 0.05 = No Significant Difference)" from cross-fold paired t-tests the
  project retired as invalid in `0d446e7`, complete with an orange reference line for a
  "target p-value" of 0.55. That is not a figure with stale numbers in it; it is a figure
  of a claim that may not be made.

  Replaced by `src/ga_trees/evaluation/figures.py` (tested) plus a thin CLI. It reads the
  fold-level CSV from `scripts/benchmark.py` and **raises** when there is none — there is no
  default data anywhere in the module, which is the property the old file lacked. Draws:
  accuracy against leaf count, per-dataset accuracy deltas (labelled descriptive, no
  significance annotation), and the Nemenyi critical-difference diagram.

______________________________________________________________________

## Phase 1 — Rebuild the protocol — **COMPLETE** (2026-08-07)

Built as `src/ga_trees/benchmark/` with `scripts/benchmark.py` as the entry point, rather
than by rewriting `scripts/experiment.py` in place: the harness that produces paper numbers
belongs in the tested package, not in a script CI runs for smoke coverage.
`scripts/experiment.py` stays as the flat-CV screening path and nothing from it belongs in
the paper.

- [x] **Nested CV** — `benchmark/nested_cv.py`. Outer `RepeatedStratifiedKFold(10, 3)`,
  inner 5-fold via `select_hyperparameters`, applied through one `BenchmarkMethod`
  interface so no method can be tuned more favourably than another by accident. Folds are
  1-indexed to match `build_seed_manifest`, and every method is seeded per fold from
  `derive_fold_seed(base, dataset, fold, method)` — sharing a stream across methods would
  correlate their results.

- [x] **Budget-matched baselines** — `benchmark/methods.py`:

  - `PrunedCARTMethod`: `ccp_alpha` taken from the data's own cost-complexity pruning path
    (capped at 12 values) crossed with a depth grid, selected by inner CV
  - `RandomTreeSearch`: same `TreeInitializer`, same `FitnessCalculator`, same param grid,
    same evaluation budget — the only difference from the GA is that there is no selection,
    crossover or mutation, so any gap is attributable to the evolutionary machinery
  - `UnconstrainedCARTMethod`: grown to purity, nothing tuned, as the single-tree ceiling
  - **The budget formula is not `population_size × n_generations`.** Elites carry their
    fitness across generations, so a run costs `pop + gens × (pop − n_elite)`. The naive
    product under-funded random search by ~9% in a smoke run, which would have biased K1
    toward the GA. `verify_budget_match` now reports realised counts per run rather than
    trusting the config; the smoke runs come back at 0.000% spread.

- [x] **Seed everything** — landed earlier in `68b9d70`. `derive_fold_seed` reaches
  `GAConfig.random_state`; `seeds-*.json` is written next to every result set. The note
  about `scripts/experiment.py:197` was stale.

- [x] **Statistics done properly** — `src/ga_trees/evaluation/statistics.py`, wired into
  `scripts/experiment.py` via `run_statistical_analysis()` (2026-08-07):

  - Wilcoxon signed-rank across *datasets*, Holm-corrected; Friedman + Nemenyi critical
    difference (Demšar 2006). The CD **diagram** is still to draw — the number is computed.
  - Equivalence via **TOST** at the pre-registered 2% absolute-accuracy margin. The
    Bayesian correlated t-test with a ROPE (Benavoli et al. 2017) is not implemented; TOST
    is what H2 is written against.
  - `ttest_rel` across CV folds is gone, along with every significance star it printed.
    Per-dataset differences are still shown, labelled descriptive, with no p-value.
  - `ddof=1` throughout, via `summarize()`.
  - Comparisons below `MIN_DATASETS_FOR_INFERENCE` (6) report `significant=False`
    regardless of p, because a signed-rank test on fewer datasets cannot reach α=0.05.
    The default 3-dataset config can no longer produce a significant result — by design.

- [x] **Scale to 20 datasets** — `paper/DATASETS.md` pre-registers them, and
  `configs/paper.yaml` now points at that list instead of iris/wine/breast_cancer. Every ID
  was resolved against the live OpenML API and confirmed to be in study 99 (CC-18);
  selection was by a rule stated before selection (500–1500 rows, ≤50 features, smallest
  class ≥ 40 so stratified 10-fold is valid without silently reducing folds).

  Verifying those IDs turned up two entries that had been serving the wrong data for the
  life of the project: `heart` pointed at OpenML 4, which is `labor` (57 rows), and
  `mammographic` pointed at OpenML 310, which is `mammography` (11183 rows). Both are
  corrected. No withdrawn claim depended on either — `CLAIMS.md` traces every published
  number to iris, wine or breast_cancer — but `--dataset heart` silently trained on
  labour-relations data, and that is the same class of failure as the rest of this audit.

  Only 8 of the loader's 15 OpenML entries were CC-18 members at all. `ionosphere`,
  `sonar`, `hepatitis`, `titanic`, `credit_fraud` and `mammography` are kept for
  exploration and excluded from the benchmark.

- [x] **Wire the config properly** — landed earlier in `68b9d70`. `classification_metric`,
  `early_stopping_rounds` and `early_stopping_tol` all reach the engine. Also stale.

**Exit criterion met:** `python scripts/benchmark.py --config configs/paper.yaml`
reproduces every number, writing fold-level results, the resolved config, the seed
manifest and the statistics table side by side. `--dry-run` reports the fit count first.

**Not yet run.** The full protocol is 20 datasets × 30 outer folds × 5 inner folds × grid
size, which is many CPU-hours and a deliberate launch, not a side effect of building the
harness. Phase 1 delivers the apparatus; Phase 3 produces the numbers.

**Early signal, and it is the uncomfortable one.** A 3-dataset screening run
(`banknote`, `wdbc`, `tic_tac_toe`, no inner tuning, `configs/fast.yaml`) put random search
within 0.4 accuracy points of the GA — mean difference −0.0043, `d_z` = −0.205 — while both
lost to inner-CV-tuned CART by roughly 9 points. Three datasets without tuning is not
evidence and the harness correctly refuses to call it significant. But it is the direction
K1 exists to catch, and it should be treated as the expected outcome until a powered run
says otherwise.

______________________________________________________________________

## Phase 2 — Make the algorithm competitive (~2–3 weeks)

One branch per item, each with an ablation entry. The ablation table is a paper section.

| #   | Change                                                                                                                                                | Where                                                        | Status |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ------ |
| 1   | **Validation-based fitness** — split outer-train into GA-train/GA-val; fit leaves on GA-train, score on GA-val; select champion on validation fitness | `benchmark/methods.py` `holdout_split`, `engine.py` `evolve` | done   |
| 2   | **Data-driven split points** — thresholds from observed midpoints of samples reaching the node, not `uniform(feature_min, feature_max)`               | `ga/split_points.py`, `engine.py`                            | done   |
| 3   | **Greedy seeding** — initialize ~20% of the population with CART trees on bootstrap samples at varying depths                                         | `engine.py` `TreeInitializer`                                | **closed — not pursued** (see below) |
| 4   | **Constraint repair** — re-check `min_samples_leaf`/`min_samples_split` against data after crossover and mutation (currently enforced only at init)   | `engine.py:357-381`, `improved_crossover.py`                 | **done** 2026-09-29, off by default |
| 5   | **Memetic local search** — cheap threshold hill-climb on the elite fraction each generation                                                           | `engine.py` `evolve`                                         | **closed — not pursued** (see below) |
| 6   | **Fix Pareto objectives** to (validation accuracy, −node_count); report hypervolume + attainment surfaces vs CART's `ccp_alpha` path                  | `benchmark/frontiers.py`                                     | done   |

______________________________________________________________________

### Items 3–5 closed (2026-09-29)

**Item 4 is done** as a correctness fix: `ga_trees.ga.repair`, switched on by
`tree.repair_constraints`, collapses every split the sample-count constraints forbid after
crossover and mutation. On one fold of four datasets 8–27% of the trees NSGA-II evaluates
break a constraint (random search: 0%), but only 1 of 79 internal nodes in the delivered
fronts does — the size objective prunes dead splits by itself. Repair stays **off** in
`configs/paper.yaml` so the committed run reproduces bit for bit; `configs/paper-repair.yaml`
is the sensitivity variant.

**Items 3 and 5 are closed without implementation.** Both exist to make the GA more
competitive, and both were still open when K2 fired. Implementing them now and re-running
would be tuning the method after seeing the verdict — the forking path the pre-registration
exists to close. They are listed in the paper as hypotheses for a new pre-registered study,
not as fixes.

### The point-estimate harness cannot decide K1 (2026-08-07)

Worth stating plainly, because it was nearly missed: **K1 and H1/K2 are written on
hypervolume**, and `scripts/benchmark.py` reports one tuned operating point per method per
fold. A hypervolume needs a *set* of models. The accuracy run answers H2/K3 and gives a
useful accuracy-side reading of H3, but on its own it cannot trigger or clear K1.

`src/ga_trees/benchmark/frontiers.py` + `scripts/frontier_benchmark.py` close that gap:

- **`ParetoGAFrontier`** runs NSGA-II over **(accuracy, −node count)** — item 6. The
  shipped objective pair was (accuracy, composite interpretability) on resubstitution data,
  and K4 forbids the composite score as a reported outcome, so a hypervolume against it
  would not have been the pre-registered measurement.
- **`RandomSearchFrontier`** keeps its whole non-dominated set rather than a single best.
  Comparing a frontier against a point would guarantee the GA wins and would not be a test.
- **Budget matching is measured, not predicted.** NSGA-II's per-generation cost depends on
  how many offspring crossover and mutation actually invalidated, so the GA runs first and
  random search is handed its realised count. Smoke runs report 0 mismatched folds out of
  9 and 0 out of 27.
- The runner refuses to read a null result from an underpowered run as K1 being triggered.

**One asymmetry is left in and documented, not fixed.** The GA's frontier is its final
population's front; random search's is an archive over everything it sampled. A point the
GA found in generation 3 and lost by generation 20 does not count for it. This handicaps
the GA — which is why it stays: giving NSGA-II an external archive changes the algorithm,
and doing it after seeing an unfavourable result would be indefensible. If an archived
variant is ever run it is an ablation row, not a replacement.

**Result — K1 clears, K2 fires.** Full protocol, 20 datasets × 30 folds, 2026-08-07.
Outcome in `paper/PREREGISTRATION.md`; evidence in `paper/evidence/frontier-2026-08-07/`.

| Comparison (Wilcoxon across 20 datasets, Holm) | Mean Δ hypervolume | p_holm     | d_z    |
| ---------------------------------------------- | ------------------ | ---------- | ------ |
| GA vs **Random Search**                        | **+0.6616**        | **0.0321** | +0.501 |
| GA vs GA (archived)                            | +0.1156            | 0.286 (ns) | +0.153 |
| GA vs CART (ccp path)                          | −4.2076            | 0.368 (ns) | −0.361 |

The GA beats budget-matched random search on **15 of 20** datasets, significantly, at a
medium effect size — **K1 does not trigger**. The four-method Friedman omnibus is not
significant (p = 0.2018), so the Nemenyi post-hoc is not licensed; K1 is defined on the
pairwise signed-rank test and both are reported.

**K2 triggers.** GA-over-CART dominance is 45%, below the 60% threshold. See the retired
target claim at the top of this file.

The archived-GA control cleared (+0.116, ns), so K1 is not an artefact of comparing
NSGA-II's final-population front against random search's archive.

**Two harness errors were found and both are recorded in `PREREGISTRATION.md`,** because a
kill criterion firing is the moment a project is most tempted to go bug-hunting and least
trustworthy when it succeeds:

1. **Selection on the test fold**, favouring random search. The first run reported K1
   TRIGGERED (−1.585, p_holm \< 0.0001, 19/20) because `RandomSearchFrontier` returned all
   ~2,545 sampled trees and let the dominance filter run on their *test* scores — a maximum
   over thousands of test evaluations nothing can deliver. Discarded; output preserved at
   `paper/evidence/frontier-2026-08-07/folds-INVALID-selection-on-test.csv`.
1. **Reference point per fold instead of per dataset**, favouring the GA. Found by reading
   the implementation against the "Fixed in advance" table, not prompted by the result. It
   moved K2 from 70% to 45% — across the threshold. Output preserved at
   `paper/evidence/frontier-2026-08-07/folds-SUPERSEDED-per-fold-reference.csv`.

______________________________________________________________________

### The screening signal was a strawman — but not for the reason we thought (2026-08-07)

Items 1 and 2 were built first on the argument that the GA drew thresholds from
`uniform(feature_min, feature_max)` while CART searches observed split points, and that
this handicap suppressed the GA and random search equally, making them look alike.

Both changes landed. **Neither is the handicap.** Measured on the same three screening
datasets (`banknote`, `wdbc`, `tic_tac_toe`, 3 outer folds, `configs/fast.yaml`):

| Arm                                 | GA − CART (pruned) | GA − Random Search |
| ----------------------------------- | ------------------ | ------------------ |
| uniform + resubstitution, no tuning | −0.0791            | +0.0068            |
| midpoint + validation, no tuning    | −0.0981            | +0.0134            |

The real handicap is the **fitness weighting**, and it is arithmetic, not search. At
`fast.yaml`'s weights (accuracy 0.65 / interpretability 0.35, `node_complexity` 0.6 within
that, `max_depth` 5 so `max_nodes` = 63) a tree must gain **22.6 accuracy points** to make
growing from a stump to CART's ~47 nodes worth it. At `paper.yaml` it is still 8.1 points.
The GA obliges: it converges to **2.7 leaves at depth 1.4** while tuned CART uses 24
leaves, and then "loses" on accuracy by 8-10 points.

So the screening run was comparing two methods at opposite ends of the complexity axis and
reading the difference as search quality. That is the same error the retired
"equivalent accuracy at 46-82% smaller" claim made, pointing the other way.

It also explains the random-search tie without reference to thresholds: **at 2-3 leaves the
reachable tree space is tiny**, so uniform sampling finds its best member about as easily
as evolution does. A search comparison run at that operating point cannot detect a
difference that exists anywhere else.

Inner CV tuning is the existing escape hatch — the grid runs `accuracy_weight` over
(0.5, 0.7, 0.9) and `max_depth` over (4, 6, 8), and at 0.9/depth-8 the required gain falls
to ~0.5%. Turning it on changes the answer:

| Arm                             | GA − CART (pruned) | GA − Random Search |
| ------------------------------- | ------------------ | ------------------ |
| uniform + resubstitution, tuned | −0.0329            | **+0.0228**        |
| midpoint + validation, tuned    | −0.0534            | +0.0081            |

**`--no-tune` screening runs are not evidence about the algorithm** and should not be
cited as such — including the 3-dataset signal recorded under Phase 1 above, which stands
corrected by this.

### Ablation: the 2×2 over items 1 and 2 (2026-08-07)

Same three screening datasets, tuned, 3 outer folds. Both changes are config-gated
(`tree.split_strategy`, `fitness.validation_fraction`), so the four cells are one binary
each.

| Arm              | GA     | Random search | GA − RS     | GA − CART | GA leaves | RS leaves |
| ---------------- | ------ | ------------- | ----------- | --------- | --------- | --------- |
| uniform + resub  | 0.8698 | 0.8471        | **+0.0228** | −0.0329   | 9.2       | 9.8       |
| uniform + val    | 0.8580 | 0.8292        | **+0.0288** | −0.0448   | 7.3       | 5.9       |
| midpoint + resub | 0.8730 | 0.8595        | +0.0134     | −0.0298   | 10.9      | 10.4      |
| midpoint + val   | 0.8493 | 0.8412        | +0.0081     | −0.0534   | 11.9      | 11.0      |

Main effects on the GA-minus-random-search gap — the quantity K1 tests:

| Change                     | Effect on GA − RS | Effect on GA | Effect on RS |
| -------------------------- | ----------------- | ------------ | ------------ |
| Validation fitness (1)     | +0.0004           | −0.0177      | −0.0181      |
| Data-driven thresholds (2) | **−0.0150**       | −0.0028      | **+0.0122**  |

**Item 2 helps the baseline, not the GA, and the premise it was built on was wrong.** The
argument for it was that the GA and random search both drew thresholds from
`uniform(feature_min, feature_max)`, so neither could exploit the data and the two were
suppressed equally. That is not what was happening. The GA *could* reach good thresholds —
`threshold_perturbation` and `feature_replacement` move them every generation, and
selection keeps the improvements. Random search draws every candidate independently and
had no such route. Sampling from observed midpoints removed **random search's** handicap.

Item 2 is kept: it is a genuine improvement to the software (each evaluation buys a
candidate from CART's own split set instead of mostly-degenerate splits), and withholding
it from the baseline to protect the gap would be indefensible. It simply makes K1 harder.

**Item 1 is gap-neutral and costs both methods ~1.8 accuracy points.** It is kept on
correctness grounds — resubstitution fitness ranks individuals by how well they memorise
the rows their own leaves were fitted on — and because H2's held-out claim cannot be made
honestly on a search that never saw held-out data. But it buys nothing measurable here.

Caveat that applies to this whole table: three datasets, three folds, differences of
0.003–0.015 against per-dataset standard deviations of 0.02–0.06. The harness itself
refuses to call anything below six datasets significant. These are directional readings
used to choose a configuration, not results.

______________________________________________________________________

### `growth_stop_prob` swept — and it is not a free win either (2026-08-07)

`scripts/sweep_growth_stop.py`. Seed-population shape first, which needs no evolution and
is not noisy:

| `growth_stop_prob` | Stumps in seed population | Median nodes | Mean depth |
| ------------------ | ------------------------- | ------------ | ---------- |
| 0.0                | 0%                        | 55–61        | 6.00       |
| 0.1                | 10–14%                    | 33–41        | 5.1–5.3    |
| 0.2                | 20–22%                    | 22–25        | 4.4–4.6    |
| **0.3 (shipped)**  | **36–48%**                | **5–11**     | 2.9–3.4    |
| 0.5                | 57–66%                    | 3            | 1.5–1.9    |
| 0.7                | 83–88%                    | 1            | 0.5–0.6    |

The concern was right: at the shipped 0.3 nearly half the initial population has no
structure for crossover to recombine, and the median individual is a handful of nodes.

But setting it to 0.0 does not fix the GA. Measured on GA-versus-random-search at fixed
weights, it makes things **worse**:

| `growth_stop_prob` | GA     | Random search | GA − RS     | GA leaves | RS leaves |
| ------------------ | ------ | ------------- | ----------- | --------- | --------- |
| 0.0                | 0.8333 | 0.8512        | **−0.0179** | 4.4       | **16.7**  |
| 0.3                | 0.8305 | 0.8297        | +0.0008     | 4.2       | 4.0       |

Read the leaf columns. Seeded with full-depth trees, random search keeps the big accurate
ones (16.7 leaves) while **the GA prunes itself back down to 4.4** — because that is what
the fitness rewards. `configs/paper.yaml` weights `prune_subtree` at 0.25 against
`expand_leaf` at 0.05, so the operator mix is five-to-one biased toward shrinking, and the
interpretability term pays for it. The GA is not failing to find large trees; it is
finding them, being handed a better fitness for destroying them, and doing so.

**Decision: K1 runs at the shipped 0.3.** Not because 0.3 is right — it clearly is not, on
seed shape alone — but because the only measurement of the quantity K1 tests says 0.0
makes the GA *lose* to random search, and picking a parameter value by looking at the
outcome variable is exactly what the pre-registration exists to prevent. The correct home
for `growth_stop_prob` is the inner-CV tuning grid, selected per fold from data, and that
belongs to a re-run rather than to a hand-set default.

This is the third finding in a row pointing at the same place: **the fitness weighting and
the mutation operator mix, not the search machinery, are what hold this method back.**
That is Phase 3's subject, and it is now the highest-value work in this plan.

Correctness fixes — done on `paper/phase-0` (2026-08-07):

- [x] `engine.py` — `t.fitness_ if t.fitness_ else -inf` treated a fitness of exactly `0.0`
  as unevaluated, ranking it below negative-fitness individuals in elitism and tournaments
  and dropping it from the generation statistics. Replaced by `_fitness_key`, where only
  `None` means unevaluated.

- [x] `engine.py` — the hardcoded `random.random() < 0.3` growth stop is now
  `TreeInitializer(growth_stop_prob=...)`, validated to `[0, 1)` and wired to
  `tree.growth_stop_prob` in every config. **Still to do: sweep it** — it is a free
  parameter that has never been tuned, and it sets how bushy the seed population is.

- [x] `multi_objective.py` — `selTournamentDCD` rejects `k == len(pop)` unless it divides by
  4, and `configs/fast.yaml` ships `population_size: 50`. The mating pool is now padded to a
  multiple of 4 and trimmed back to `n`.

- [x] `multi_objective.py` — `tools.assignCrowdingDist` does not exist in DEAP 1.4 (it lives
  in `deap.tools.emo`), so the LDD-16 line raised `AttributeError` on every real run. The
  unit tests hid this by patching the missing symbol in with `create=True`, and only ever
  used population sizes divisible by 4.

- [x] **Pareto front collapse — diagnosed and fixed** (2026-08-07). The first un-stubbed
  run returned all 50 individuals on front 0 with identical objectives. Cause: **NSGA-II
  has no duplicate elimination.** Crowding distance does not remove clones — identical
  points sit at distance 0 from each other, and once the merged pool is one big front there
  is nothing else to select, so clones fill the population. Instrumented on iris, distinct
  objective vectors fell 26 → 2 within five generations and never recovered.

  `TreeGenotype.structural_signature()` gives a hashable fingerprint of tree shape, splits
  and predictions; `ParetoOptimizer._deduplicate` drops repeats before environmental
  selection (`eliminate_duplicates=True` by default, as in pymoo). The returned front is
  deduplicated unconditionally — the population may carry clones so its size stays fixed,
  but a front of N copies is not a front.

  Measured over 20 generations at `population_size=50`:

  | Dataset | Distinct front points (before → after) | Best accuracy (before → after) |
  | ------- | -------------------------------------- | ------------------------------ |
  | iris    | 2 → 3                                  | 0.9467 → 0.9600                |
  | wine    | 2 → 7                                  | 0.9045 → 0.9270                |

  Wine now traces a monotone accuracy/interpretability trade-off across 7 points, which is
  the shape H1 needs.

**Still open on the Pareto path, before any hypervolume number:**

- [x] **Hypervolume implemented** (2026-08-07) — `src/ga_trees/evaluation/hypervolume.py`.
  Exact 2-D sweep on **(accuracy, node count)**, not the composite score, against the
  pre-registered shared reference point (accuracy 0, max nodes over all methods on that
  dataset + margin). `frontier()` deduplicates objective vectors *before* dominance
  filtering and `Frontier` reports `n_evaluated`, `n_distinct` and `len()` separately, so
  the 27-trees-at-3-points case cannot be reported as a 27-point frontier.
  `cart_pruning_frontier()` builds H1's comparator from the `ccp_alpha` path.
- **Front size is not the number of distinct objective points.** Post-fix iris returns 27
  structurally distinct trees at only 3 objective points; reporting "front size" would
  overstate the result ~9×. Now structurally prevented, see above.
- **The front still thins over time** — iris front-0 distinct points drift 5 → 2 across 15
  generations even with deduplication. Deduplication stops the catastrophic collapse; it
  does not by itself maintain spread. Random immigrants on the top-up path are the obvious
  next lever and were deliberately not added here, since they change search behaviour and
  need their own ablation row.
- Objectives are still `(accuracy, composite interpretability)` on **resubstitution** data.
  Item 6 above (switch to validation accuracy and node count) still stands.

**Exit criterion:** ablation table shows each change's isolated contribution to
hypervolume and held-out accuracy.

______________________________________________________________________

## Phase 3 — Interpretability construct — **COMPLETE** (2026-09-29)

**Decision: option (a) now, option (b) as a follow-up paper.**

- [x] Demote the composite score to a *search heuristic only* (`InterpretabilityCalculator`
  docstring, README, `docs/core-concepts/interpretability.md`)
- [x] Report interpretability using established proxies: #leaves, mean weighted decision-path
  length, #distinct features used — every `scripts/benchmark.py` row carries them, and the
  paper reports nothing else (K4)
- [x] Document that `semantic_coherence` and `feature_coherence` are search-guidance terms
  with no claimed human-interpretability validity

Rationale: `semantic_coherence` (std of feature depths) is invented and ungrounded, yet
`configs/paper.yaml` weights it 0.30. `feature_coherence = 1 − n_used/n_total` scales with
dataset dimensionality, giving high-dimensional datasets a free high score.

Follow-up paper (option b): ground each sub-metric in Lipton 2018, Doshi-Velez & Kim 2017,
Rudin 2019, Lage et al.; run a forward-simulation human study (n≈30).

______________________________________________________________________

## Phase 4 — Positioning (parallel with 1–3)

Three literatures must be engaged or the paper is desk-rejected as reinvention:

1. **Evolutionary tree induction** — Barros et al. 2012, *A Survey of Evolutionary
   Algorithms for Decision-Tree Induction*, IEEE TSMC (~100 papers). Also GALE, EVO-Tree,
   cGA-based approaches. Position against them explicitly.
1. **Optimal sparse trees** — GOSDT (Lin et al. 2020), OSDT, DL8.5, MurTree, OCT
   (Bertsimas & Dunn 2017). Run GOSDT as a baseline where feasible.
   *Our angle:* exact methods optimize a fixed sparsity penalty and scale badly; a GA
   gives an anytime, whole-frontier answer with arbitrary non-decomposable objectives.
1. **Interpretability measurement** — Rudin 2019, Doshi-Velez & Kim 2017.

______________________________________________________________________

## Phase 5 — Write

- [x] JOSS paper drafted (`paper/paper.md`, `paper/paper.bib`) — 2026-08-07, 938 words,
  within JOSS's 250–1000 range, all 16 citations resolving. Sells the *software*: the
  evolutionary tree representation with non-decomposable objectives, and the nested-CV
  harness with budget-matched baselines and pre-registered kill criteria. No performance
  claim appears in it, so it does not depend on how K1 lands.

  **Two things need a human decision before submission:**

  - **ORCID is a placeholder** (`0000-0000-0000-0000`). JOSS requires a real one.
  - **Authorship.** `git shortlog` shows four other contributors — LuF8y / Abd_Alrazak
    Qahwaji (15 commits), shreeshbhat04-ctrl / Shreesha HB (4), yousefdeeb-112004 (1).
    JOSS expects everyone who made a significant software contribution to be listed. The
    draft currently names one author.

- [ ] Research paper draft to GECCO format

- [x] Figures regenerated from committed CSVs only; no hardcoded numbers anywhere —
  enforced by construction, see the Phase 0 entry

______________________________________________________________________

## Kill criteria

Pre-registered in `paper/PREREGISTRATION.md`. Read them before running Phase 3.

______________________________________________________________________

## Timeline

| Weeks | Work                                |
| ----- | ----------------------------------- |
| 1     | Phase 0 + JOSS submission drafted   |
| 2–3   | Phase 1 protocol rebuild            |
| 4–6   | Phase 2 algorithm fixes + ablations |
| 7     | Phase 3 interpretability reframe    |
| 8–9   | Full re-run on ~20 datasets         |
| 10–14 | Phase 5 writing + revision          |

Verify the GECCO call-for-papers deadline before committing to it — dates shift yearly.
