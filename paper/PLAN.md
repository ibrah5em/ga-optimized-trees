# Publication Plan — ga-optimized-trees

**Started:** 2026-08-04
**Targets:** JOSS (software, in parallel) + GECCO or Applied Soft Computing / SWEVO (research)
**Claim:** frontier dominance, *not* accuracy parity with CART

______________________________________________________________________

## Why this plan exists

Two mutually inconsistent result sets live in this repo, and the public-facing claim is
the older, smaller, unreproducible one. See `CLAIMS.md` for the audit. The plan below
rebuilds the experimental protocol, fixes the algorithm, and replaces the claim with one
the architecture can actually support.

**Retired claim:** "46–82% smaller trees with statistically equivalent accuracy."

**Target claim:**

> A multi-objective evolutionary search traces the accuracy–complexity frontier for
> decision trees in a single run, dominating (by hypervolume) the frontier obtainable
> from CART's cost-complexity pruning path, and admitting non-decomposable objectives
> that greedy and exact methods cannot express.

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

One item deliberately left open, tracked in `results/PROVENANCE.md`:
`scripts/visualize_comprehensive.py` still holds the hardcoded `RESULTS` and
`PAPER_RESULTS` dicts. Running it regenerates figures asserting the withdrawn claims.
Deleting the output while leaving the generator in place fixes nothing.

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

| #   | Change                                                                                                                                                | Where                                                                |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 1   | **Validation-based fitness** — split outer-train into GA-train/GA-val; fit leaves on GA-train, score on GA-val; select champion on validation fitness | `fitness/calculator.py` (`X_val` already supported), `engine.py:449` |
| 2   | **Data-driven split points** — thresholds from observed midpoints of samples reaching the node, not `uniform(feature_min, feature_max)`               | `engine.py:136,317-323,368-372`                                      |
| 3   | **Greedy seeding** — initialize ~20% of the population with CART trees on bootstrap samples at varying depths                                         | `engine.py` `TreeInitializer`                                        |
| 4   | **Constraint repair** — re-check `min_samples_leaf`/`min_samples_split` against data after crossover and mutation (currently enforced only at init)   | `engine.py:357-381`, `improved_crossover.py`                         |
| 5   | **Memetic local search** — cheap threshold hill-climb on the elite fraction each generation                                                           | `engine.py` `evolve`                                                 |
| 6   | **Fix Pareto objectives** to (validation accuracy, −node_count); report hypervolume + attainment surfaces vs CART's `ccp_alpha` path                  | `ga/multi_objective.py:186`                                          |

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

- **Front size is not the number of distinct objective points.** Post-fix iris returns 27
  structurally distinct trees at only 3 objective points. Hypervolume must be computed on
  distinct objective vectors; reporting "front size" would overstate the result ~9×.
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

## Phase 3 — Interpretability construct (~1 week)

**Decision: option (a) now, option (b) as a follow-up paper.**

- [ ] Demote the composite score to a *search heuristic only*
- [ ] Report interpretability using established proxies: #leaves, mean weighted decision-path
  length, #distinct features used
- [ ] Document that `semantic_coherence` and `feature_coherence` are search-guidance terms
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

- [ ] JOSS paper (`paper/paper.md`, `paper/paper.bib`) — **start immediately**, does not
  depend on empirical results
- [ ] Research paper draft to GECCO format
- [ ] Figures regenerated from committed CSVs only; no hardcoded numbers anywhere

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
