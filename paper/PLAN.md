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

## Phase 0 — Stop the bleeding (2–3 days)

- [ ] `git tag pre-paper-audit` on current `main`
- [ ] Complete `paper/CLAIMS.md` audit (started — see file)
- [ ] Strip unsupported numbers from `README.md`, `docs/research/benchmarks.md`,
  `docs/research/results.md`, `docs/research/methodology.md`; replace with
  "results under revision"
- [ ] Delete or regenerate `results/tables/paper-results.csv` (produced by unversioned code)
- [ ] Remove the "Target: 24-77% smaller trees" header comment from `configs/paper.yaml`

**Exit criterion:** no claim is public that the code on `main` cannot reproduce.

______________________________________________________________________

## Phase 1 — Rebuild the protocol (~2 weeks)

Rewrite `scripts/experiment.py` into a real benchmark harness. This is where the paper is
won or lost.

- [ ] **Nested CV.** Outer 10-fold × 3 repeats for reporting; inner 5-fold for *all*
  hyperparameter selection — GA weights and rates, CART `ccp_alpha` + `max_depth`,
  RF, XGBoost. Every method gets the same treatment, no exceptions.
- [ ] **Budget-matched baselines:**
  - CART with cost-complexity pruning tuned by inner CV
  - **Random search over the same tree space with the same evaluation count**
    (`population_size × n_generations`)
  - CART unconstrained (re-implement properly — the archived rows came from lost code)
- [ ] **Seed everything.** `random_state` into `GAConfig` (currently dropped at
  `scripts/experiment.py:197`), deterministic per-fold seeds, `seeds.json` artifact.
- [ ] **Statistics done properly:**
  - Wilcoxon signed-rank across *datasets*; Friedman + Nemenyi with critical-difference
    diagrams (Demšar 2006)
  - Equivalence via **TOST** with a pre-registered 2% absolute-accuracy margin, or the
    Bayesian correlated t-test (Benavoli et al. 2017) with a ROPE
  - Retire `ttest_rel` across CV folds (`scripts/experiment.py:456`) — folds share
    training data, violates independence (Dietterich 1998)
  - `np.std(..., ddof=1)` throughout
- [ ] **Scale to ~20 datasets** from OpenML CC-18. iris/wine/breast_cancer are saturated.
- [ ] **Wire the config properly.** `classification_metric` and `early_stopping_rounds`
  are read from YAML and silently dropped (`experiment.py:197,224`) — `paper.yaml`
  currently misrepresents what ran.

**Exit criterion:** one command reproduces every number; every number is committed
alongside the config and seed that produced it.

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

Correctness fixes to fold in:

- `engine.py:179,187,441,449` — `t.fitness_ if t.fitness_ else -inf` treats a fitness of
  exactly `0.0` as `-inf`
- `engine.py:121` — hardcoded `random.random() < 0.3` growth stop; document and sweep it
- `multi_objective.py:120` — `selTournamentDCD` asserts `len(pop) % 4 == 0`; pop=50 breaks

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
