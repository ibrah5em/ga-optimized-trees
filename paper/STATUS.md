# Status — 2026-08-07

Where the paper effort stands after Phase 2 items 1–2 and the pre-registered frontier run.
Read `PREREGISTRATION.md` for the binding decisions and `PLAN.md` for the full history.

______________________________________________________________________

## The headline

**K1 cleared. K2 fired.**

| Kill criterion                         | Outcome                                          |
| -------------------------------------- | ------------------------------------------------ |
| **K1** — random search matches the GA  | **Not triggered.** GA ahead, p_holm = 0.032      |
| **K2** — GA fails to dominate CART     | **TRIGGERED.** 45% dominance, threshold is 60%   |
| **K3** — accuracy loss vs CART > 2%    | Not measured — run stopped, see below            |
| **K4** — composite score as an outcome | Held. It appears nowhere in any reported measure |

Evolution does contribute something over random sampling of the same space at an equal
budget. It does not beat CART's pruning path often enough to make the frontier claim.

**The claim this project set out to prove is dead.** What remains:

> Over the same tree space and an exactly matched evaluation budget, evolutionary search
> recovers a better accuracy–complexity frontier than random sampling (+0.66 hypervolume,
> p = 0.032 Holm-corrected, 15/20 datasets, d_z = 0.50), and the representation admits
> non-decomposable objectives that greedy induction cannot express.

That is H3 — a mechanism result. It is real and it is publishable in the
evolutionary-tree-induction literature, but it is not "we beat CART," and the paper must
not be written as though it were.

______________________________________________________________________

## What was built

| Area                     | Where                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------ |
| Data-driven split points | `ga/split_points.py`, wired into `TreeInitializer` and all three threshold mutations |
| Validation-based fitness | `benchmark/methods.py:holdout_split`, `GAEngine.evolve(X_val=, y_val=)`              |
| Hypervolume              | `evaluation/hypervolume.py` — exact 2-D sweep on (accuracy, node count)              |
| Frontier benchmark       | `benchmark/frontiers.py`, `scripts/frontier_benchmark.py` — answers K1/K2            |
| Figures                  | `evaluation/figures.py`, `scripts/visualize_comprehensive.py` — raises without data  |
| `growth_stop_prob` sweep | `scripts/sweep_growth_stop.py`                                                       |
| JOSS paper               | `paper/paper.md`, `paper/paper.bib` — 938 words, 16 citations                        |

Both Phase 2 changes are config-gated (`tree.split_strategy`, `fitness.validation_fraction`)
so the ablation isolates them rather than asserting them.

______________________________________________________________________

## Three findings that matter more than the two changes

**1. The premise behind data-driven split points was wrong.** It was adopted to remove a
handicap suppressing the GA and random search equally. Measured, it raises *random
search's* accuracy by +0.0122 and moves the GA's by −0.0028, shrinking the K1 gap by
−0.0150. The GA could already reach good thresholds through `threshold_perturbation` over
generations; random search draws every candidate independently and could not. Fixing the
initializer removed the **baseline's** handicap. Kept anyway — it makes K1 harder, not
easier, and withholding it from the baseline would be indefensible.

**2. The fitness weighting is the real handicap, and it is arithmetic, not search.** At
`configs/fast.yaml`'s weights a tree must gain **22.6 accuracy points** to justify growing
from a stump to CART's ~47 nodes; at `configs/paper.yaml` it is still 8.1. The GA obliges,
converging to 2.7 leaves against tuned CART's 24, then "losing" on accuracy by 8–10 points.
**Any GA-vs-CART accuracy comparison at a fixed weighting is comparing two points at
opposite ends of the complexity axis.** Check the `leaves` column before reading any gap.

**3. `growth_stop_prob = 0.3` seeds 36–48% of the population as stumps** — confirmed by
measurement, no evolution needed. But setting it to 0.0 makes the GA *lose* to random
search (−0.018): seeded with full-depth trees, random search keeps the big accurate ones
(16.7 leaves) while the GA prunes back to 4.4, because `paper.yaml` weights `prune_subtree`
at 0.25 against `expand_leaf` at 0.05 and the interpretability term pays for it. Left at
0.3 pending inner-CV tuning; picking it by looking at the outcome variable is what the
pre-registration exists to prevent.

All three point at the same place: **the fitness weighting and the mutation operator mix,
not the search machinery.** That makes Phase 3 the highest-value work left.

______________________________________________________________________

## Two harness bugs, both found in my own code, both recorded

A kill criterion firing is the moment a project is most tempted to go bug-hunting and least
trustworthy when it succeeds. Both rejected result sets are committed under
`paper/evidence/frontier-2026-08-07/` so the audit trail points at files, not descriptions.

1. **Selection on the test fold**, favouring random search. The first run reported K1
   TRIGGERED (−1.585, p_holm \< 0.0001, 19/20) because `RandomSearchFrontier` returned all
   ~2,545 sampled trees and let dominance filtering run on their *test* scores — a maximum
   over thousands of test evaluations that nothing can deliver. The tell was
   `n_candidates`: 2,545 against the GA's 10.
1. **Reference point per fold instead of per dataset**, favouring the GA. Found by reading
   the implementation against the "Fixed in advance" table, not prompted by the result. It
   moved K2 from 70% to 45% — across the threshold.

______________________________________________________________________

## Open

- **K3 / H2 not measured.** `scripts/benchmark.py --config configs/paper.yaml` was stopped
  after 3h13m with 2 of 20 datasets done; it was pacing at ~9–11h. It only determines
  whether "equivalent accuracy" may appear in the paper, and the ablation already suggests
  it fails — the GA sits 3–5 points behind tuned CART, well outside the 2% margin. Restart
  with `--outer-repeats 1` (under 2h, needs a Deviations entry) if the number is wanted.
- **JOSS submission needs two human decisions:** the ORCID is a placeholder, and
  `git shortlog` shows four other contributors who may warrant authorship.
- **Phase 3** — demote the composite interpretability score to a search heuristic and
  report established proxies. Now the highest-value work, per the three findings above.
- **Phase 2 items 3, 4, 5** — greedy seeding, constraint repair, memetic local search —
  remain unimplemented.
- **The target claim in `PLAN.md` and the README framing need rewriting** around H3.
