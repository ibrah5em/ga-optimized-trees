# Status — 2026-09-29: plan closed

Every phase of `PLAN.md` is complete or deliberately closed, every kill criterion has a
verdict, and the paper is drafted. Read `PREREGISTRATION.md` for the binding decisions and
their outcomes, `PLAN.md` for the history, and `paper/gecco/` for the paper.

______________________________________________________________________

## The headline

| Kill criterion                          | Outcome                                                            |
| --------------------------------------- | ------------------------------------------------------------------ |
| **K1** — random search matches the GA   | **Not triggered.** GA ahead by +0.66 hypervolume, p_holm = 0.032, 15/20 datasets |
| **K2** — GA fails to dominate CART      | **Triggered.** 45% dominance over CART's pruning path; threshold 60% |
| **K3** — accuracy loss vs CART > 2%     | **Triggered.** 8/20 datasets lose > 2 points (threshold 6/20); H2 rejected |
| **K4** — composite score as an outcome  | **Held.** It appears in no reported measure                       |

**What may be claimed:**

> Over the same tree space and an exactly matched evaluation budget, evolutionary search
> recovers a better accuracy–complexity frontier than random sampling (H3). It does not
> beat CART's cost-complexity pruning path (H1 rejected), and its tuned single tree is not
> equivalent in accuracy to tuned CART (H2 rejected).

**What may not:** "smaller trees at equivalent accuracy", "dominates CART", or any
interpretability claim based on the composite score.

______________________________________________________________________

## Why the GA loses where it loses

The losses in K2 and K3 fall on the same eight datasets (vowel, eucalyptus, tic-tac-toe,
vehicle, qsar-biodeg, credit-g, climate-crashes, banknote): problems where accuracy keeps
rising with tree size. The GA's delivered fronts stop at a median of 5.4 nodes against
CART's 32, and across datasets its hypervolume deficit tracks the accuracy only larger
trees reach (Spearman 0.85).

An exploratory ablation on the four worst datasets (control arm reproduces the committed
fronts exactly): dropping the validation split closes 29% of the gap, doubling the budget
28%, and removing the small-tree bias in initialisation and mutation **nothing** (−4%).
Delivered trees stay at 15–40 nodes in every arm against CART's 78–162. Most of the
truncation is unexplained by these three factors.

______________________________________________________________________

## Robustness and exploratory checks

All added after K1/K2 were known; none can change a verdict, and none did.

| Check                                       | Result                                                              |
| ------------------------------------------- | ------------------------------------------------------------------- |
| Constraint repair on (Phase 2 item 4)       | K1 +0.85 (p_holm 0.034), K2 45% — unchanged                         |
| CART path capped at the GA's depth          | GA dominance 40% — the depth asymmetry does not explain K2          |
| GOSDT regularisation path (19/20 datasets)  | Beats CART's path on 7/19; GA vs GOSDT not significant (p_holm 0.98) |
| Reproduction of the committed frontier run  | Bit-identical on the datasets re-run, all 20 datasets' data verified |

______________________________________________________________________

## Problems found and fixed in this round

1. **pandas 3 silently re-encoded two datasets** (`dresses_sales`, `credit_approval`):
   missing categorical values stopped mapping to one token. Fixed; all 20 datasets now
   reproduce the committed CART frontiers exactly (`scripts/verify_dataset_identity.py`).
1. **Saved run configs did not reproduce their runs.** `yaml.dump` sorted
   `mutation_types`, and the operator is drawn by position. Configs are now written in
   order; the evidence READMEs say to reproduce from `configs/`.
1. **Constraints were enforced only at initialisation** — 8–27% of evaluated GA trees
   broke them. `ga_trees.ga.repair`, off by default so the committed run still reproduces.
1. **`benchmark.py` wrote nothing until the end**, so a container restart lost 4 hours of
   K3. It now checkpoints each dataset.

______________________________________________________________________

## Before submission — human decisions

- **Venue and format.** `paper/gecco/main.tex` is ACM sigconf, double-blind. Check the
  current GECCO call (deadline, page limit, template version). It needs a TeX build with
  `acmart` (Overleaf); none was available in the environment that drafted it.
- **Authors.** `git shortlog` shows four other contributors (see `PLAN.md`, Phase 5).
- **ORCID.** The JOSS draft (`paper/paper.md`) still has a placeholder.
- **Anonymised repository link** for review.
- **References without DOIs** in `paper/gecco/refs.bib` were not machine-checked.
