# Frontier truncation diagnostic — 2026-09-29 (exploratory)

`scripts/frontier_diagnostic.py --datasets vowel,eucalyptus,vehicle,tic_tac_toe`, first 10
outer folds of the committed split, committed per-fold GA seeds. **Added after K2 was known.
It decides nothing and does not reopen K2.**

One change per arm, everything else as `configs/paper.yaml`:

| Arm              | Change                                                               |
| ---------------- | -------------------------------------------------------------------- |
| `base`           | none — control; reproduces the committed GA fronts exactly           |
| `resubstitution` | `fitness.validation_fraction: 0`                                     |
| `grow-bias`      | `growth_stop_prob: 0`, `expand_leaf` and `prune_subtree` weights swapped |
| `2x-budget`      | population and generations doubled                                   |

`folds.csv` has normalised hypervolume, largest delivered tree and best accuracy per
(dataset, method, fold), scored against one reference per dataset over all arms and the
committed methods. `points.csv` has the new arms' test fronts.

Share of the GA-to-CART normalised-hypervolume gap closed, mean over the four datasets:
resubstitution 29%, 2× budget 28%, grow-bias −4%. Largest delivered trees stay at 15–40
nodes in every arm; CART's reach 78–162.

A first run of this script loaded the evidence copy of the config, whose sorted
`mutation_types` does not reproduce the committed run (see `../frontier-2026-08-07/README.md`);
its control arm did not match, it was discarded, and this is the re-run from
`configs/paper.yaml`.
