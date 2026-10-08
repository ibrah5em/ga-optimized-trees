# Interpretability: what is measured, and what is only searched for

This page states the project's position on interpretability. It applies to anything written
about results from this framework.

## Reported measures

Interpretability is reported with established structural proxies only
\[Freitas 2014; Piltaver et al. 2016\]:

| Measure                       | Definition                                                        | Where it comes from                             |
| ----------------------------- | ----------------------------------------------------------------- | ----------------------------------------------- |
| **Node count**                | Internal nodes + leaves                                           | `ga_tree_complexity`, `tree_.node_count` (CART) |
| **Leaf count**                | Number of rules the tree encodes                                  | same                                            |
| **Mean decision-path length** | Depth of the leaf each sample lands in, averaged over the samples | same — weighted by the data, not by leaf count  |
| **Distinct features used**    | Features that appear in at least one split                        | same                                            |

These are properties of the tree that anyone can recompute. None of them is claimed to *be*
interpretability; they are the size- and length-based proxies that the comprehensibility
literature most often uses, and every one is reported for every method, including CART.

The frontier benchmark uses **node count** as its complexity axis, fixed before the benchmark was run.

## The composite score is a search heuristic

`FitnessCalculator` combines four terms into one "interpretability" number used by the
weighted-sum fitness:

| Term                 | What it computes                                          | Why it is not an outcome measure                                                    |
| -------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `node_complexity`    | Decreasing function of node count                         | Redundant with node count, which is reported directly                               |
| `feature_coherence`  | `1 − n_used / n_total`                                    | Scales with dataset dimensionality: a 30-feature dataset gets a high score for free |
| `tree_balance`       | Balance of left/right subtree depths                      | No evidence that balance helps a human read a tree                                  |
| `semantic_coherence` | Standard deviation of the depths at which features appear | Invented for this project; no grounding in the literature and no human study        |

The composite steers the single-objective search toward small trees, and in that role it
works. It has **no claimed validity as a measure of human interpretability**, and it appears
in no reported result. Validating any of these terms would need a forward-simulation user
study \[Doshi-Velez & Kim 2017; Lipton 2018\].

## A consequence worth knowing

The weighting in the composite is the dominant factor in how big a single-objective GA tree
ends up. At `configs/paper.yaml`'s weights a tree must gain about 8 accuracy points to justify
growing from a stump to the size of a tuned CART tree (22.6 points at `configs/fast.yaml`).
So a single GA tree and a single CART tree compared on accuracy alone are usually two points
at opposite ends of the size axis. Compare frontiers, or check the leaf column, before reading
an accuracy gap.
