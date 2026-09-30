---
title: "ga-optimized-trees: evolutionary induction of accuracy--complexity frontiers for decision trees"
tags:
  - Python
  - machine learning
  - decision trees
  - genetic algorithms
  - multi-objective optimization
  - interpretability
authors:
  - name: Ibrahem Hasaki
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Antioch (Antakya) Syrian Private University, Syria
    index: 1
date: 7 August 2026
bibliography: paper.bib
---

# Summary

Decision trees are induced almost universally by greedy recursive partitioning: at each
node, CART [@breiman1984] picks the split that maximises a local impurity reduction, and
tree size is controlled afterwards by pruning. This produces one tree per configuration,
optimises a criterion that must decompose over nodes, and gives the user no direct control
over where the resulting model sits on the trade-off between predictive accuracy and
structural simplicity.

`ga-optimized-trees` induces decision trees by evolutionary search instead. A population of
whole trees is evolved under selection, subtree crossover and four structural mutation
operators, so the objective being optimised is a property of the *entire* tree rather than
of one split. Two search modes are provided: a scalarised single-objective mode, and a
multi-objective mode built on NSGA-II [@deb2002] via DEAP [@fortin2012], which returns a
Pareto front of non-dominated trees from a single run. Because candidates are scored as
complete models, the framework accepts objectives that greedy induction structurally
cannot express — leaf count, mean weighted decision-path length, the number of distinct
features a model consults, or any user-supplied function of a fitted tree.

The package ships with a second component that is unusual for a method library: a nested
cross-validation benchmark harness (`ga_trees.benchmark`), run against a pre-registered
selection from the OpenML CC-18 suite [@bischl2021], in which every method — the GA, a
budget-matched random search over the same tree space, cost-complexity-pruned CART, an
unpruned CART, and a random forest — implements one interface and receives identical
inner-CV tuning, identical outer folds and independently derived per-fold seeds. Search
budgets are *measured* rather than assumed, and the statistics layer implements the
Demšar [@demsar2006] protocol for comparisons across datasets, including Wilcoxon
signed-rank tests with Holm correction, Friedman with Nemenyi critical differences, and
TOST equivalence testing. Comparisons with too few datasets to reach significance are
reported as underpowered by construction rather than by reviewer vigilance.

# Statement of need

Evolutionary decision-tree induction is a mature research area — Barros et al.
[@barros2012] survey roughly a hundred methods — but almost none of that work is available
as maintained, installable software. Researchers wanting to build on it typically
reimplement a genetic tree inducer from a paper description, which makes results across the
literature difficult to compare and slow to reproduce. `scikit-learn` [@pedregosa2011]
provides only greedy induction; `pymoo` [@blank2020] provides excellent general
multi-objective optimisation but no tree representation, genetic operators, or evaluation
protocol; exact optimal-tree solvers such as GOSDT [@lin2020] and optimal classification
trees [@bertsimas2017] optimise a fixed sparsity penalty, must be re-solved per penalty
value, and scale poorly in the number of features. This package targets the gap between
them: an anytime search that returns a whole frontier per run and admits objectives the
exact and greedy formulations cannot state.

The intended users are researchers in evolutionary machine learning and in interpretable
modelling, and practitioners in domains where an accuracy--complexity operating point is a
decision to be made deliberately rather than a by-product of a pruning parameter. In
clinical, credit and regulatory settings, model complexity is frequently a hard requirement
rather than a preference [@rudin2019], and the ability to inspect the whole attainable
frontier before committing to a model is a practical need rather than a methodological
nicety.

The benchmark harness addresses a second, more uncomfortable need. Comparisons of tree
inducers are unusually easy to get wrong: selecting hyperparameters on the folds used for
reporting inflates results [@varma2006; @cawley2010]; comparing a tuned proposal against an
untuned baseline is common; per-fold paired t-tests treat non-independent folds as
independent; and a small tree and a large tree evaluated only on accuracy are being
compared at different points of a trade-off rather than against each other. This package
was itself the subject of an internal audit in which previously published figures could not
be reproduced from the code that supposedly generated them. The harness, the
pre-registration document (`paper/PREREGISTRATION.md`) with binding kill criteria, and the
figure generators — which read committed run output and refuse to draw anything when no run
exists — are the response to that audit, and are offered as a reusable pattern rather than
as project-specific bookkeeping.

# Design

The core representation is a recursive `Node`/`TreeGenotype` structure carrying depth and
minimum-sample constraints. Prediction and leaf fitting use iterative, index-partitioned
traversal rather than per-sample recursion, which keeps evaluation cost proportional to the
number of nodes and avoids recursion limits on deep trees.

Split thresholds are drawn from the midpoints between consecutive distinct observed values
of the samples reaching each node — the same candidate set CART enumerates — rather than
uniformly across a feature's range, and are filtered so that every candidate satisfies the
minimum-samples-per-leaf constraint by construction. Fitness may be evaluated on a
stratified holdout carved from the fitting data, so that leaf predictions are fitted and
scored on disjoint rows; both behaviours are configurable so their contributions can be
ablated rather than asserted.

Multi-objective search performs explicit duplicate elimination on a structural signature of
each tree before environmental selection. NSGA-II's crowding distance does not remove
identical individuals — clones sit at zero distance from one another — and without this the
returned front collapses to repeated copies of a single point within a few generations.

# Availability

`ga-optimized-trees` is available on GitHub at
<https://github.com/ibrah5em/ga-optimized-trees> under the MIT licence, is installable via
`pip install -e .`, supports Python 3.8--3.12, and is tested on Linux, macOS and Windows in
continuous integration. It builds on NumPy [@harris2020], SciPy [@virtanen2020],
scikit-learn [@pedregosa2011], DEAP [@fortin2012] and Matplotlib [@hunter2007].
Documentation, worked notebooks, and the pre-registered benchmark protocol are included in
the repository.

# Acknowledgements

We thank the contributors to the repository for code, review and testing.

# References
