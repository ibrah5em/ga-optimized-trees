# Research Methodology

> **⚠️ Under revision.** The protocol described below is the one the code on `main`
> currently implements, annotated with what is wrong with it. It is being replaced — see
> `paper/PLAN.md`. Do not treat this page as a description of a sound experimental design.

## Current protocol, and its defects

### Datasets

- **Iris**: 150 samples, 4 features, 3 classes
- **Wine**: 178 samples, 13 features, 3 classes
- **Breast Cancer**: 569 samples, 30 features, 2 classes

All three are saturated — accuracy differences between reasonable methods sit inside fold
noise, so the suite cannot separate methods. The replacement suite is roughly 20 datasets
from OpenML CC-18, frozen before any run.

### Cross-Validation

- **20-fold stratified CV**, same folds for the GA and the baselines

The shared folds are real, and paired comparison across them is the right intent. Two
problems: there is no inner loop, so nothing is tuned and model selection happens on the same
folds used for reporting; and the seed does not reach the search. `scripts/experiment.py`
constructs `GAConfig` without `random_state`, so seed 42 reaches only the CV splitter — **GA
runs are not reproducible**.

### Hyperparameters

**GA (`configs/paper.yaml`)**:

```yaml
population_size: 80
n_generations: 40
crossover_prob: 0.72
mutation_prob: 0.18
accuracy_weight: 0.68
interpretability_weight: 0.32
max_depth: 6
```

Nothing here is tuned; the values were fixed by hand. Two keys in the config file are read
and then silently discarded — `classification_metric` never reaches `FitnessCalculator`, and
`early_stopping_rounds` never reaches `GAConfig`. The tree constraints
(`min_samples_split`, `min_samples_leaf`) are enforced only at initialization: neither
crossover nor `expand_leaf` re-checks them against data, so evolved trees can violate the
constraints the config advertises.

Fitness is **resubstitution accuracy**. Leaf predictions are fit on `X` and then scored on
the same `X`, and the champion is selected on training fit. A validation split is supported
by `FitnessCalculator` but never supplied.

**CART Baseline**:

```python
DecisionTreeClassifier(max_depth=6, random_state=42)
```

Untuned, with no `ccp_alpha` and no `min_samples_*` matched to the GA's constraints. This is
the central flaw in the size comparison: it is effectively an unpruned baseline, so any tree
penalised for node count looks small next to it.

### Statistical Tests

- **Paired t-test** across folds, **Cohen's d**, α = 0.05

The t-test across cross-validation folds is invalid — folds share training data and are not
independent (Dietterich 1998). Standard deviations use `np.std` at default `ddof=0`. And a
non-significant result was reported as equivalence, which does not follow.

## Replacement protocol

Specified in `paper/PLAN.md`, pre-registered in `paper/PREREGISTRATION.md`:

| Item             | Value                                                                              |
| ---------------- | ---------------------------------------------------------------------------------- |
| Outer evaluation | 10-fold stratified CV × 3 repeats                                                  |
| Inner tuning     | 5-fold CV, applied identically to every method                                     |
| Baselines        | inner-CV-tuned CART, unconstrained CART, budget-matched random search, RF, XGBoost |
| Fitness          | fit leaves on a GA-train split, score on a held-out GA-validation split            |
| Across-dataset   | Wilcoxon signed-rank; Friedman + Nemenyi with critical-difference diagrams         |
| Equivalence      | TOST against a pre-registered 2% absolute-accuracy margin                          |
| Multiplicity     | Holm correction across datasets, α = 0.05                                          |
| Complexity axis  | node count                                                                         |
| Seeding          | `random_state` into `GAConfig`, deterministic per-fold seeds, `seeds.json`         |
