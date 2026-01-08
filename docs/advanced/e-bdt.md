**E-BDT (Evolved Bayesian Decision Trees)**

Overview
- E-BDT extends the GA-optimized decision tree framework with optional Bayesian/probabilistic nodes and leaves.
- Provides uncertainty-aware predictions, confidence intervals, and calibration metrics while remaining backward compatible.

Key components
- `BayesianNode` (src/ga_trees/genotype/tree_genotype.py): internal node with `threshold_mean`, `threshold_std`, and `threshold_dist_type` (`normal`, `laplace`, `uniform`). Deterministic `threshold` is populated from `threshold_mean` for compatibility.
- Leaf parameters: `leaf_alpha` (Dirichlet concentration vector) and `leaf_samples_count` for Bayesian updating.
- Probabilistic helpers: `sample_threshold()`, `sample_leaf_distribution()`, `get_soft_decision_prob()` for Monte Carlo and analytic routing probabilities.
- `TreeGenotype.mode`: can be `'deterministic'` (default) or `'bayesian'`.
- `BayesianConfig` (src/ga_trees/configs/bayesian_config.py): centralized configuration for priors, sampling and fitness weights.

Configuration
- Example config files are provided in the repository root `configs/`:
  - `configs/bayesian_default.yaml`
  - `configs/bayesian_medical.yaml`

Usage
- To enable Bayesian mode, construct `TreeGenotype(..., mode='bayesian', bayesian_config=...)` where `bayesian_config` is a dict or `BayesianConfig` instance.
- Use `get_soft_decision_prob(node, x_row)` to compute probabilistic routing at inference time. Combine with `sample_leaf_distribution()` across Monte Carlo samples to obtain predictive distributions and confidence intervals.

Testing
- Unit tests covering the new components are provided under `tests/unit/`:
  - `test_bayesian_config.py`
  - `test_probabilistic_helpers.py`
  - `test_tree_genotype_bayesian.py`

Next steps
- Integrate the probabilistic prediction flow into the fitness and evaluation pipelines.
- Add automated calibration computation and populate `TreeGenotype.mean_calibration_error` and `prediction_uncertainty` in evaluation.
