# Scripts

Run any script from the project root after installing the package:

```bash
pip install -e .
python scripts/train.py --config configs/default.yaml --dataset iris
```

| Script                       | Purpose                                                                  |
| ---------------------------- | ------------------------------------------------------------------------ |
| `train.py`                   | Train a single GA tree on a dataset                                      |
| `benchmark.py`               | Nested-CV benchmark: one tuned tree per method (GA, random search, CART) |
| `frontier_benchmark.py`      | Accuracy–size frontiers per method, compared by hypervolume              |
| `experiment.py`              | The original quick cross-validated comparison on a few small datasets    |
| `run_pareto_optimization.py` | Approximate the accuracy–size trade-off by sweeping fitness weights      |
| `hyperopt_with_optuna.py`    | Hyperparameter search with Optuna (`pip install -e .[optimization]`)     |
| `sweep_growth_stop.py`       | Measure the effect of `tree.growth_stop_prob` on seeding and accuracy    |
| `visualize_comprehensive.py` | Figures from a `benchmark.py` results CSV                                |
| `readme_figures.py`          | Regenerate the two images in the README                                  |
| `validate_setup.py`          | Check the installation and dependencies                                  |
| `dataset_integration.py`     | Smoke-test dataset loading                                               |

Everything writes under `results/`, which git ignores. See `results/PROVENANCE.md` for
what may be committed from there.
