# Scripts

Run any script from the project root after installing the package:

```bash
pip install -e .
python scripts/train.py --config configs/default.yaml --dataset iris
python scripts/experiment.py --config configs/default.yaml
```

## Available Scripts

| Script                     | Purpose                                                    |
| -------------------------- | ---------------------------------------------------------- |
| train.py                   | Train a single GA-optimized tree                           |
| experiment.py              | Run cross-validated benchmark experiments                  |
| hyperopt_with_optuna.py    | Hyperparameter optimization with Optuna                    |
| run_pareto_optimization.py | Multi-objective Pareto optimization                        |
| validate_setup.py          | Verify installation and dependencies                       |
| visualize_comprehensive.py | Generate all visualization and publication-quality figures |
| dataset_integration.py     | Test dataset loading                                       |

## Visualization Output

`visualize_comprehensive.py` generates two sets of figures under `results/figures/`:

**Comprehensive figures** (5-fold CV, `configs/default.yaml`):

- `accuracy_comparison.png`
- `tree_size_comparison.png`
- `tradeoff_scatter.png`
- `speed_comparison.png`
- `summary_table.png`
- `key_findings.png`

**Publication figures** (20-fold CV, `configs/paper.yaml` — saved as PNG + PDF):

- `paper_fig1_size_reduction`
- `paper_fig2_statistical_equiv`
- `paper_fig3_pareto_tradeoff`
- `paper_table_summary.png`

To reproduce the paper figures, run `experiment.py` with `configs/paper.yaml` first.
