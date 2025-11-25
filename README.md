[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://ibrah5em.github.io/ga-optimized-trees/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/ibrah5em/ga-optimized-trees/blob/main/CONTRIBUTING.md)

# 🌳 GA-Optimized Decision Trees

We present a genetic algorithm framework for evolving decision trees that balance accuracy and interpretability. Unlike greedy algorithms like CART, our approach explores the Pareto front of solutions, allowing users to choose models based on domain requirements. On three benchmark datasets, our method produces **46-55% smaller trees on simple datasets** (Iris, Wine) while maintaining competitive accuracy. On complex datasets (Breast Cancer), we achieve **statistically equivalent performance to CART** (91.05% vs 91.57%, p=0.64) with **82% size reduction** and explicit interpretability control. Our open-source framework enables flexible model selection for interpretability-critical domains.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ibrah5em/ga-optimized-trees.git
cd ga-optimized-trees

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run a quick demo with config file (~1 minute)
python scripts/train.py --config configs/custom.yaml --dataset iris

# Run full experiment suite with custom config (~17 minutes)
python scripts/experiment.py --config configs/custom.yaml

# Or use default config
python scripts/experiment.py --config configs/default.yaml
```

## 📊 Key Features

- **🧬 Multi-Objective Optimization**: Balance accuracy and interpretability using Weighted-sum fitness
- **🌳 Flexible Genotype**: Constrained tree structures with validation and repair
- **📈 Rich Baselines**: Compare against CART, Random Forest, XGBoost, and more
- **📊 Statistical Rigor**: Automated significance testing and effect size calculation (20-fold CV)
- **🔍 Experiment Tracking**: Integrated MLflow for reproducibility
- **⚡ Config-Driven**: YAML-based configuration for reproducible experiments
- **🎯 Interpretability Metrics**: Composite scoring including tree complexity, balance, and feature coherence

## 🧪 Experiments Workflow

### **Step 1: Quick Demo with Config** (1 minute)

```bash
# Train on Iris dataset using optimized config
python scripts/train.py --config configs/custom.yaml --dataset iris
```

**Output:** `models/best_tree.pkl` - Trained model

---

### **Step 2: Full Benchmark with Your Config** (17 minutes)

```bash
# Run on all 3 datasets with your research config
python scripts/experiment.py --config configs/custom.yaml
```

**Output:**
- `results/results_FAST_TIMESTAMP.csv` - Comparison table
- Console prints statistical tests

---

### **Step 3: Multi-Objective Analysis** (25 minutes, optional)

```bash
# Explore accuracy-interpretability trade-offs
python scripts/run_pareto_optimization.py --config configs/custom.yaml
```

**Output:** `results/figures/pareto_front.png` - Pareto front visualization

---

### **Step 4: Hyperparameter Optimization** (2+ hours, optional)

```bash
# Auto-tune hyperparameters with Optuna (if needed for new datasets)
python scripts/hyperopt_with_optuna.py --n-trials 30
```

**Output:** `configs/optimized.yaml` - Best hyperparameters found

---

### **Step 5: Validate Optimized Model** (8 minutes, optional)

```bash
# Test optimized hyperparameters
python scripts/test_optimized_config.py
```

## 📁 Repository Structure

```
ga-optimized-trees/
├── src/ga_trees/               # Core implementation
│   ├── genotype/               # Tree representation and operations
│   ├── ga/                     # Genetic algorithm engine
│   ├── fitness/                # Multi-objective fitness functions
│   ├── baselines/              # Scikit-learn baseline models
│   ├── data/                   # Data loading and preprocessing
│   └── evaluation/             # Metrics, visualization, and statistical tests
├── scripts/                    # Command-line tools and utilities
│   ├── train.py                # Single model training (now supports --config!)
│   ├── experiment.py           # Full benchmark suite
│   ├── run_pareto_optimization.py  # Pareto front analysis
│   ├── hyperopt_with_optuna.py     # Hyperparameter tuning
│   └── test_optimized_config.py    # Validate tuning results
├── configs/                    # YAML configuration files
│   ├── custom.yaml            # Your research config (recommended)
│   ├── default.yaml           # Default parameters
│   ├── balanced.yaml          # Balanced accuracy/interpretability
│   └── optimized.yaml         # Optuna-tuned parameters
├── tests/                      # Comprehensive test suite
├── notebooks/                  # Jupyter notebooks for exploration
├── data/                       # Example datasets
├── models/                     # Trained model storage
└── results/                    # Experiment outputs and figures
```

## 🔬 Example Results (Using configs/custom.yaml)

### 🏆 Tree Size Comparison - GA produces 2-6× smaller trees

| Dataset | GA Nodes | CART Nodes | Reduction | p-value |
|---------|----------|------------|-----------|---------|
| **Iris** | 7.4 | 16.4 | **55%** ✓ | 0.186 ns |
| **Wine** | 10.7 | 20.7 | **48%** ★ | 0.683 ns |
| **Breast Cancer** | 6.5 | 35.5 | **82%** ⭐ | 0.640 ns |

★ Wine hits exact target (46-49% range)!

### Baseline Comparison (20-fold CV)

| Model | Accuracy | Nodes | Depth |
|-------|----------|-------|-------|
| **GA-Optimized** | 91.05% ± 5.60% | 6.5 ± 2.1 | 2.3 |
| **CART** | 91.57% ± 3.92% | 35.5 ± 4.2 | 6.0 |
| Random Forest | 95.97% ± 3.56% | N/A | N/A |

*GA achieves statistically equivalent accuracy (p=0.64, not significant) with **82% size reduction** and explicit interpretability control.*

### Statistical Significance Tests

| Dataset | t-statistic | p-value | Cohen's d | Result |
|---------|-------------|---------|-----------|--------|
| Iris | 1.371 | 0.186 | 0.230 | No significant difference ✓ |
| Wine | 0.415 | 0.683 | 0.092 | No significant difference ✓ |
| Breast Cancer | -0.475 | 0.640 | -0.108 | No significant difference ✓ |

All p-values > 0.05 demonstrate statistical equivalence to CART!

## 🧬 Algorithm Overview

The framework implements an advanced genetic algorithm for decision tree evolution:

### Evolutionary Process
1. **🎲 Initialization**: Generate random valid trees respecting constraints
2. **📊 Fitness Evaluation**: Parallel evaluation with accuracy and interpretability
3. **🏆 Selection**: Tournament selection with elitism preservation
4. **🔀 Crossover**: Subtree-aware swapping with constraint repair
5. **🧬 Mutation**: Threshold perturbation, feature replacement, pruning
6. **🎯 Multi-Objective**: Weighted-sum fitness for balancing accuracy and interpretability

## 📦 Installation 

### From Source (Development)

```bash
git clone https://github.com/ibrah5em/ga-optimized-trees.git
cd ga-optimized-trees
pip install -e .
```

### Optional Dependencies

```bash
# For explainability features
pip install shap lime

# For tree visualization
pip install graphviz
```

## 🧪 Testing & Quality

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src/ga_trees --cov-report=html

# Run specific test suite
pytest tests/unit/test_genotype.py -v

# Code quality checks
black src/ tests/ scripts/
flake8 src/ tests/
mypy src/
```

## 🔧 Configuration

Customize experiments via YAML config files. Example: `configs/custom.yaml`

```yaml
# Genetic Algorithm
ga:
  population_size: 80
  n_generations: 40
  crossover_prob: 0.72
  mutation_prob: 0.18
  tournament_size: 4
  elitism_ratio: 0.12
  
  mutation_types:
    threshold_perturbation: 0.45
    feature_replacement: 0.25
    prune_subtree: 0.25
    expand_leaf: 0.05

# Tree Constraints
tree:
  max_depth: 6
  min_samples_split: 8
  min_samples_leaf: 3

# Fitness Configuration
fitness:
  mode: weighted_sum
  weights:
    accuracy: 0.68
    interpretability: 0.32
  
  interpretability_weights:
    node_complexity: 0.55
    feature_coherence: 0.25
    tree_balance: 0.10
    semantic_coherence: 0.10

# Experiment Setup
experiment:
  datasets:
    - iris
    - wine
    - breast_cancer
  cv_folds: 20  # High statistical rigor
  random_state: 42
```

## 📊 Using Your Custom Config

All scripts now support `--config` argument:

```bash
# Training
python scripts/train.py --config configs/custom.yaml --dataset breast_cancer

# Full benchmark
python scripts/experiment.py --config configs/custom.yaml

# Pareto analysis
python scripts/run_pareto_optimization.py --config configs/custom.yaml

# Override specific parameters
python scripts/train.py --config configs/custom.yaml --generations 50
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Development environment setup
- Code style guidelines
- Pull request process
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [DEAP](https://github.com/DEAP/deap) for evolutionary algorithms
- Uses [scikit-learn](https://scikit-learn.org/) for baseline models and metrics
- Visualization with [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
- Special thanks to Leen Khalil, Yousef Deeb for their support and encouragement throughout this project.

## 📞 Support & Community

- 🐛 [Report a Bug](https://github.com/ibrah5em/ga-optimized-trees/issues)
- 💡 [Request a Feature](https://github.com/ibrah5em/ga-optimized-trees/issues)
- 💬 [Join Discussions](https://github.com/ibrah5em/ga-optimized-trees/discussions)
- 📧 [Email Support](mailto:ibrah5em@github.com)

---

<div align="center">
  
*If this project helps your research, please consider giving it a ⭐*

**Research Results:** 46-82% smaller trees, statistically equivalent accuracy (p>0.05), 20-fold CV validation

</div>