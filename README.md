
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://ibrah5em.github.io/ga-optimized-trees/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/ibrah5em/ga-optimized-trees/blob/main/CONTRIBUTING.md)
[![GitHub Issues](https://img.shields.io/github/issues/ibrah5em/ga-optimized-trees)](https://github.com/ibrah5em/ga-optimized-trees/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/ibrah5em/ga-optimized-trees)](https://github.com/ibrah5em/ga-optimized-trees/pulls)

# 🌳 GA-Optimized Decision Trees

We present a genetic algorithm framework for evolving decision trees that balance accuracy and interpretability. Unlike greedy algorithms like CART, our approach explores the Pareto front of solutions, allowing users to choose models based on domain requirements. On three benchmark datasets, our method produces 46-49% smaller trees on simple datasets (Iris, Wine) while maintaining competitive accuracy. On complex datasets (Breast Cancer), we achieve statistically equivalent performance to CART (91.92% vs 92.80%, p=0.55) with explicit interpretability control. Automated hyperparameter tuning with Optuna improves performance by 1.94%. Our open-source framework enables flexible model selection for interpretability-critical domains.

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

# Run a quick demo (Iris dataset, ~1 minutes)
python scripts/train.py --dataset iris --generations 20 --population 50

# Run full experiment suite (Iris dataset + wine dataset + breast_cancer dataset, ~2 minutes)
python scripts/experiment.py --config configs/default.yaml

# You Can run the same script using other config file, like optimized.yaml
python scripts/experiment.py --config configs/optimized.yaml
```

## 📊 Key Features

- **🧬 Multi-Objective Optimization**: Balance accuracy and interpretability using NSGA-II
- **🌳 Flexible Genotype**: Constrained tree structures with validation and repair
- **📈 Rich Baselines**: Compare against CART, Random Forest, XGBoost, and more
- **📊 Statistical Rigor**: Automated significance testing and effect size calculation
- **🔍 Experiment Tracking**: Integrated MLflow for reproducibility
- **⚡ Parallel Execution**: Multiprocessing for fitness evaluation
- **🎯 Interpretability Metrics**: Composite scoring including tree complexity, balance, and feature coherence
- **🐳 Docker Support**: Reproducible containerized execution

## 🧪 Experiments Workflow

### **Step 1: Quick Demo** (1 minute)

```bash
# Train on Iris dataset
python scripts/train.py --dataset iris --generations 20 --population 50
```

**Output:** `models/best_tree.pkl` - Trained model

---

### **Step 2: Full Benchmark** (2-5 minutes)

```bash
# Run on all 3 datasets with default parameters
python scripts/experiment.py --config configs/default.yaml
```

**Output:**

- `results/results_FAST_TIMESTAMP.csv` - Comparison table
- Console prints statistical tests

---

### **Step 3: Multi-Objective Analysis** (15 minutes)

```bash
# Explore accuracy-interpretability trade-offs
python scripts/run_pareto_optimization.py
```

**Output:** `results/figures/pareto_front.png` - Pareto front visualization

---

### **Step 4: Hyperparameter Optimization** (20 minutes, 30 trials)

```bash
# Auto-tune hyperparameters with Optuna
python scripts/hyperopt_with_optuna.py
```

**Output:** `configs/optimized.yaml` - Best hyperparameters found

---

### **Step 5: Validate Optimized Model** (8 minutes)

```bash
# Test optimized hyperparameters
python scripts/test_optimized_config.py
```

**Output:** `results/optimized_comparison.json` with statistics

**Example result:**

```json
{
  "optimized_ga": {
    "accuracy_mean": 0.9192,
    "nodes_mean": 36.6
  },
  "cart": {
    "accuracy_mean": 0.9280,
    "nodes_mean": 27.4
  },
  "statistics": {
    "p_value": 0.5457,  # Not significant!
    "cohens_d": -0.276   # Small effect
  }
}
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
│   ├── evaluation/             # Metrics, visualization, and statistical tests
│   └── tracking/               # MLflow experiment tracking
├── scripts/                    # Command-line tools and utilities
├── tests/                      # Comprehensive test suite
├── notebooks/                  # Jupyter notebooks for exploration
├── configs/                    # YAML configuration files
├── data/                       # Example datasets
├── models/                     # Trained model storage
└── results/                    # Experiment outputs and figures
```

## 🔬 Example Results

### Accuracy vs Interpretability Trade-off

<img width="2930" height="2351" alt="tradeoff_scatter" src="https://github.com/user-attachments/assets/09e4cf0a-f5e4-46a4-a1aa-9bedbf50763b" />

### 🏆 Tree Size Comparison - GA produces 2-7× smaller trees

<img width="3547" height="1745" alt="tree_size_comparison" src="https://github.com/user-attachments/assets/83c90fd0-f64e-4dca-833c-99c617ee04d8" />

### Baseline Comparison (Breast Cancer Dataset)

| Model | Accuracy | Nodes | Depth |
|-------|----------|-------|-------|
| **GA-Optimized** | 91.92% ± 3.85% | 28.6 ± 9.2 | 4.8 |
| **CART** | 92.80% ± 2.30% | 27.4 ± 3.4 | 5.0 |
| Random Forest | 95.08% ± 1.18% | N/A | N/A |

*GA achieves competitive accuracy (p=0.55, not significant) with explicit interpretability control.*


## 🐳 Docker Usage

> **⚠️ Note:** The Docker setup is currently under development and these commands may not work as expected. We are working to stabilize the containerized environment.

```bash
# Build image
docker build -t ga-trees:latest .

# Run experiment
docker run -v $(pwd)/results:/app/results ga-trees:latest \
    python scripts/experiment.py --config configs/default.yaml

# Start API server
docker run -p 8000:8000 ga-trees:latest \
    uvicorn src.ga_trees.api.main:app --host 0.0.0.0

# Run with GPU support
docker run --gpus all -v $(pwd)/results:/app/results ga-trees:latest \
    python scripts/train.py --dataset large_dataset --generations 200
```

## 🧬 Algorithm Overview

The framework implements an advanced genetic algorithm for decision tree evolution:

### Evolutionary Process
1. **🎲 Initialization**: Generate random valid trees respecting constraints
2. **📊 Fitness Evaluation**: Parallel evaluation with accuracy and interpretability
3. **🏆 Selection**: Tournament selection with elitism preservation
4. **🔀 Crossover**: Subtree-aware swapping with constraint repair
5. **🧬 Mutation**: Threshold perturbation, feature replacement, pruning
6. **🎯 Multi-Objective**: NSGA-II for Pareto-optimal solutions


## 📦 Installation 

### From Source (Development)

```bash
git clone https://github.com/ibrah5em/ga-optimized-trees.git
cd ga-optimized-trees
pip install -e 
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

Customize experiments via `configs/default.yaml`:

```yaml
ga:
  population_size: 100
  n_generations: 50
  crossover_prob: 0.7
  mutation_prob: 0.2
  tournament_size: 3
  elitism_ratio: 0.1

tree_constraints:
  max_depth: 5
  min_samples_split: 10
  min_samples_leaf: 5

fitness:
  mode: weighted_sum  # or 'pareto'
  weights:
    accuracy: 0.7
    interpretability: 0.3
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

</div>
