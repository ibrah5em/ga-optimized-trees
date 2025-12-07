# 🌳 GA-Optimized Decision Trees

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Issues](https://img.shields.io/github/issues/ibrah5em/ga-optimized-trees)](https://github.com/ibrah5em/ga-optimized-trees/issues)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-brightgreen)](docs/README.md)

**A genetic algorithm framework for evolving decision trees that balance accuracy and interpretability.**

Unlike greedy algorithms like CART that only optimize for accuracy, our multi-objective approach explores the Pareto front of solutions, allowing you to choose models based on your domain requirements. Achieve **46-82% smaller trees** with **statistically equivalent accuracy** (validated with 20-fold CV, p > 0.05).

---

## 🎯 Key Features

- **🧬 Multi-Objective Optimization**: Balance accuracy and interpretability using weighted-sum fitness
- **🌳 Flexible Tree Representation**: Constrained binary decision trees with validation and repair mechanisms
- **📈 Comprehensive Baselines**: Compare against CART, Random Forest, XGBoost with statistical rigor
- **📊 Statistical Validation**: Automated significance testing with 20-fold cross-validation
- **🔍 Interpretability Metrics**: Composite scoring including tree complexity, balance, and feature coherence
- **⚡ Configuration-Driven**: YAML-based configs for reproducible experiments
- **🎯 Custom Fitness Functions**: Easy extension for domain-specific optimization
- **📚 Extensive Documentation**: 40+ markdown files covering every aspect

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ibrah5em/ga-optimized-trees.git
cd ga-optimized-trees

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install (choose one)
pip install -e .              # Core only
pip install -e .[all]         # All features
pip install -e .[dev]         # Development setup
```

### Quick Demo (1 minute)

```bash
# Train on Iris with optimized config
python scripts/train.py --config configs/custom.yaml --dataset iris
```

**Output:** Trained model saved to `models/best_tree.pkl`

### Full Benchmark (17 minutes)

```bash
# Run on all 3 datasets with statistical tests
python scripts/experiment.py --config configs/custom.yaml
```

**Output:**
- Results CSV with comparison table
- Statistical tests (p-values, Cohen's d)
- Tree size analysis

---

## 📊 Benchmark Results

### Accuracy Comparison (20-fold CV)

| Dataset | GA Accuracy | CART Accuracy | p-value | Conclusion |
|---------|-------------|---------------|---------|------------|
| **Iris** | 94.55 ± 8.07% | 92.41 ± 10.43% | 0.186 | No significant difference ✓ |
| **Wine** | 88.19 ± 10.39% | 87.22 ± 10.70% | 0.683 | No significant difference ✓ |
| **Breast Cancer** | 91.05 ± 5.60% | 91.57 ± 3.92% | 0.640 | No significant difference ✓ |

### Tree Size Comparison

| Dataset | GA Nodes | CART Nodes | Size Reduction | Status |
|---------|----------|------------|----------------|--------|
| **Iris** | 7.4 | 16.4 | **55%** ✓ | Much smaller |
| **Wine** | 10.7 | 20.7 | **48%** ★ | Target achieved! |
| **Breast Cancer** | 6.5 | 35.5 | **82%** ⭐ | Extremely compact |

★ **Wine hits exact target range (46-49%)!**

### Key Findings

✅ **Statistical Equivalence**: All p-values > 0.05 (no significant difference in accuracy)  
✅ **Size Reduction**: 46-82% smaller trees than CART  
✅ **Interpretability Control**: Explicit multi-objective optimization  
✅ **Validation**: 20-fold cross-validation with rigorous statistical testing

---

## 🧬 How It Works

### Evolutionary Process

1. **🎲 Initialization**: Generate random valid trees respecting constraints
2. **📊 Fitness Evaluation**: Parallel evaluation balancing accuracy + interpretability
3. **🏆 Selection**: Tournament selection with elitism
4. **🔀 Crossover**: Subtree-aware swapping with constraint repair
5. **🧬 Mutation**: Threshold perturbation, feature replacement, pruning, expansion
6. **🎯 Multi-Objective**: Weighted-sum fitness for accuracy-interpretability balance

### Fitness Function

```
Fitness = w₁ × Accuracy + w₂ × Interpretability

Interpretability = Σ (wᵢ × ComponentScoreᵢ)
  Components:
    • Node Complexity (50%): e^(-nodes/15)
    • Feature Coherence (10%): 1 - (unique_features / internal_nodes)
    • Tree Balance (10%): 1 - (std_depths / max_depth)
    • Semantic Coherence (30%): 1 - (entropy / max_entropy)

Default Weights: 68% accuracy, 32% interpretability
```

---

## 📖 Documentation

Comprehensive documentation in `docs/`:

### 📚 Getting Started
- [Installation Guide](docs/getting-started/installation.md) - Detailed setup instructions
- [Configuration Guide](docs/getting-started/Configuration.md) - YAML configuration explained

### 🔬 Core Concepts
- [Architecture Overview](docs/core-concepts/architecture.md) - System design and components
- [Genotype API](docs/api-reference/genotype.md) - Tree structure reference

### 📖 User Guides
- [Training Models](docs/user-guides/training.md) - Train custom models
- [Running Experiments](docs/user-guides/experiments.md) - Benchmark experiments
- [Hyperparameter Tuning](docs/user-guides/hyperparameter-tuning.md) - Optuna optimization
- [Visualization](docs/user-guides/visualization.md) - Plot results

### 🎓 Advanced Topics
- [Baseline Comparisons](docs/advanced/baselines.md) - Compare with CART, RF, XGBoost
- [Custom Fitness Functions](docs/advanced/custom-fitness.md) - Domain-specific optimization
- [Custom Operators](docs/advanced/custom-operators.md) - Extend mutation/crossover
- [Statistical Testing](docs/advanced/statistical-tests.md) - Rigorous evaluation

### 🔬 Research
- [Methodology](docs/research/methodology.md) - Experimental design
- [Benchmark Results](docs/research/benchmarks.md) - Complete results
- [Publications](docs/research/publications.md) - Citation information

### ❓ FAQ
- [Frequently Asked Questions](docs/faq/faq.md) - Common questions answered

**📚 [Browse Full Documentation →](docs/README.md)**

---

## 🎯 Usage Examples

### Example 1: Train on Custom Dataset

```python
import numpy as np
from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator

# Load your data
X_train, y_train = load_your_data()

# Setup
n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))
feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                 for i in range(n_features)}

# Configure GA
ga_config = GAConfig(population_size=80, n_generations=40)
initializer = TreeInitializer(n_features=n_features, n_classes=n_classes,
                             max_depth=6, min_samples_split=8, min_samples_leaf=3)
fitness_calc = FitnessCalculator(accuracy_weight=0.68, interpretability_weight=0.32)
mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)

# Train
ga_engine = GAEngine(ga_config, initializer, fitness_calc.calculate_fitness, mutation)
best_tree = ga_engine.evolve(X_train, y_train, verbose=True)

print(f"Final tree: {best_tree.get_num_nodes()} nodes, depth {best_tree.get_depth()}")
```

### Example 2: Custom Fitness for Medical Diagnosis

```python
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor
from sklearn.metrics import recall_score

class MedicalFitness(FitnessCalculator):
    def calculate_fitness(self, tree, X, y):
        self.predictor.fit_leaf_predictions(tree, X, y)
        y_pred = self.predictor.predict(tree, X)
        
        # Prioritize recall (sensitivity) for disease detection
        recall = recall_score(y, y_pred, average='weighted')
        interpretability = self.interp_calc.calculate_composite_score(
            tree, self.interpretability_weights
        )
        
        # 70% recall, 30% interpretability
        return 0.70 * recall + 0.30 * interpretability

# Use in training
medical_fitness = MedicalFitness(accuracy_weight=0.70, interpretability_weight=0.30)
ga_engine = GAEngine(ga_config, initializer, medical_fitness.calculate_fitness, mutation)
```

### Example 3: Pareto Front Analysis

```bash
# Explore accuracy-interpretability trade-offs
python scripts/run_pareto_optimization.py --config configs/custom.yaml --dataset breast_cancer
```

**Output:** Pareto front visualization showing optimal trade-off solutions

---

## ⚙️ Configuration

Customize experiments via YAML config files:

```yaml
# configs/custom.yaml
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

tree:
  max_depth: 6
  min_samples_split: 8
  min_samples_leaf: 3

fitness:
  mode: weighted_sum
  weights:
    accuracy: 0.68
    interpretability: 0.32
  
  interpretability_weights:
    node_complexity: 0.50
    feature_coherence: 0.10
    tree_balance: 0.10
    semantic_coherence: 0.30

experiment:
  datasets: [iris, wine, breast_cancer]
  cv_folds: 20
  random_state: 42
```

**All scripts support `--config` argument:**

```bash
python scripts/train.py --config configs/custom.yaml --dataset breast_cancer
python scripts/experiment.py --config configs/custom.yaml
python scripts/run_pareto_optimization.py --config configs/custom.yaml
```

---

## 📁 Repository Structure

```
ga-optimized-trees/
├── src/ga_trees/           # Core implementation
│   ├── genotype/           # Tree representation (Node, TreeGenotype)
│   ├── ga/                 # GA engine (initialization, selection, crossover, mutation)
│   ├── fitness/            # Multi-objective fitness functions
│   ├── baselines/          # Scikit-learn baseline models
│   ├── data/               # Data loading utilities
│   └── evaluation/         # Metrics, visualization, statistical tests
├── scripts/                # Command-line tools
│   ├── train.py            # Single model training (supports --config!)
│   ├── experiment.py       # Full benchmark suite
│   ├── run_pareto_optimization.py  # Pareto front analysis
│   ├── hyperopt_with_optuna.py     # Hyperparameter tuning
│   └── test_optimized_config.py    # Validate tuning results
├── configs/                # YAML configuration files
│   ├── custom.yaml         # Recommended research config
│   ├── default.yaml        # Default parameters
│   ├── balanced.yaml       # Balanced accuracy/interpretability
│   └── optimized.yaml      # Optuna-tuned parameters
├── tests/                  # Comprehensive test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── docs/                   # 40+ markdown documentation files
│   ├── getting-started/    # Installation, tutorials, configuration
│   ├── core-concepts/      # Architecture, algorithms
│   ├── user-guides/        # Training, experiments, visualization
│   ├── advanced/           # Baselines, custom fitness, operators
│   ├── research/           # Methodology, benchmarks, publications
│   └── faq/                # Frequently asked questions
├── data/                   # Dataset storage
├── models/                 # Trained model storage
└── results/                # Experiment outputs and figures
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v --cov=src/ga_trees

# Run specific tests
pytest tests/unit/test_genotype.py -v
pytest tests/integration/test_end_to_end.py -v

# Code quality checks
black src/ tests/ scripts/
flake8 src/ tests/
mypy src/
```

**Test Coverage**: Unit tests for all core components + integration tests for end-to-end workflows

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development environment setup
- Code style guidelines (Black, isort, flake8)
- Testing requirements
- Pull request process

### Quick Start for Contributors

```bash
# Clone and setup
git clone https://github.com/ibrah5em/ga-optimized-trees.git
cd ga-optimized-trees

# Install with dev dependencies
pip install -e .[dev]

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Format code (automatic with pre-commit)
black src/ tests/ scripts/
isort src/ tests/ scripts/
```

---

## 📊 Example Results Visualization

### Tree Size Reduction

```
Iris:           ████████░░░░░░░░░░░ 55% smaller
Wine:           █████████░░░░░░░░░░ 48% smaller (Target!)
Breast Cancer:  █████████████░░░░░░ 82% smaller
```

### Accuracy Comparison

```
                    GA          CART        RF
Iris:               94.55%      92.41%      95.33%
Wine:               88.19%      87.22%      97.75%
Breast Cancer:      91.05%      91.57%      95.08%
```

**Conclusion**: GA achieves statistically equivalent accuracy to CART (p > 0.05) with drastically smaller trees, making models more interpretable for human understanding.

---

## 🎓 Use Cases

### Healthcare
- **Interpretable diagnosis models** (e.g., 6-node tree for breast cancer screening)
- **Regulatory compliance** (models must be explainable to FDA)
- **Clinical decision support** (doctors need to understand predictions)

### Finance
- **Transparent credit scoring** (fair lending requirements)
- **Fraud detection** (explain why transaction was flagged)
- **Risk assessment** (regulatory compliance with model explainability)

### Legal
- **Explainable classification** (court-admissible evidence)
- **Fair hiring/admission** (avoid black-box bias)
- **Regulatory compliance** (GDPR "right to explanation")

### Research
- **Comparative ML studies** (interpretability vs accuracy)
- **Feature importance analysis** (what drives predictions?)
- **Model compression** (deploy simpler models)

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

```
Copyright (c) 2025 Ibrahem Hasaki and LuF8y

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 🙏 Acknowledgments

- **Built with [DEAP](https://github.com/DEAP/deap)** for evolutionary algorithms
- **Uses [scikit-learn](https://scikit-learn.org/)** for baseline models and metrics
- **Visualization with [Matplotlib](https://matplotlib.org/)** and [Seaborn](https://seaborn.pydata.org/)
- **Special thanks** to Leen Khalil and Yousef Deeb for their support throughout this project

---

## 📞 Support & Community

- 🐛 [Report a Bug](https://github.com/ibrah5em/ga-optimized-trees/issues)
- 💡 [Request a Feature](https://github.com/ibrah5em/ga-optimized-trees/issues)
- 💬 [Join Discussions](https://github.com/ibrah5em/ga-optimized-trees/discussions)
- 📧 [Email Support](mailto:ibrah5em@github.com)
- 📚 [Full Documentation](docs/README.md)

---

## 📈 Project Stats

- **40+ Documentation Files**: Comprehensive guides for every feature
- **130+ Tests**: Thorough unit and integration testing
- **20-Fold CV Validation**: Statistical rigor in benchmarks
- **3 Benchmark Datasets**: Iris, Wine, Breast Cancer
- **Multiple Baselines**: CART, Pruned CART, Random Forest, XGBoost
- **Python 3.8-3.12**: Broad compatibility

---

## 🚀 Quick Links

| Resource | Link |
|----------|------|
| **Documentation** | [docs/README.md](docs/README.md) |
| **Installation** | [docs/getting-started/installation.md](docs/getting-started/installation.md) |
| **Configuration** | [docs/getting-started/Configuration.md](docs/getting-started/Configuration.md) |
| **Training Guide** | [docs/user-guides/training.md](docs/user-guides/training.md) |
| **Experiments** | [docs/user-guides/experiments.md](docs/user-guides/experiments.md) |
| **Custom Fitness** | [docs/advanced/custom-fitness.md](docs/advanced/custom-fitness.md) |
| **FAQ** | [docs/faq/faq.md](docs/faq/faq.md) |
| **Contributing** | [CONTRIBUTING.md](CONTRIBUTING.md) |

---

<div align="center">

**⭐ If this project helps your research, please consider giving it a star! ⭐**

**Research Achievement**: 46-82% smaller trees • Statistically equivalent accuracy (p > 0.05) • 20-fold CV validation

[🚀 Get Started](#-quick-start) • [📚 Documentation](docs/README.md) • [🤝 Contribute](CONTRIBUTING.md)

</div>