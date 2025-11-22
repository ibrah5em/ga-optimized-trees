
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://ibrah5em.github.io/ga-optimized-trees/)

# 🌳 GA-Optimized Decision Trees

A production-ready framework for evolving interpretable decision trees using genetic algorithms with multi-objective optimization. Balance model performance with human-understandable explanations through evolutionary computation.

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

# Run a quick demo (Iris dataset, ~5 minutes)
python scripts/train.py --dataset iris --generations 20 --population 50

# Run full experiment suite (~30 minutes on 8 cores)
python scripts/experiment.py --config configs/small_experiment.yaml
```

## 📊 Key Features

- **🧬 Multi-Objective Optimization**: Balance accuracy and interpretability using NSGA-II
- **🌳 Flexible Genotype**: Constrained tree structures with validation and repair
- **📈 Rich Baselines**: Compare against CART, Random Forest, XGBoost, and more
- **📊 Statistical Rigor**: Automated significance testing and effect size calculation
- **🔍 Experiment Tracking**: Integrated MLflow for reproducibility
- **⚡ Parallel Execution**: Multiprocessing for fitness evaluation
- **🎯 Interpretability Metrics**: Composite scoring including tree complexity, balance, and feature coherence
- **🌐 REST API**: FastAPI endpoint for model serving and deployment
- **🐳 Docker Support**: Reproducible containerized execution

## 🏗️ Architecture Overview

```mermaid
graph TB
    A[Data Loader] --> B[GA Engine]
    B --> C[Fitness Evaluation]
    C --> D[Multi-Objective Optimization]
    D --> E[Pareto Front]
    E --> F[Model Selection]
    F --> G[Interpretable Tree]
    
    H[Baseline Models] --> I[Comparison]
    G --> I
    I --> J[Statistical Analysis]
```

## 🧪 Running Experiments

### Basic Training
```bash
# Train on a single dataset
python scripts/train.py \
    --dataset wine \
    --generations 50 \
    --population 100 \
    --output models/wine_model.pkl

# Evaluate trained model
python scripts/evaluate.py \
    --model models/wine_model.pkl \
    --dataset wine \
    --test-size 0.2
```

### Multi-Objective Optimization
```bash
# Evolve Pareto front
python scripts/train.py \
    --dataset breast_cancer \
    --mode pareto \
    --generations 100 \
    --population 200 \
    --objectives accuracy interpretability
```

### Full Experiment Suite
```bash
# Run all experiments with baselines
python scripts/experiment.py \
    --config configs/full_experiment.yaml \
    --n-jobs 8 \
    --output results/
```

### Hyperparameter Optimization
```bash
# Auto-tune GA hyperparameters
python scripts/hyperopt.py \
    --dataset credit_default \
    --n-trials 50 \
    --output configs/optimized.yaml
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

![Pareto Front](https://raw.githubusercontent.com/ibrah5em/ga-optimized-trees/main/results/figures/pareto_front_breast_cancer.png)

### Baseline Comparison (Breast Cancer Dataset)

| Model | Accuracy | F1 Score | Tree Size | Depth | Interpretability |
|-------|----------|----------|-----------|-------|------------------|
| **GA-Optimized** | **0.953 ± 0.012** | **0.951 ± 0.013** | **15.2 ± 2.1** | **4.8 ± 0.5** | **0.87 ± 0.04** |
| CART | 0.932 ± 0.018 | 0.928 ± 0.019 | 28.4 ± 5.3 | 7.2 ± 1.2 | 0.62 ± 0.07 |
| Pruned CART | 0.941 ± 0.015 | 0.937 ± 0.016 | 19.6 ± 3.2 | 5.5 ± 0.8 | 0.75 ± 0.05 |
| Random Forest | 0.968 ± 0.010 | 0.967 ± 0.011 | N/A | N/A | 0.45 ± 0.08 |

*Results with 5-fold cross-validation. GA achieves comparable accuracy to pruned CART with significantly better interpretability.*

## 🐳 Docker Usage

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

### Interpretability Metric
```
I = w1 * (1 - TreeComplexity) + w2 * FeatureCoherence + 
    w3 * TreeBalance + w4 * SemanticCoherence
```

## 📦 Installation 

### From Source (Development)

```bash
git clone https://github.com/ibrah5em/ga-optimized-trees.git
cd ga-optimized-trees
pip install -e ".[dev,api,docs]"
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

## 📈 Experiment Tracking

Results are automatically logged to MLflow for comprehensive tracking:

```bash
# Start MLflow UI
mlflow ui --backend-store-uri results/mlruns

# View at http://localhost:5000
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

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@software{ibrahim_ga_trees_2025,
  title = {GA-Optimized Decision Trees: A Framework for Interpretable Machine Learning},
  author = {Ibrahim, Mohamed},
  year = {2025},
  url = {https://github.com/ibrah5em/ga-optimized-trees},
  version = {1.0.0}
}
```

## 🙏 Acknowledgments

- Built with [DEAP](https://github.com/DEAP/deap) for evolutionary algorithms
- Uses [scikit-learn](https://scikit-learn.org/) for baseline models and metrics
- Experiment tracking with [MLflow](https://mlflow.org/)
- Visualization with [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)

## 📞 Support & Community

- 🐛 [Report a Bug](https://github.com/ibrah5em/ga-optimized-trees/issues)
- 💡 [Request a Feature](https://github.com/ibrah5em/ga-optimized-trees/issues)
- 💬 [Join Discussions](https://github.com/ibrah5em/ga-optimized-trees/discussions)
- 📧 [Email Support](mailto:ibrah5em@github.com)

---

<div align="center">
  
**Made with ❤️ by [Ibrahem Hasaki](https://github.com/ibrah5em)**

*If this project helps your research, please consider giving it a ⭐*

</div>
