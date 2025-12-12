# GA-Optimized Decision Trees Documentation

Welcome to the comprehensive documentation for the GA-Optimized Decision Trees framework. This documentation provides detailed guides for researchers, developers, and practitioners.

## 📚 Documentation Structure

### Getting Started
- [Quick Start Guide](quickstart.md) - Get up and running in 5 minutes
- [Installation Guide](getting-started/installation.md) - Detailed installation instructions
- [Basic Tutorial](getting-started/tutorial.md) - Step-by-step beginner tutorial
- [Configuration Guide](getting-started/configuration.md) - Understanding YAML configs
- [Dataset loading](data/dataset-loader.md) - preprocessing


### Core Concepts
- [Architecture Overview](core-concepts/architecture.md) - System design and components
- [Genetic Algorithm](core-concepts/genetic-algorithm.md) - How the GA works
- [Tree Representation](core-concepts/tree-representation.md) - Genotype structure
- [Fitness Functions](core-concepts/fitness-functions.md) - Multi-objective optimization
- [Interpretability Metrics](core-concepts/interpretability.md) - How we measure interpretability

### API Reference
- [Genotype Module](api-reference/genotype.md) - Tree structure classes
- [GA Engine](api-reference/ga-engine.md) - Evolution engine
- [Fitness Calculator](api-reference/fitness.md) - Fitness evaluation
- [Operators](api-reference/operators.md) - Selection, crossover, mutation
- [Evaluation Tools](api-reference/evaluation.md) - Metrics and visualization

### User Guides
- [Training Models](user-guides/training.md) - How to train custom models
- [Running Experiments](user-guides/experiments.md) - Benchmark experiments
- [Hyperparameter Tuning](user-guides/hyperparameter-tuning.md) - Optimize with Optuna
- [Pareto Optimization](user-guides/pareto-optimization.md) - Multi-objective exploration
- [Visualization](user-guides/visualization.md) - Plotting results
- [Model Export](user-guides/model-export.md) - Save and load models

### Advanced Topics
- [Multi-Objective Optimization](advanced/multi-objective.md) - NSGA-II implementation
- [Custom Fitness Functions](advanced/custom-fitness.md) - Extend the framework
- [Custom Operators](advanced/custom-operators.md) - Add new genetic operators
- [Baseline Comparisons](advanced/baselines.md) - Compare with CART, RF, XGBoost
- [Statistical Testing](advanced/statistical-tests.md) - Rigorous evaluation

### Research
- [Methodology](research/methodology.md) - Research approach
- [Results](research/results.md) - Experimental results
- [Benchmarks](research/benchmarks.md) - Dataset performance
- [Publications](research/publications.md) - Academic papers

### Development
- [Contributing Guide](development/contributing.md) - How to contribute
- [Code Style](development/code-style.md) - Coding standards
- [Testing](development/testing.md) - Test suite overview
- [CI/CD Pipeline](development/ci-cd.md) - Continuous integration

### Examples
- [Iris Classification](examples/iris.md) - Simple example
- [Medical Diagnosis](examples/medical.md) - Healthcare application
- [Credit Scoring](examples/credit.md) - Financial application
- [Custom Dataset](examples/custom-dataset.md) - Using your own data

### FAQ & Troubleshooting
- [Frequently Asked Questions](faq/faq.md)
- [Troubleshooting](faq/troubleshooting.md)
- [Performance Tips](faq/performance.md)

## 🎯 Quick Navigation

### For Researchers
1. [Architecture Overview](core-concepts/architecture.md)
2. [Methodology](research/methodology.md)
3. [Results](research/results.md)

### For Practitioners
1. [Quick Start](quickstart.md)
2. [Training Models](user-guides/training.md)
3. [Examples](examples/)

### For Developers
1. [Contributing Guide](development/contributing.md)
2. [API Reference](api-reference/)
3. [Testing](development/testing.md)

## 📖 Documentation Conventions

- **Code blocks** use syntax highlighting
- **Commands** start with `$` or `python`
- **File paths** use `monospace` formatting
- **Important notes** are highlighted in callouts
- **Examples** include expected output

## 🔗 External Resources

- [GitHub Repository](https://github.com/ibrah5em/ga-optimized-trees)
- [Issue Tracker](https://github.com/ibrah5em/ga-optimized-trees/issues)
- [Discussions](https://github.com/ibrah5em/ga-optimized-trees/discussions)

## 📝 Documentation Updates

This documentation is continuously updated. Last updated: November 2025 based on:

```
docs/
├── README.md                           # Main documentation hub ✓
├── getting-started/
│   ├── quickstart.md                   # Quick start (5 min) ✓
│   ├── installation.md                 # Detailed installation
│   ├── tutorial.md                     # Step-by-step tutorial
│   └── configuration.md                # Config guide ✓
├── core-concepts/
│   ├── architecture.md                 # System design ✓
│   ├── genetic-algorithm.md            # GA details
│   ├── tree-representation.md          # Genotype structure
│   ├── fitness-functions.md            # Fitness calculation
│   └── interpretability.md             # Interpretability metrics
├── api-reference/
│   ├── genotype.md                     # Tree API ✓
│   ├── ga-engine.md                    # GA Engine API
│   ├── fitness.md                      # Fitness API
│   ├── operators.md                    # Genetic operators API
│   └── evaluation.md                   # Evaluation tools API
├── user-guides/
│   ├── training.md                     # Training models
│   ├── experiments.md                  # Running experiments
│   ├── hyperparameter-tuning.md        # Optuna tuning
│   ├── pareto-optimization.md          # Multi-objective
│   ├── visualization.md                # Plotting
│   └── model-export.md                 # Save/load models
├── advanced/
│   ├── multi-objective.md              # NSGA-II
│   ├── custom-fitness.md               # Custom fitness
│   ├── custom-operators.md             # Custom operators
│   ├── baselines.md                    # Baseline comparisons
│   └── statistical-tests.md            # Statistical methods
├── research/
│   ├── methodology.md                  # Research approach
│   ├── results.md                      # Experimental results
│   ├── benchmarks.md                   # Dataset performance
│   └── publications.md                 # Academic papers
├── development/
│   ├── contributing.md                 # Contributing guide
│   ├── code-style.md                   # Style guide
│   ├── testing.md                      # Testing
│   └── ci-cd.md                        # CI/CD pipeline
├── examples/
│   ├── iris.md                         # Iris example
│   ├── medical.md                      # Healthcare app
│   ├── credit.md                       # Financial app
│   └── custom-dataset.md               # Custom data
└── faq/
    ├── faq.md                          # FAQ ✓
    ├── troubleshooting.md              # Troubleshooting
    └── performance.md                  # Performance tips
```

For corrections or improvements, please open an issue or submit a pull request.
