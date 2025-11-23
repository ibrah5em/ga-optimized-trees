# Contributing to GA-Optimized Decision Trees

Thank you for your interest in contributing to GA-Optimized Decision Trees! This document provides guidelines and instructions for contributing to this project.

## 🚀 Quick Start

### Development Environment Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/ga-optimized-trees.git
   cd ga-optimized-trees
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   pip install -r requirements.txt
   ```

3. **Verify setup**
   ```bash
   pytest tests/unit/test_basic.py -v
   ```

## 📋 Contribution Workflow

### 1. Issue First
- Check existing [issues](https://github.com/ibrah5em/ga-optimized-trees/issues) before creating new ones
- For bugs: include steps to reproduce, expected vs actual behavior
- For features: describe use case and proposed implementation
- Use appropriate labels (bug, enhancement, documentation, etc.)

### 2. Branch Naming
Use descriptive branch names:
```
feature/add-new-mutation-operator
bugfix/fix-crossover-error  
docs/update-contributing-guide
```

### 3. Development Process

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
pytest tests/ -v --cov=src/ga_trees

# Format code
black src/ tests/ scripts/
flake8 src/ tests/

# Commit changes
git add .
git commit -m "feat: add new mutation operator for tree pruning"
```

### 4. Pull Request Process
1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Ensure all tests pass**
4. **Update README.md** if introducing new features
5. **Create PR** with clear description and references to issues

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v

# Run with coverage
pytest tests/ -v --cov=src/ga_trees --cov-report=html

# Run specific test file
pytest tests/unit/test_genotype.py -v
```

### Writing Tests
- Follow AAA pattern (Arrange-Act-Assert)
- Use descriptive test names
- Include both positive and negative test cases

Example:
```python
def test_tree_crossover_creates_valid_offspring():
    # Arrange
    parent1 = create_sample_tree()
    parent2 = create_sample_tree()
    
    # Act
    child1, child2 = crossover(parent1, parent2)
    
    # Assert
    assert child1.is_valid()
    assert child2.is_valid()
    assert child1.depth <= MAX_DEPTH
```

## 📝 Code Style

### Python Style Guide
We follow [PEP 8](https://pep8.org/) with these specific rules:

**Imports** (grouped and sorted):
```python
# Standard library
import os
import sys
from typing import List, Dict

# Third-party
import numpy as np
import pandas as pd

# Local
from src.ga_trees.genotype.tree import TreeGenotype
```

**Naming Conventions**:
- Classes: `CamelCase` (`DecisionTreeGenotype`)
- Functions/Methods: `snake_case` (`calculate_fitness`)
- Variables: `snake_case` (`population_size`)
- Constants: `UPPER_SNAKE_CASE` (`MAX_DEPTH`)

### Documentation
- Use Google-style docstrings for public functions/classes
- Include type hints for all function parameters and returns

Example:
```python
def evaluate_tree(tree: TreeGenotype, 
                  X: np.ndarray, 
                  y: np.ndarray) -> float:
    """Evaluate tree performance on dataset.
    
    Args:
        tree: Tree genotype to evaluate
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        
    Returns:
        Accuracy score between 0 and 1
        
    Raises:
        ValueError: If tree is invalid or data shapes don't match
    """
    # Implementation...
```

## 🏗️ Project Structure

### Where to Put Your Code
```
src/ga_trees/
├── genotype/           # Tree representations
│   ├── base.py        # Base tree class
│   ├── operators/     # Crossover, mutation
│   └── constraints/   # Tree validation
├── fitness/           # Objective functions
│   ├── base.py        # Base fitness class
│   ├── multi_obj.py   # Multi-objective fitness
│   └── metrics/       # Interpretability metrics
├── ga/                # Genetic algorithm core
│   ├── engine.py      # Main GA engine
│   ├── selection.py   # Selection operators
│   └── population.py  # Population management
└── evaluation/        # Model evaluation
    ├── metrics.py     # Performance metrics
    ├── statistical.py # Statistical tests
    └── visualization/ # Plotting utilities
```

### Adding New Features

**1. New Genetic Operator**:
- Place in `src/ga_trees/genotype/operators/`
- Inherit from base operator class
- Add to operator registry in `__init__.py`

**2. New Fitness Metric**:
- Place in `src/ga_trees/fitness/metrics/`
- Implement `calculate()` method
- Register in fitness calculator

**3. New Baseline Model**:
- Place in `src/ga_trees/baselines/`
- Implement standard scikit-learn interface
- Add to comparison scripts

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment**:
   ```bash
   python -c "import sys; print(sys.version)"
   pip list | grep -E "(ga-trees|numpy|scikit-learn)"
   ```

2. **Steps to Reproduce**
3. **Expected vs Actual Behavior**
4. **Error Logs** (if any)

## 💡 Feature Requests

For feature requests, please describe:

1. **Use Case**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Alternatives Considered**: Other approaches you've considered
4. **Additional Context**: Any other relevant information

## 📚 Documentation

### Updating Documentation
- Code changes may require docstring updates
- Update README.md for user-facing changes
- Add examples in `notebooks/` for new features
- Update docstrings when modifying functions

### Building Documentation
```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html
```

## 🔧 Development Tools

### Pre-commit Hooks
We use pre-commit to ensure code quality:

```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run on all files
pre-commit run --all-files
```

### Code Quality Checks
```bash
# Format code
black src/ tests/ scripts/

# Linting
flake8 src/ tests/

# Type checking
mypy src/

# Sort imports
isort src/ tests/ scripts/
```

## 🎯 Focus Areas for Contributions

### High Priority
- Performance optimizations
- Additional genetic operators
- New interpretability metrics
- Enhanced visualization tools
- Additional dataset support

### Medium Priority
- Extended baseline comparisons
- Advanced hyperparameter optimization
- Additional statistical tests
- Documentation improvements

### Experimental
- Novel tree representations
- Alternative multi-objective algorithms
- Hybrid approaches with other ML techniques

## 🤝 Community

### Discussion Channels
- [GitHub Discussions](https://github.com/ibrah5em/ga-optimized-trees/discussions) for questions and ideas
- [GitHub Issues](https://github.com/ibrah5em/ga-optimized-trees/issues) for bugs and feature requests

### Code Review Process
1. **Initial Review**: Within 48 hours for new PRs
2. **Feedback Cycle**: Typically 1-2 rounds of review
3. **Merge Criteria**: All tests pass + 1 approved review

## 📄 License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

## 🙏 Acknowledgments

Special thanks to all our contributors! Your efforts make this project better for everyone.

