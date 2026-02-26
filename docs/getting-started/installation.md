# Installation Guide

> ⚠️ **Planned Feature:** The `[api]` extra (FastAPI/Uvicorn) and MLflow experiment tracking listed below are optional dependencies for features not yet implemented in the current release.

This guide covers all installation methods for `ga-optimized-trees`.

## Quick Start

```bash
# Minimal installation (core dependencies only)
pip install -e .

# Or with all features
pip install -e .[all]
```

## Installation Methods

### 1. Development Installation (Recommended)

For development or if you want to modify the code:

```bash
# Clone repository
git clone https://github.com/ibrah5em/ga-optimized-trees.git
cd ga-optimized-trees

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### 2. User Installation (From Source)

```bash
git clone https://github.com/ibrah5em/ga-optimized-trees.git
cd ga-optimized-trees
pip install .
```

### 3. From PyPI (When Published)

```bash
# Not yet available - coming soon!
# pip install ga-optimized-trees
```

## Optional Dependencies

The package has several optional dependency groups:

### Visualization (`viz`)

For tree visualization with Graphviz:

```bash
pip install -e .[viz]

# System dependencies (required for graphviz):
# Ubuntu/Debian:
sudo apt-get install graphviz

# macOS:
brew install graphviz

# Windows: Download from https://graphviz.org/download/
```

### Optimization (`optimization`)

For hyperparameter tuning and experiment tracking:

```bash
pip install -e .[optimization]
```

Includes:

- Optuna (Bayesian optimization)
- MLflow (experiment tracking)

### Baseline Models (`baselines`)

For comparing against ensemble methods:

```bash
pip install -e .[baselines]
```

Includes:

- XGBoost
- LightGBM

### Explainability (`explainability`)

For model interpretation tools:

```bash
pip install -e .[explainability]
```

Includes:

- SHAP
- LIME

### API Support (`api`)

For building web interfaces:

```bash
pip install -e .[api]
```

Includes:

- FastAPI
- Uvicorn
- Pydantic

### All Optional Dependencies

Install everything at once:

```bash
pip install -e .[all]
```

## Development Setup

For contributors:

```bash
# Install with development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Check code quality
black src/ tests/ scripts/
isort src/ tests/ scripts/
flake8 src/ tests/ scripts/
mypy src/
```

## Complete Setup (Everything)

For the full experience with all features:

```bash
pip install -e .[full]
```

This includes:

- Core dependencies
- All optional features (viz, optimization, baselines, explainability, api)
- Development tools (testing, linting, formatting)
- Documentation tools

## Requirements Files

Alternative installation using requirements files:

```bash
# Core only
pip install -r requirements.txt

# With optional dependencies
pip install -r requirements.txt
pip install -r requirements-optional.txt

# Development setup
pip install -r requirements-dev.txt
```

## Verification

Test your installation:

```bash
# Import check
python -c "import ga_trees; print('✓ Installation successful!')"

# Run quick test
python scripts/train.py --dataset iris --generations 5 --population 10

# Run test suite
pytest tests/unit/ -v
```

## Troubleshooting

### Issue: "No module named 'ga_trees'"

**Solution:**

```bash
# Make sure you installed in editable mode
pip install -e .

# Or add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Issue: Graphviz not found

**Solution:**

```bash
# Install system package first, then Python package
# Ubuntu/Debian:
sudo apt-get install graphviz
pip install graphviz

# macOS:
brew install graphviz
pip install graphviz
```

### Issue: NumPy version conflict

**Solution:**

```bash
# Clean install
pip uninstall numpy
pip install "numpy>=1.24.0,<2.0.0"
```

### Issue: DEAP import error

**Solution:**

```bash
pip install deap>=1.4.1
```

## Platform-Specific Notes

### Windows

```bash
# Use Anaconda for easier setup
conda create -n ga-trees python=3.11
conda activate ga-trees
pip install -e .
```

### macOS (Apple Silicon)

```bash
# Use native ARM build
pip install -e .

# If issues with numpy/scipy, use conda:
conda install numpy scipy
pip install -e . --no-deps
```

### Linux

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev build-essential

# Then install package
pip install -e .
```

## Docker Installation

```bash
# Build image
docker build -t ga-trees:latest .

# Run container
docker run --rm ga-trees:latest python scripts/train.py --dataset iris

# With mounted volumes
docker run --rm \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/models:/app/models \
  ga-trees:latest \
  python scripts/train.py --dataset breast_cancer
```

## Upgrading

```bash
# Pull latest changes
git pull origin main

# Reinstall
pip install -e . --upgrade
```

## Uninstalling

```bash
# Remove package
pip uninstall ga-optimized-trees

# Remove virtual environment
deactivate
rm -rf venv/

# Clean build artifacts
rm -rf build/ dist/ *.egg-info/
```

## Next Steps

After installation:

1. **Quick Tutorial**: See `docs/getting-started/quickstart.md`
1. **Run Examples**: Try `python scripts/train.py --dataset iris`
1. **Read Docs**: Check `docs/README.md`
1. **Run Tests**: `pytest tests/ -v`

## Support

If you encounter issues:

1. Check [Troubleshooting Guide](../faq/troubleshooting.md)
1. Search [GitHub Issues](https://github.com/ibrah5em/ga-optimized-trees/issues)
1. Open a new issue with:
   - Python version: `python --version`
   - OS and version
   - Full error traceback
   - Installation method used
