# Installation Guide

Complete installation instructions for GA-Optimized Decision Trees framework.

## Prerequisites

- **Python**: 3.8 or higher (3.10+ recommended)
- **OS**: Linux, macOS, or Windows
- **Memory**: 4GB RAM minimum (8GB+ recommended for large datasets)
- **Storage**: 500MB for dependencies

## Installation Methods

### Method 1: Install from Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/ibrah5em/ga-optimized-trees.git
cd ga-optimized-trees

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

**Verify installation:**

```bash
python -c "import ga_trees; print('✓ Installation successful!')"
```

### Method 2: Docker Installation

```bash
# Build Docker image
docker build -t ga-trees:latest .

# Run training example
docker run --rm ga-trees:latest python scripts/train.py --dataset iris --generations 10

# Run with mounted volumes (to save results)
docker run --rm \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/models:/app/models \
  ga-trees:latest \
  python scripts/train.py --dataset breast_cancer
```

### Method 3: Docker Compose (Multi-Service)

```bash
# Run training service
docker-compose up ga-train

# Run full experiment suite
docker-compose up ga-experiment

# Run Jupyter notebook server
docker-compose --profile notebook up ga-notebook
# Access at: http://localhost:8888
```

## Core Dependencies

The framework requires these core packages (installed automatically):

|Package|Version|Purpose|
|---|---|---|
|`numpy`|≥1.24.0|Numerical computations|
|`pandas`|≥2.0.0|Data manipulation|
|`scikit-learn`|≥1.3.0|ML utilities, baselines|
|`scipy`|≥1.10.0|Statistical functions|
|`deap`|≥1.4.1|Genetic algorithm framework|
|`matplotlib`|≥3.7.0|Visualization|
|`seaborn`|≥0.12.0|Statistical plots|
|`networkx`|≥3.1|Graph structures|
|`mlflow`|≥2.8.0|Experiment tracking|
|`optuna`|≥3.4.0|Hyperparameter optimization|
|`xgboost`|≥2.0.0|Baseline comparisons|
|`pyyaml`|≥6.0|Configuration files|
|`tqdm`|≥4.65.0|Progress bars|

## Optional Dependencies

### For Visualization

```bash
# Tree visualization with Graphviz
pip install graphviz

# System-level Graphviz (required)
# Ubuntu/Debian:
sudo apt-get install graphviz
# macOS:
brew install graphviz
# Windows: Download from https://graphviz.org/download/
```

### For Explainability

```bash
# SHAP for feature importance
pip install shap

# LIME for local explanations
pip install lime
```

### For API Server

```bash
# FastAPI for REST API
pip install fastapi uvicorn pydantic
```

### For Development

```bash
# Testing
pip install pytest pytest-cov

# Code quality
pip install black isort flake8 mypy

# Or install all dev dependencies:
pip install -e ".[dev]"
```

## Verification Tests

### Quick Test

```bash
# Run basic tests
pytest tests/unit/test_basic.py -v
```

### Full Test Suite

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src/ga_trees --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Integration Test

```bash
# Train a model on Iris dataset (should complete in ~30 seconds)
python scripts/train.py --dataset iris --generations 10 --population 30

# Expected output:
# ✓ Model saved to: models/best_tree.pkl
# Test Accuracy: 0.93+
```

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-venv build-essential

# For Graphviz
sudo apt-get install graphviz graphviz-dev
```

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python graphviz
```

### Windows

```bash
# Use Anaconda for easier setup
conda create -n ga-trees python=3.10
conda activate ga-trees
conda install numpy pandas scikit-learn matplotlib seaborn

# Then install remaining packages
pip install -r requirements.txt
pip install -e .
```

## Common Installation Issues

### Issue 1: `contourpy` version conflict

**Error:** `ERROR: No matching distribution for contourpy<1.3.3`

**Solution:**

```bash
# Pin contourpy version
pip install "contourpy>=1.0.0,<1.3.3"
```

### Issue 2: `deap` import error

**Error:** `ModuleNotFoundError: No module named 'deap'`

**Solution:**

```bash
pip install deap>=1.4.1
```

### Issue 3: Graphviz not found

**Error:** `ExecutableNotFound: failed to execute ['dot', '-Tsvg']`

**Solution:** Install system-level Graphviz (see Optional Dependencies above)

### Issue 4: NumPy version conflict

**Error:** `numpy.ndarray size changed`

**Solution:**

```bash
pip install --upgrade numpy scipy scikit-learn --force-reinstall
```

## Environment Configuration

### MLflow Tracking (Optional)

```bash
# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
```

### CUDA Support (For Future GPU Acceleration)

```bash
# Currently CPU-only, but prepared for GPU support
# Install CUDA-enabled XGBoost if needed:
pip install xgboost[gpu]
```

## Updating the Framework

```bash
# Pull latest changes
git pull origin main

# Reinstall dependencies
pip install -r requirements.txt --upgrade
pip install -e . --upgrade
```

## Uninstallation

```bash
# Remove virtual environment
deactivate
rm -rf venv/

# Or if using pip:
pip uninstall ga-optimized-trees

# Clean build artifacts
rm -rf build/ dist/ *.egg-info/
```

## Next Steps

- **Quick Start**: See Quickstart Guide for a 5-minute introduction
- **Tutorial**: Follow the Step-by-Step Tutorial
- **Configuration**: Learn about Configuration Files
- **Examples**: Check Examples for domain-specific use cases

## Troubleshooting

If you encounter issues not covered here, please:

1. Check Troubleshooting Guide
2. Search [GitHub Issues](https://github.com/ibrah5em/ga-optimized-trees/issues)
3. Open a new issue with:
    - Python version (`python --version`)
    - OS and version
    - Full error traceback
    - Steps to reproduce