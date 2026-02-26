# Migration Guide: Enhanced Package Configuration

> ⚠️ **Planned Feature:** The `[api]` extra (FastAPI/Uvicorn) and MLflow integration listed in this guide are not yet implemented in the current release.

## What Changed?

The project has been updated to use modern Python packaging standards with proper dependency management.

## Key Changes

### 1. **pyproject.toml** - Modern Standard

The main configuration now uses `pyproject.toml` (PEP 518/621 compliant):

- ✅ All dependencies properly declared
- ✅ Optional dependencies as extras
- ✅ Tool configurations consolidated
- ✅ Better dependency resolution

### 2. **requirements.txt** - Core Only

`requirements.txt` now contains **only core dependencies**:

```txt
numpy>=1.24.0,<2.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0
deap>=1.4.1
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
tqdm>=4.65.0
```

### 3. **New Files**

- `requirements-optional.txt` - Optional features (graphviz, optuna, mlflow, etc.)
- `requirements-dev.txt` - Development tools (pytest, black, mypy, etc.)
- `.pre-commit-config.yaml` - Git hooks for code quality
- `MANIFEST.in` - Package distribution configuration
- `INSTALLATION.md` - Comprehensive installation guide

### 4. **Enhanced CI/CD**

Updated `.github/workflows/ci.yml` with:

- Multi-platform testing (Ubuntu, Windows, macOS)
- Python 3.8-3.12 support
- Code quality checks
- Package building and validation

## How to Migrate

### For Users (Installing the Package)

**Before:**

```bash
pip install -r requirements.txt
pip install -e .
```

**Now (Recommended):**

```bash
# Core only
pip install -e .

# With all features
pip install -e .[all]

# With specific features
pip install -e .[viz,optimization,baselines]
```

**Or use requirements files:**

```bash
pip install -r requirements.txt  # Core only
pip install -r requirements-optional.txt  # Optional features
```

### For Developers

**Before:**

```bash
pip install -r requirements.txt
pip install pytest black flake8
```

**Now:**

```bash
# Install with dev dependencies
pip install -e .[dev]

# Setup pre-commit hooks
pre-commit install

# Run checks
black src/ tests/ scripts/
isort src/ tests/ scripts/
flake8 src/ tests/
mypy src/
pytest tests/ -v
```

### For CI/CD

**Before:**

```yaml
- run: pip install -r requirements.txt
```

**Now:**

```yaml
- run: |
    pip install -e .
    pip install pytest pytest-cov
```

## Installation Options

### Minimal (Core Only)

```bash
pip install -e .
```

### With Visualization

```bash
pip install -e .[viz]
```

### With Optimization Tools

```bash
pip install -e .[optimization]
```

### Everything

```bash
pip install -e .[full]
```

## Dependency Groups

### Core (Always Installed)

- numpy, pandas, scikit-learn, scipy
- deap (genetic algorithms)
- matplotlib, seaborn (basic plotting)
- pyyaml, tqdm

### Optional Extras

#### `viz` - Tree Visualization

- graphviz
- networkx

#### `optimization` - Hyperparameter Tuning

- optuna
- mlflow

#### `baselines` - Comparison Models

- xgboost
- lightgbm

#### `explainability` - Model Interpretation

- shap
- lime

#### `api` - Web Interface

- fastapi
- uvicorn
- pydantic

#### `dev` - Development Tools

- pytest, pytest-cov
- black, isort, flake8, mypy
- pre-commit

#### `all` - All Optional Features

- Installs: viz + optimization + baselines + explainability + api

#### `full` - Everything

- Installs: all + dev + docs

## Breaking Changes

### None! 🎉

This is a **non-breaking change**. Old installation methods still work:

```bash
# Still works
pip install -r requirements.txt
pip install -e .
```

But you'll see warnings about missing optional dependencies (which you can ignore or install as needed).

## Benefits of New System

### 1. **Cleaner Dependency Management**

- Core dependencies clearly separated from optional ones
- No unnecessary packages for basic usage
- Faster installation for minimal setups

### 2. **Better Developer Experience**

```bash
# One command for full dev setup
pip install -e .[dev]

# Automatic code formatting on commit
pre-commit install
```

### 3. **Improved CI/CD**

- Multi-platform testing
- Parallel test execution
- Better error reporting
- Package validation

### 4. **Future-Proof**

- PEP 517/518/621 compliant
- Ready for PyPI publication
- Modern tooling support

## Testing Your Migration

### 1. Clean Installation Test

```bash
# Remove old environment
deactivate
rm -rf venv/

# Create fresh environment
python -m venv venv
source venv/bin/activate

# Install with new method
pip install -e .

# Verify
python -c "import ga_trees; print('✓ Success')"
pytest tests/unit/ -v
```

### 2. Feature Test

```bash
# Test optional features
pip install -e .[viz]
python scripts/visualize_comprehensive.py

pip install -e .[optimization]
python scripts/hyperopt_with_optuna.py --preset fast --dataset iris
```

### 3. Development Test

```bash
pip install -e .[dev]
pre-commit run --all-files
pytest tests/ -v --cov=src/ga_trees
```

## Troubleshooting

### Issue: "Extra 'xyz' not found"

**Cause:** Typo in extra name or old pip version

**Solution:**

```bash
pip install --upgrade pip
pip install -e .[all]  # Note: brackets, not parentheses
```

### Issue: Pre-commit hooks failing

**Solution:**

```bash
# Update hooks
pre-commit autoupdate

# Run manually
pre-commit run --all-files
```

### Issue: Import errors after migration

**Solution:**

```bash
# Reinstall in editable mode
pip install -e . --force-reinstall --no-deps
```

## Rollback (If Needed)

If you need to revert:

```bash
# Use old requirements file
git checkout HEAD~1 requirements.txt
pip install -r requirements.txt
```

## Questions?

1. See [GitHub Issues](https://github.com/ibrah5em/ga-optimized-trees/issues)
1. Read [Contributing Guide](../../CONTRIBUTING.md)

## Summary

✅ **What to do:**

- Use `pip install -e .` for minimal installation
- Use `pip install -e .[all]` for everything
- Install `pre-commit` for development: `pip install -e .[dev]`

❌ **What NOT to do:**

- Don't manually install packages from requirements.txt
- Don't ignore pre-commit warnings
- Don't skip testing after migration

🎉 **Benefits:**

- Cleaner dependencies
- Faster installation
- Better tooling
- Future-proof setup
