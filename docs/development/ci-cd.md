# CI/CD Pipeline

Continuous integration and deployment setup.

## GitHub Actions Workflow

Location: `.github/workflows/ci.yml`

### Jobs

#### 1. Test Job

Runs on multiple platforms and Python versions:

- **Platforms**: Ubuntu, Windows, macOS
- **Python**: 3.8, 3.9, 3.10, 3.11, 3.12

**Steps:**

1. Checkout code
1. Setup Python with pip cache
1. Install dependencies
1. Run unit tests with coverage
1. Run quick training test
1. Upload coverage to Codecov

#### 2. Lint Job

Code quality checks:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking (continue-on-error)

#### 3. Integration Job

Integration tests:

- Full integration test suite
- Experiment script validation

#### 4. Documentation Job

Build documentation:

- Sphinx HTML build
- Check for broken links

#### 5. Package Job

Build and validate package:

- Build wheel and source distribution
- Check with twine
- Upload artifacts

#### 6. Notification Job

Final status check:

- ✅ All checks passed
- ❌ Some checks failed

## Running CI Locally

### Quick Test

```bash
# Run what CI runs
pytest tests/unit/ -v --cov=src/ga_trees
black --check src/ tests/ scripts/
isort --check-only src/ tests/ scripts/
flake8 src/ tests/ scripts/
```

### Full CI Simulation

```bash
# Install dev dependencies
pip install -e .[dev]

# Run all checks
pytest tests/ -v --cov=src/ga_trees --cov-report=html
black src/ tests/ scripts/
isort src/ tests/ scripts/
flake8 src/ tests/
mypy src/
```

## Pre-commit Hooks

**Setup:**

```bash
pip install pre-commit
pre-commit install
```

**Hooks run automatically on commit:**

- Trailing whitespace removal
- End-of-file fixer
- YAML validation
- Black formatting
- isort import sorting
- flake8 linting
- mypy type checking

**Manual run:**

```bash
pre-commit run --all-files
```

## Code Coverage

**Target:** 80%+ coverage

**Generate report:**

```bash
pytest tests/ --cov=src/ga_trees --cov-report=html
open htmlcov/index.html
```

## Deployment

### PyPI Release (Future)

```bash
# Build
python -m build

# Check
twine check dist/*

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```
