#!/usr/bin/env python3
"""
Validate package setup and dependencies.

Run this script to check if everything is properly configured.
"""

import shlex
import subprocess
import sys
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def check_import(module_name, package_name=None, optional=False):
    """Check if a module can be imported."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        if optional:
            print(f"⚠ {package_name} (optional - not installed)")
        else:
            print(f"✗ {package_name} (REQUIRED - missing!)")
        return not optional


def check_file_exists(filepath):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✓ {filepath}")
        return True
    else:
        print(f"✗ {filepath} (missing)")
        return False


def run_command(cmd, description):
    """Run a shell command and report success."""
    try:
        result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✓ {description}")
            return True
        else:
            print(f"✗ {description}: {result.stderr[:100]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {description} (timeout)")
        return False
    except Exception as e:
        print(f"✗ {description}: {e}")
        return False


def main():
    """Run all validation checks."""
    print("\n" + "🔍 " * 35)
    print("  GA-OPTIMIZED TREES - SETUP VALIDATION")
    print("🔍 " * 35)

    all_passed = True

    # 1. Python Version
    print_header("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    if version >= (3, 8):
        print("✓ Python version OK")
    else:
        print("✗ Python 3.8+ required")
        all_passed = False

    # 2. Core Dependencies
    print_header("Core Dependencies")
    core_deps = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("scipy", "scipy"),
        ("deap", "deap"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("yaml", "pyyaml"),
        ("tqdm", "tqdm"),
    ]

    for module, package in core_deps:
        if not check_import(module, package, optional=False):
            all_passed = False

    # 3. Package Installation
    print_header("Package Installation")
    try:
        import ga_trees

        print("✓ ga_trees package found")
        print(f"  Location: {ga_trees.__file__}")
    except ImportError:
        print("✗ ga_trees package not found")
        print("  Run: pip install -e .")
        all_passed = False

    # 4. Optional Dependencies
    print_header("Optional Dependencies")
    optional_deps = [
        ("graphviz", "graphviz (viz)"),
        ("networkx", "networkx (viz)"),
        ("optuna", "optuna (optimization)"),
        ("mlflow", "mlflow (optimization)"),
        ("xgboost", "xgboost (baselines)"),
        ("lightgbm", "lightgbm (baselines)"),
        ("shap", "shap (explainability)"),
        ("lime", "lime (explainability)"),
        ("fastapi", "fastapi (api)"),
    ]

    for module, package in optional_deps:
        check_import(module, package, optional=True)

    # 5. Development Tools
    print_header("Development Tools (Optional)")
    dev_tools = [
        ("pytest", "pytest"),
        ("black", "black"),
        ("isort", "isort"),
        ("flake8", "flake8"),
        ("mypy", "mypy"),
    ]

    for module, package in dev_tools:
        check_import(module, package, optional=True)

    # 6. Configuration Files
    print_header("Configuration Files")
    config_files = [
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        "README.md",
        "LICENSE",
    ]

    for filepath in config_files:
        if not check_file_exists(filepath):
            all_passed = False

    # 7. Source Structure
    print_header("Source Code Structure")
    required_paths = [
        "src/ga_trees/__init__.py",
        "src/ga_trees/genotype/tree_genotype.py",
        "src/ga_trees/ga/engine.py",
        "src/ga_trees/fitness/calculator.py",
        "tests/unit/test_genotype.py",
        "scripts/train.py",
        "configs/default.yaml",
    ]

    for filepath in required_paths:
        if not check_file_exists(filepath):
            all_passed = False

    # 8. Quick Functional Test
    print_header("Functional Tests")

    try:
        from ga_trees.genotype.tree_genotype import (
            TreeGenotype,
            create_internal_node,
            create_leaf_node,
        )

        left = create_leaf_node(0, 1)
        right = create_leaf_node(1, 1)
        root = create_internal_node(0, 0.5, left, right, 0)
        tree = TreeGenotype(root=root, n_features=4, n_classes=2)
        print("✓ Tree creation")
    except Exception as e:
        print(f"✗ Tree creation: {e}")
        all_passed = False

    try:
        import numpy as np

        from ga_trees.fitness.calculator import FitnessCalculator

        X = np.random.rand(10, 4)
        y = np.random.randint(0, 2, 10)
        fitness_calc = FitnessCalculator()
        fitness_calc.calculate_fitness(tree, X, y)
        print("✓ Fitness calculation")
    except Exception as e:
        print(f"✗ Fitness calculation: {e}")
        all_passed = False

    # 9. Command Line Tools
    print_header("Command Line Tools")
    run_command("python scripts/train.py --help", "Training script help")
    run_command("python scripts/experiment.py --help", "Experiment script help")

    # 10. Summary
    print_header("Summary")
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("\nYou're ready to use ga-optimized-trees!")
        print("\nNext steps:")
        print("  1. Run quick demo: python scripts/train.py --dataset iris --generations 5")
        print("  2. Run full experiment: python scripts/experiment.py")
        print("  3. Check documentation: docs/README.md")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nTo fix:")
        print("  1. Install core dependencies: pip install -e .")
        print("  2. Install optional features: pip install -e .[all]")
        print("  3. Install dev tools: pip install -e .[dev]")
        print("  4. Re-run this script: python validate_setup.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
