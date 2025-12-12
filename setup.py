"""
Setup configuration for ga-optimized-trees.

This file is maintained for backward compatibility.
Modern installation should use pyproject.toml.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Genetic Algorithm Framework for Evolving Interpretable Decision Trees"

# Core dependencies
INSTALL_REQUIRES = [
    "numpy>=1.24.0,<2.0.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "scipy>=1.10.0",
    "deap>=1.4.1",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "viz": [
        "graphviz>=0.20.0",
        "networkx>=3.1",
    ],
    "optimization": [
        "optuna>=3.4.0",
        "mlflow>=2.8.0",
    ],
    "baselines": [
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
    ],
    "explainability": [
        "shap>=0.43.0",
        "lime>=0.2.0",
    ],
    "api": [
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
    ],
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-xdist>=3.3.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.1.0",
        "mypy>=1.5.0",
        "pre-commit>=3.5.0",
    ],
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.3.0",
        "sphinx-autodoc-typehints>=1.24.0",
    ],
}

# Convenience groups
EXTRAS_REQUIRE["all"] = [
    dep
    for group in ["viz", "optimization", "baselines", "explainability", "api"]
    for dep in EXTRAS_REQUIRE[group]
]
EXTRAS_REQUIRE["full"] = [dep for deps in EXTRAS_REQUIRE.values() for dep in deps]

setup(
    name="ga-optimized-trees",
    version="1.0.0",
    author="Ibrahem Hasaki, LuF8y",
    author_email="ibrah5em@github.com",
    description="Genetic Algorithm Framework for Evolving Interpretable Decision Trees",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ibrah5em/ga-optimized-trees",
    project_urls={
        "Bug Tracker": "https://github.com/ibrah5em/ga-optimized-trees/issues",
        "Documentation": "https://github.com/ibrah5em/ga-optimized-trees/tree/main/docs",
        "Source Code": "https://github.com/ibrah5em/ga-optimized-trees",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "ga-trees=ga_trees.cli:main",
            "ga-train=ga_trees.scripts.train:main",
            "ga-experiment=ga_trees.scripts.experiment:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ga_trees": ["py.typed"],
    },
    zip_safe=False,
)
