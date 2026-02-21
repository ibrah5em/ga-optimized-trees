"""
FAST experiment script - 10x faster, better interpretability.

Supports both default parameters and YAML configuration files.

Usage:
    python scripts/experiment.py
    python scripts/experiment.py --config configs/default.yaml
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scipy import stats
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ga_trees.baselines import XGBoostBaseline
from ga_trees.data.dataset_loader import DatasetLoader
from ga_trees.fitness.calculator import FitnessCalculator, InterpretabilityCalculator, TreePredictor
from ga_trees.ga.engine import GAConfig, GAEngine, Mutation, TreeInitializer


class FastInterpretabilityCalculator(InterpretabilityCalculator):
    """FIXED interpretability that properly penalizes large trees."""

    @staticmethod
    def calculate_composite_score(tree, weights):
        """Fixed composite score."""
        score = 0.0

        # Node complexity - PROPERLY penalize large trees
        if "node_complexity" in weights:
            num_nodes = tree.get_num_nodes()
            # Exponential penalty for large trees
            node_score = np.exp(-num_nodes / 15.0)  # Sweet spot around 15 nodes
            score += weights["node_complexity"] * node_score

        # Feature coherence
        if "feature_coherence" in weights:
            internal_nodes = tree.get_internal_nodes()
            if internal_nodes:
                features_used = tree.get_features_used()
                coherence = 1.0 - (len(features_used) / max(len(internal_nodes), 1))
                score += weights["feature_coherence"] * max(0.0, coherence)

        # Tree balance - CAPPED to avoid rewarding overgrowth
        if "tree_balance" in weights:
            balance = tree.get_tree_balance()
            # Only reward balance if tree isn't too large
            if tree.get_num_nodes() <= 30:
                score += weights["tree_balance"] * balance
            else:
                # Penalty for large unbalanced trees
                score += weights["tree_balance"] * balance * 0.5

        # Semantic coherence
        if "semantic_coherence" in weights:
            leaves = tree.get_all_leaves()
            if len(leaves) > 1:
                predictions = [l.prediction for l in leaves if l.prediction is not None]
                if predictions:
                    unique = len(set(predictions))
                    # More coherent if fewer unique predictions
                    semantic = 1.0 - (unique / len(predictions))
                    score += weights["semantic_coherence"] * semantic

        return score


class FastFitnessCalculator(FitnessCalculator):
    """Faster fitness with better interpretability."""

    def __init__(
        self,
        mode="weighted_sum",
        accuracy_weight=0.7,
        interpretability_weight=0.3,
        interpretability_weights=None,
    ):
        super().__init__(mode, accuracy_weight, interpretability_weight, interpretability_weights)
        # Use fixed interpretability calculator
        self.interp_calc = FastInterpretabilityCalculator()


def load_config(config_path=None):
    """Load configuration from YAML file or use defaults."""
    default_config = {
        "ga": {
            "population_size": 50,
            "n_generations": 30,
            "crossover_prob": 0.7,
            "mutation_prob": 0.2,
            "tournament_size": 3,
            "elitism_ratio": 0.15,
            "mutation_types": {
                "threshold_perturbation": 0.5,
                "feature_replacement": 0.3,
                "prune_subtree": 0.15,
                "expand_leaf": 0.05,
            },
        },
        "tree": {"max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 5},
        "fitness": {
            "mode": "weighted_sum",
            "weights": {
                "accuracy": 0.65,
                "interpretability": 0.35,
            },
            "interpretability_weights": {
                "node_complexity": 0.6,
                "feature_coherence": 0.2,
                "tree_balance": 0.1,
                "semantic_coherence": 0.1,
            },
        },
        "experiment": {
            "datasets": ["iris", "wine", "breast_cancer"],
            "cv_folds": 5,
            "random_state": 42,
        },
    }

    if config_path and Path(config_path).exists():
        print(f"Loading configuration from: {config_path}")
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)

        # Deep merge configurations
        config = _merge_configs(default_config, user_config)
    else:
        if config_path:
            print(f"Config file {config_path} not found, using defaults")
        else:
            print("No config specified, using defaults")
        config = default_config

    return config


def _merge_configs(default, user):
    """Recursively merge user configuration with defaults."""
    result = default.copy()

    for key, value in user.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def load_dataset(name, label_column=None):
    """Load dataset by name. Prefer `DatasetLoader`, fall back to sklearn loaders.

    Returns full (X, y) without train/test split.
    """
    # If `name` is a path to a local file, load it directly
    p = Path(name)
    if p.exists():
        ext = p.suffix.lower()
        if ext in (".csv", ".txt", ".tsv"):
            df = pd.read_csv(p)
        elif ext in (".xls", ".xlsx"):
            df = pd.read_excel(p)
        else:
            raise ValueError(f"Unsupported file extension for dataset: {ext}")

        if df.shape[1] < 2:
            raise ValueError(
                "Dataset file must contain at least one feature column and one target column"
            )

        # Determine label column
        if label_column is None:
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        else:
            if isinstance(label_column, int) or (
                isinstance(label_column, str) and label_column.isdigit()
            ):
                idx = int(label_column)
                if idx < 0 or idx >= df.shape[1]:
                    raise IndexError(f"label column index out of range: {idx}")
                y = df.iloc[:, idx].values
                X = df.drop(df.columns[idx], axis=1).values
            else:
                col = label_column
                if col not in df.columns:
                    raise ValueError(f"label column '{col}' not found in file")
                y = df[col].values
                X = df.drop(columns=[col]).values

        return X, y

    # Fast path for common sklearn datasets
    if name in {"iris", "wine", "breast_cancer"}:
        if name == "iris":
            return load_iris(return_X_y=True)
        if name == "wine":
            return load_wine(return_X_y=True)
        if name == "breast_cancer":
            return load_breast_cancer(return_X_y=True)

    # Otherwise use DatasetLoader which supports OpenML and other sources
    try:
        loader = DatasetLoader()
        data = loader.load_dataset(name, test_size=0.2, standardize=False, stratify=True)

        if isinstance(data, dict):
            X = np.vstack([data["X_train"], data["X_test"]])
            y = np.hstack([data["y_train"], data["y_test"]])
            return X, y

        raise ValueError(f"DatasetLoader returned unexpected result for '{name}'")
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{name}': {e}")


def run_ga_experiment(X, y, dataset_name, config, n_folds=5):
    """Run GA with FAST settings using configuration."""
    print(f"\n{'='*70}")
    print(f"Running FAST GA on {dataset_name}")
    print(f"{'='*70}")

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=config["experiment"]["random_state"]
    )
    results = {"test_acc": [], "test_f1": [], "nodes": [], "depth": [], "features": [], "time": []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_folds}...", end=" ", flush=True)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Setup
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y))
        feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) for i in range(n_features)}

        # Use configuration
        ga_config = GAConfig(
            population_size=config["ga"]["population_size"],
            n_generations=config["ga"]["n_generations"],
            crossover_prob=config["ga"]["crossover_prob"],
            mutation_prob=config["ga"]["mutation_prob"],
            tournament_size=config["ga"]["tournament_size"],
            elitism_ratio=config["ga"]["elitism_ratio"],
            mutation_types=config["ga"]["mutation_types"],
        )

        initializer = TreeInitializer(
            n_features=n_features,
            n_classes=n_classes,
            max_depth=config["tree"]["max_depth"],
            min_samples_split=config["tree"]["min_samples_split"],
            min_samples_leaf=config["tree"]["min_samples_leaf"],
        )

        # FAST FITNESS with FIXED interpretability
        fitness_config = config["fitness"]
        # Support both nested (weights.accuracy) and flat (accuracy_weight) formats
        if "weights" in fitness_config:
            acc_w = fitness_config["weights"]["accuracy"]
            interp_w = fitness_config["weights"]["interpretability"]
        else:
            acc_w = fitness_config.get("accuracy_weight", 0.65)
            interp_w = fitness_config.get("interpretability_weight", 0.35)
        fitness_calc = FastFitnessCalculator(
            mode=fitness_config["mode"],
            accuracy_weight=acc_w,
            interpretability_weight=interp_w,
            interpretability_weights=fitness_config["interpretability_weights"],
        )

        mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)

        # Train
        start = time.time()
        ga_engine = GAEngine(ga_config, initializer, fitness_calc.calculate_fitness, mutation)
        best_tree = ga_engine.evolve(X_train, y_train, verbose=False)
        elapsed = time.time() - start

        # Evaluate
        predictor = TreePredictor()
        y_pred = predictor.predict(best_tree, X_test)

        results["test_acc"].append(accuracy_score(y_test, y_pred))
        results["test_f1"].append(f1_score(y_test, y_pred, average="weighted"))
        results["nodes"].append(best_tree.get_num_nodes())
        results["depth"].append(best_tree.get_depth())
        # Record number of distinct features used by the GA tree
        try:
            results["features"].append(best_tree.get_num_features_used())
        except Exception:
            results["features"].append(np.nan)
        results["time"].append(elapsed)

        print(
            f"Acc={results['test_acc'][-1]:.3f}, "
            f"Nodes={results['nodes'][-1]}, "
            f"Time={elapsed:.1f}s"
        )

    return results


def run_cart_experiment(X, y, dataset_name, config, n_folds=5):
    """Run CART baseline."""
    print(f"\n{'='*70}")
    print(f"Running CART on {dataset_name}")
    print(f"{'='*70}")

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=config["experiment"]["random_state"]
    )
    results = {"test_acc": [], "test_f1": [], "nodes": [], "depth": [], "features": [], "time": []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_folds}...", end=" ")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        start = time.time()
        model = DecisionTreeClassifier(
            max_depth=config["tree"]["max_depth"], random_state=config["experiment"]["random_state"]
        )
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        y_pred = model.predict(X_test)

        results["test_acc"].append(accuracy_score(y_test, y_pred))
        results["test_f1"].append(f1_score(y_test, y_pred, average="weighted"))
        results["nodes"].append(model.tree_.node_count)
        results["depth"].append(model.tree_.max_depth)
        # Number of unique features used by CART (ignore -2/-1 placeholders)
        try:
            used = np.unique(model.tree_.feature[model.tree_.feature >= 0])
            results["features"].append(len(used))
        except Exception:
            results["features"].append(np.nan)
        results["time"].append(elapsed)

        print(f"Acc={results['test_acc'][-1]:.3f}, Time={elapsed:.1f}s")

    return results


def run_rf_experiment(X, y, dataset_name, config, n_folds=5):
    """Run Random Forest baseline."""
    print(f"\n{'='*70}")
    print(f"Running Random Forest on {dataset_name}")
    print(f"{'='*70}")

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=config["experiment"]["random_state"]
    )
    results = {"test_acc": [], "test_f1": [], "time": []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_folds}...", end=" ")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        start = time.time()
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=config["tree"]["max_depth"],
            random_state=config["experiment"]["random_state"],
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        y_pred = model.predict(X_test)

        results["test_acc"].append(accuracy_score(y_test, y_pred))
        results["test_f1"].append(f1_score(y_test, y_pred, average="weighted"))
        results["time"].append(elapsed)

        print(f"Acc={results['test_acc'][-1]:.3f}, Time={elapsed:.1f}s")

    return results


def run_xgboost_experiment(X, y, dataset_name, config, n_folds=5):
    """Run XGBoost baseline (if available)."""
    print(f"\n{'='*70}")
    print(f"Running XGBoost on {dataset_name}")
    print(f"{'='*70}")

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=config["experiment"]["random_state"]
    )
    results = {"test_acc": [], "test_f1": [], "time": []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_folds}...", end=" ")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        start = time.time()
        # Instantiate baseline wrapper which will handle missing xgboost
        model = XGBoostBaseline(
            max_depth=config["tree"]["max_depth"],
            n_estimators=100,
            random_state=config["experiment"]["random_state"],
        )
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        # If xgboost isn't installed the baseline sets model.model to None
        if model.model is None:
            print("skipping (XGBoost not installed)")
            results["test_acc"].append(np.nan)
            results["test_f1"].append(np.nan)
            results["time"].append(elapsed)
            continue

        y_pred = model.predict(X_test)

        results["test_acc"].append(accuracy_score(y_test, y_pred))
        results["test_f1"].append(f1_score(y_test, y_pred, average="weighted"))
        results["time"].append(elapsed)

        print(f"Acc={results['test_acc'][-1]:.3f}, Time={elapsed:.1f}s")

    return results


def print_summary(all_results, config, config_name="default"):
    """Print summary with tree size analysis."""
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}\n")

    data = []
    for dataset_name, models in all_results.items():
        for model_name, results in models.items():
            acc_mean = np.mean(results["test_acc"])
            acc_std = np.std(results["test_acc"])
            f1_mean = np.mean(results["test_f1"])
            f1_std = np.std(results["test_f1"])
            time_mean = np.mean(results["time"])

            row = {
                "Dataset": dataset_name,
                "Model": model_name,
                "Test Acc": f"{acc_mean:.4f} ± {acc_std:.4f}",
                "Test F1": f"{f1_mean:.4f} ± {f1_std:.4f}",
                "Time (s)": f"{time_mean:.2f}",
            }

            if "nodes" in results:
                row["Nodes"] = f"{np.mean(results['nodes']):.1f}"
                row["Depth"] = f"{np.mean(results['depth']):.1f}"

            data.append(row)

    df = pd.DataFrame(data)
    print(df.to_string(index=False))

    # Tree size comparison
    print(f"\n{'='*70}")
    print("Tree Size Analysis (GA vs CART)")
    print(f"{'='*70}\n")

    test_stats = []

    for dataset_name in all_results.keys():
        ga_nodes = np.mean(all_results[dataset_name]["GA-Optimized"]["nodes"])
        cart_nodes = np.mean(all_results[dataset_name]["CART"]["nodes"])
        ratio = ga_nodes / cart_nodes

        status = (
            "✓ Smaller"
            if ratio < 1.0
            else (
                "✓✓ Much smaller" if ratio < 0.7 else ("~ Similar" if ratio < 1.3 else "✗ Larger")
            )
        )

        print(
            f"{dataset_name:20s}: GA={ga_nodes:5.1f}, CART={cart_nodes:5.1f}, "
            f"Ratio={ratio:.2f}x  {status}"
        )

    # Statistical tests
    print(f"\n{'='*70}")
    print("Statistical Tests (GA vs CART)")
    print(f"{'='*70}\n")

    for dataset_name in all_results.keys():
        ga_acc = all_results[dataset_name]["GA-Optimized"]["test_acc"]
        cart_acc = all_results[dataset_name]["CART"]["test_acc"]

        t_stat, p_value = stats.ttest_rel(ga_acc, cart_acc)

        pooled_std = np.sqrt((np.var(ga_acc) + np.var(cart_acc)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(ga_acc) - np.mean(cart_acc)) / pooled_std
        else:
            cohens_d = 0.0

        sig = (
            "***"
            if p_value < 0.001
            else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
        )

        print(f"{dataset_name:20s}: t={t_stat:6.3f}, p={p_value:.4f} {sig}, d={cohens_d:.3f}")

        # collect stats for CSV export
        test_stats.append(
            {
                "test_name": "GA vs CART",
                "dataset": dataset_name,
                "t": float(t_stat),
                "p": float(p_value),
                "d": float(cohens_d),
            }
        )

    # Adjust p-values (Bonferroni) and write statistics CSV for auditability
    if test_stats:
        m = len(test_stats)
        for entry in test_stats:
            entry["p_adjusted"] = min(entry["p"] * m, 1.0)

        # ensure results directory exists
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        stats_date = datetime.now().strftime("%Y-%m-%d")
        stats_file = output_dir / f"stats-{config_name}-{stats_date}.csv"

        with open(stats_file, "w", newline="") as csvfile:
            fieldnames = ["test_name", "dataset", "t", "p", "p_adjusted", "d"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in test_stats:
                writer.writerow(
                    {
                        "test_name": entry["test_name"],
                        "dataset": entry["dataset"],
                        "t": entry["t"],
                        "p": entry["p"],
                        "p_adjusted": entry["p_adjusted"],
                        "d": entry["d"],
                    }
                )

        print(f"\n✓ Statistical test details saved to: {stats_file}")

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    results_file = output_dir / f"result-{config_name}-{date_str}.csv"
    df.to_csv(results_file, index=False)

    # Save configuration used
    config_file = output_dir / f"config-{config_name}-{date_str}.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n✓ Results saved to: {results_file}")
    print(f"✓ Config saved to: {config_file}")


def main():
    """Run FAST experiments with configurable parameters."""
    parser = argparse.ArgumentParser(description="Run GA-optimized decision tree experiments")
    parser.add_argument("--config", type=str, help="Path to configuration YAML file")
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="(optional) Label column name or index for local files (default: last column)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of datasets to run (overrides config experiment.datasets)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # derive a compact config name for file naming
    config_name = Path(args.config).stem if args.config else "default"

    print(f"\n{'='*70}")
    print("GA-Optimized Decision Trees: FAST Version")
    print("Optimized for: Speed + Small Trees")
    if args.config:
        print(f"Configuration: {args.config}")
    else:
        print("Configuration: Default parameters")
    print(f"{'='*70}")

    # Determine datasets (config or CLI override)
    if args.datasets:
        chosen_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        chosen_datasets = config["experiment"]["datasets"]

    # Print key configuration parameters
    print("\nKey Configuration:")
    print(f"  GA: {config['ga']['population_size']} pop, {config['ga']['n_generations']} gen")
    print(f"  Tree: max_depth={config['tree']['max_depth']}")
    fc = config["fitness"]
    if "weights" in fc:
        print(f"  Fitness: acc_weight={fc['weights']['accuracy']}, interp_weight={fc['weights']['interpretability']}")
    else:
        print(f"  Fitness: acc_weight={fc.get('accuracy_weight', 'N/A')}, interp_weight={fc.get('interpretability_weight', 'N/A')}")
    print(f"  Datasets: {', '.join(chosen_datasets)}")

    datasets = chosen_datasets
    all_results = {}

    total_start = time.time()

    # parse label column argument
    label_col = args.label_column
    if label_col is not None and isinstance(label_col, str) and label_col.isdigit():
        label_col = int(label_col)

    for dataset_name in datasets:
        X, y = load_dataset(dataset_name, label_column=label_col)
        print(f"\n{dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")

        dataset_results = {}
        dataset_results["GA-Optimized"] = run_ga_experiment(
            X, y, dataset_name, config, n_folds=config["experiment"]["cv_folds"]
        )
        dataset_results["CART"] = run_cart_experiment(
            X, y, dataset_name, config, n_folds=config["experiment"]["cv_folds"]
        )
        dataset_results["Random Forest"] = run_rf_experiment(
            X, y, dataset_name, config, n_folds=config["experiment"]["cv_folds"]
        )

        # XGBoost baseline (may be skipped if xgboost not installed)
        dataset_results["XGBoost"] = run_xgboost_experiment(
            X, y, dataset_name, config, n_folds=config["experiment"]["cv_folds"]
        )

        all_results[dataset_name] = dataset_results

    total_time = time.time() - total_start

    print_summary(all_results, config, config_name)

    print(f"\n{'='*70}")
    print(f"Total Time: {total_time:.1f}s (~{total_time/60:.1f} minutes)")
    print("Experiment Complete! 🎉")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
