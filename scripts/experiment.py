"""
FAST experiment script - 10x faster, better interpretability.

Supports both default parameters and YAML configuration files.

Usage:
    python scripts/experiment.py
    python scripts/experiment.py --config configs/default.yaml
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ga_trees.baselines import XGBoostBaseline
from ga_trees.data.dataset_loader import DatasetLoader
from ga_trees.evaluation.statistics import (
    ALPHA,
    DEFAULT_EQUIVALENCE_MARGIN,
    MIN_DATASETS_FOR_INFERENCE,
    compare_all_to_reference,
    equivalence_test,
    friedman_nemenyi,
    per_dataset_means,
    summarize,
)
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor
from ga_trees.ga.engine import (
    DEFAULT_GROWTH_STOP_PROB,
    GAConfig,
    GAEngine,
    Mutation,
    TreeInitializer,
)
from ga_trees.ga.split_points import MIDPOINT_STRATEGY
from ga_trees.reproducibility import build_seed_manifest, derive_fold_seed


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
    results = {
        "test_acc": [],
        "test_f1": [],
        "nodes": [],
        "depth": [],
        "features": [],
        "time": [],
        "seeds": [],
    }

    base_seed = config["experiment"]["random_state"]

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

        # Distinct per fold, but fixed across invocations. Reusing base_seed for
        # every fold would make all folds repeat one search, because
        # GAEngine.evolve seeds random/numpy globally.
        fold_seed = derive_fold_seed(base_seed, dataset_name, fold, method="ga")
        results["seeds"].append(fold_seed)

        # Use configuration
        ga_config = GAConfig(
            population_size=config["ga"]["population_size"],
            n_generations=config["ga"]["n_generations"],
            crossover_prob=config["ga"]["crossover_prob"],
            mutation_prob=config["ga"]["mutation_prob"],
            tournament_size=config["ga"]["tournament_size"],
            elitism_ratio=config["ga"]["elitism_ratio"],
            mutation_types=config["ga"]["mutation_types"],
            random_state=fold_seed,
            early_stopping_rounds=config["ga"].get("early_stopping_rounds"),
            early_stopping_tol=config["ga"].get("early_stopping_tol", 1e-6),
        )

        initializer = TreeInitializer(
            n_features=n_features,
            n_classes=n_classes,
            max_depth=config["tree"]["max_depth"],
            min_samples_split=config["tree"]["min_samples_split"],
            min_samples_leaf=config["tree"]["min_samples_leaf"],
            growth_stop_prob=config["tree"].get("growth_stop_prob", DEFAULT_GROWTH_STOP_PROB),
            split_strategy=config["tree"].get("split_strategy", MIDPOINT_STRATEGY),
        )

        # Fitness configuration
        fitness_config = config["fitness"]
        # Support both nested (weights.accuracy) and flat (accuracy_weight) formats
        if "weights" in fitness_config:
            acc_w = fitness_config["weights"]["accuracy"]
            interp_w = fitness_config["weights"]["interpretability"]
        else:
            acc_w = fitness_config.get("accuracy_weight", 0.65)
            interp_w = fitness_config.get("interpretability_weight", 0.35)
        fitness_calc = FitnessCalculator(
            mode=fitness_config["mode"],
            accuracy_weight=acc_w,
            interpretability_weight=interp_w,
            interpretability_weights=fitness_config["interpretability_weights"],
            classification_metric=fitness_config.get("classification_metric", "accuracy"),
            regression_metric=fitness_config.get("regression_metric", "neg_mse"),
        )

        mutation = Mutation(
            n_features=n_features,
            feature_ranges=feature_ranges,
            X=X_train,
            min_samples_leaf=config["tree"]["min_samples_leaf"],
            split_strategy=config["tree"].get("split_strategy", MIDPOINT_STRATEGY),
        )

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


#: Method every other method is tested against.
REFERENCE_MODEL = "GA-Optimized"

#: Baseline for the H2 equivalence claim in paper/PREREGISTRATION.md.
EQUIVALENCE_BASELINE = "CART"

#: Columns of results/stats-*.csv. Wider than the old (t, p, d) schema because
#: a reader has to be able to tell which test produced a number and whether it
#: was powered enough to mean anything.
STATS_CSV_FIELDS = [
    "test",
    "scope",
    "comparison",
    "n_datasets",
    "statistic",
    "p_value",
    "p_adjusted",
    "effect_size",
    "effect_type",
    "significant",
    "note",
]


def run_statistical_analysis(
    all_results, metric="test_acc", reference=None, equivalence_baseline=None
):
    """Compare methods across datasets and print the results.

    Every test pairs on *datasets*, using each dataset's mean over outer folds
    as a single observation. Fold-level paired tests are not reported at all:
    folds share training data, so their p-values are not interpretable
    (Dietterich 1998).

    Args:
        all_results: ``{dataset: {method: {metric: [per-fold values]}}}``.
        metric: Per-fold metric to aggregate and test on.
        reference: Method every other method is tested against. Defaults to
            ``REFERENCE_MODEL``.
        equivalence_baseline: Baseline for the H2 TOST. Defaults to
            ``EQUIVALENCE_BASELINE``. The nested harness names its tuned CART
            "CART (pruned)", so callers must say which baseline they mean
            rather than have the equivalence test silently skip.

    Returns:
        List of dicts matching ``STATS_CSV_FIELDS``, ready for CSV export.
    """
    reference = reference or REFERENCE_MODEL
    equivalence_baseline = equivalence_baseline or EQUIVALENCE_BASELINE
    print(f"\n{'='*70}")
    print("Statistical Analysis (paired across datasets)")
    print(f"{'='*70}\n")

    dataset_names, scores = per_dataset_means(all_results, metric=metric)
    rows = []

    if reference not in scores:
        print(f"  {reference} is missing from the results; no tests run.")
        return rows
    if len(dataset_names) < 2:
        print(f"  Only {len(dataset_names)} dataset(s); across-dataset inference needs >= 2.")
        return rows

    n = len(dataset_names)
    print(f"Unit of analysis: {n} dataset(s) — {', '.join(dataset_names)}")
    print(f"Each observation is a dataset's mean {metric} over outer folds.\n")

    if n < MIN_DATASETS_FOR_INFERENCE:
        print(
            f"  ⚠ UNDERPOWERED: with {n} datasets the smallest attainable two-sided\n"
            f"    signed-rank p-value is {2 / 2 ** n:.3f}. Reaching alpha={ALPHA} needs\n"
            f"    >= {MIN_DATASETS_FOR_INFERENCE} datasets. Everything below is descriptive.\n"
        )

    # --- Wilcoxon signed-rank, reference vs each baseline, Holm-corrected ---
    print(f"Wilcoxon signed-rank vs {reference} (Holm-corrected):")
    for comparison in compare_all_to_reference(scores, reference):
        if comparison.p_value is None:
            print(f"  {comparison.method_b:24s}: not computable — {comparison.note}")
        else:
            verdict = "significant" if comparison.significant else "ns"
            print(
                f"  {comparison.method_b:24s}: mean diff={comparison.mean_difference:+.4f}, "
                f"W={comparison.statistic:.1f}, p={comparison.p_value:.4f}, "
                f"p_holm={comparison.p_adjusted:.4f} ({verdict}), "
                f"d_z={comparison.effect_size:+.3f}"
            )
        rows.append(
            {
                "test": "wilcoxon_signed_rank",
                "scope": "across_datasets",
                "comparison": f"{comparison.method_a} vs {comparison.method_b}",
                "n_datasets": comparison.n_datasets,
                "statistic": comparison.statistic,
                "p_value": comparison.p_value,
                "p_adjusted": comparison.p_adjusted,
                "effect_size": comparison.effect_size,
                "effect_type": "cohens_dz",
                "significant": comparison.significant,
                "note": comparison.note,
            }
        )

    # --- TOST equivalence against the pre-registered margin (H2) ---
    if equivalence_baseline in scores:
        equivalence = equivalence_test(
            scores[reference],
            scores[equivalence_baseline],
            margin=DEFAULT_EQUIVALENCE_MARGIN,
            method_a=reference,
            method_b=equivalence_baseline,
        )
        print(
            f"\nTOST equivalence vs {equivalence_baseline} "
            f"(margin=±{equivalence.margin:.0%} absolute {metric}):"
        )
        print(
            f"  mean diff={equivalence.mean_difference:+.4f}, "
            f"{1 - 2 * ALPHA:.0%} CI=[{equivalence.ci_low:+.4f}, {equivalence.ci_high:+.4f}], "
            f"p={equivalence.p_value:.4f} → "
            f"{'EQUIVALENT' if equivalence.equivalent else 'NOT shown equivalent'}"
        )
        if not equivalence.equivalent:
            print("  A non-significant difference is not equivalence; do not report it as one.")
        rows.append(
            {
                "test": "tost_equivalence",
                "scope": "across_datasets",
                "comparison": f"{equivalence.method_a} vs {equivalence.method_b}",
                "n_datasets": equivalence.n_datasets,
                "statistic": "",
                "p_value": equivalence.p_value,
                "p_adjusted": "",
                "effect_size": equivalence.mean_difference,
                "effect_type": f"mean_difference (margin={equivalence.margin})",
                "significant": equivalence.equivalent,
                "note": equivalence.note,
            }
        )

    # --- Friedman omnibus + Nemenyi critical difference (Demsar 2006) ---
    friedman = friedman_nemenyi(scores)
    print("\nFriedman + Nemenyi (all methods):")
    ranked = sorted(friedman.average_ranks.items(), key=lambda kv: kv[1])
    print("  Average ranks (1 = best): " + ", ".join(f"{m}={r:.2f}" for m, r in ranked))
    if friedman.p_value is None:
        print(f"  Omnibus test not run — {friedman.note}")
    else:
        print(f"  chi2={friedman.statistic:.3f}, p={friedman.p_value:.4f}")
        print(f"  Nemenyi critical difference (alpha={ALPHA}): {friedman.critical_difference:.3f}")
        if friedman.p_value < ALPHA:
            for method in friedman.methods:
                if method == reference:
                    continue
                gap = friedman.rank_gap(reference, method)
                mark = "separated" if friedman.differs(reference, method) else "not separated"
                print(f"    {reference} vs {method:20s}: rank gap={gap:.2f} ({mark})")
        else:
            print("  Omnibus not significant; pairwise post-hoc comparisons are not licensed.")
    rows.append(
        {
            "test": "friedman",
            "scope": "across_datasets",
            "comparison": " vs ".join(friedman.methods),
            "n_datasets": friedman.n_datasets,
            "statistic": friedman.statistic,
            "p_value": friedman.p_value,
            "p_adjusted": "",
            "effect_size": friedman.critical_difference,
            "effect_type": "nemenyi_critical_difference",
            "significant": (friedman.p_value is not None and friedman.p_value < ALPHA),
            "note": friedman.note or "; ".join(f"{m}={r:.3f}" for m, r in ranked),
        }
    )

    # --- Per-dataset differences, descriptive only ---
    if equivalence_baseline in scores:
        print("\nPer-dataset differences (descriptive — no fold-level p-values):")
        for i, dataset_name in enumerate(dataset_names):
            ref_score = scores[reference][i]
            base_score = scores[equivalence_baseline][i]
            print(
                f"  {dataset_name:20s}: {reference}={ref_score:.4f}, "
                f"{equivalence_baseline}={base_score:.4f}, diff={ref_score - base_score:+.4f}"
            )

    return rows


def print_summary(all_results, config, config_name="default"):
    """Print summary with tree size analysis."""
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}\n")

    data = []
    for dataset_name, models in all_results.items():
        for model_name, results in models.items():
            acc = summarize(results["test_acc"])
            f1 = summarize(results["test_f1"])
            time_mean = np.mean(results["time"])

            row = {
                "Dataset": dataset_name,
                "Model": model_name,
                "Test Acc": f"{acc.mean:.4f} ± {acc.std:.4f}",
                "Test F1": f"{f1.mean:.4f} ± {f1.std:.4f}",
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

    # Statistical analysis. Inference is across datasets, never across CV folds:
    # folds share training data, so a paired test over them violates the
    # independence assumption and its p-value is not interpretable
    # (Dietterich 1998). The per-fold ttest_rel that used to live here is gone
    # for that reason, along with every star it printed.
    test_stats = run_statistical_analysis(all_results)

    if test_stats:
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        stats_date = datetime.now().strftime("%Y-%m-%d")
        stats_file = output_dir / f"stats-{config_name}-{stats_date}.csv"

        with open(stats_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=STATS_CSV_FIELDS)
            writer.writeheader()
            for entry in test_stats:
                writer.writerow({key: entry.get(key, "") for key in STATS_CSV_FIELDS})

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
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Save the seeds actually used. Without this, "seeded" is an assertion rather
    # than something a reader can check - see results/PROVENANCE.md.
    base_seed = config["experiment"]["random_state"]
    seeds_file = output_dir / f"seeds-{config_name}-{date_str}.json"
    manifest = build_seed_manifest(
        base_seed=base_seed,
        dataset_names=list(all_results.keys()),
        n_folds=config["experiment"]["cv_folds"],
        methods=("ga",),
    )
    # The GA derives a per-fold seed; the sklearn baselines take base_seed
    # directly, since their fits are deterministic given (data, random_state).
    manifest["baseline_random_state"] = base_seed
    manifest["recorded"] = {
        dataset: {model: res["seeds"] for model, res in models.items() if res.get("seeds")}
        for dataset, models in all_results.items()
    }
    with open(seeds_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")
    print(f"✓ Config saved to: {config_file}")
    print(f"✓ Seeds saved to: {seeds_file}")


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
        acc = fc["weights"]["accuracy"]
        interp = fc["weights"]["interpretability"]
        print(f"  Fitness: acc_weight={acc}, interp_weight={interp}")
    else:
        acc = fc.get("accuracy_weight", "N/A")
        interp = fc.get("interpretability_weight", "N/A")
        print(f"  Fitness: acc_weight={acc}, interp_weight={interp}")
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
