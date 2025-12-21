"""
prepare_professor_consultation_PAPER_EXACT.py

Generates statistical consultation files using EXACT parameters from the paper:
"Evolving Interpretable Decision Trees via Multi-Objective Genetic Algorithms"

CRITICAL: Uses paper configuration to match published results.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_paper_config():
    """Load EXACT configuration from paper."""

    # Try to load from YAML file first
    config_path = Path("configs/paper_config.yaml")
    if config_path.exists():
        print(f"✓ Loading paper config from: {config_path}")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    # Otherwise use hardcoded paper values
    print("⚠ Config file not found, using hardcoded paper values")
    return {
        "ga": {
            "population_size": 80,
            "n_generations": 40,
            "crossover_prob": 0.72,
            "mutation_prob": 0.18,
            "tournament_size": 4,
            "elitism_ratio": 0.12,
            "mutation_types": {
                "threshold_perturbation": 0.45,
                "feature_replacement": 0.25,
                "prune_subtree": 0.25,
                "expand_leaf": 0.05,
            },
        },
        "tree": {"max_depth": 6, "min_samples_split": 8, "min_samples_leaf": 3},
        "fitness": {
            "mode": "weighted_sum",
            "weights": {"accuracy": 0.68, "interpretability": 0.32},
            "interpretability_weights": {
                "node_complexity": 0.50,
                "feature_coherence": 0.10,
                "tree_balance": 0.10,
                "semantic_coherence": 0.30,
            },
        },
        "experiment": {
            "datasets": ["iris", "wine", "breast_cancer"],
            "cv_folds": 20,
            "random_state": 42,
        },
    }


def verify_paper_configuration(config: dict):
    """Verify configuration matches paper specifications."""

    print("\n" + "=" * 70)
    print("VERIFYING PAPER CONFIGURATION")
    print("=" * 70)

    checks = []

    # GA parameters
    checks.append(("Population Size", config["ga"]["population_size"], 80))
    checks.append(("Generations", config["ga"]["n_generations"], 40))
    checks.append(("Crossover Prob", config["ga"]["crossover_prob"], 0.72))
    checks.append(("Mutation Prob", config["ga"]["mutation_prob"], 0.18))
    checks.append(("Tournament Size", config["ga"]["tournament_size"], 4))
    checks.append(("Elitism Ratio", config["ga"]["elitism_ratio"], 0.12))

    # Tree constraints
    checks.append(("Max Depth", config["tree"]["max_depth"], 6))
    checks.append(("Min Samples Split", config["tree"]["min_samples_split"], 8))
    checks.append(("Min Samples Leaf", config["tree"]["min_samples_leaf"], 3))

    # Fitness weights
    checks.append(("Accuracy Weight", config["fitness"]["weights"]["accuracy"], 0.68))
    checks.append(
        ("Interpretability Weight", config["fitness"]["weights"]["interpretability"], 0.32)
    )

    # CV folds
    checks.append(("CV Folds", config["experiment"]["cv_folds"], 20))

    all_correct = True
    for name, actual, expected in checks:
        match = "✓" if actual == expected else "✗"
        if actual != expected:
            all_correct = False
            print(f"  {match} {name:25s}: {actual:8} (expected {expected})")
        else:
            print(f"  {match} {name:25s}: {actual}")

    if not all_correct:
        print("\n❌ WARNING: Configuration does not match paper!")
        print("   Results may differ from published values.")
        response = input("\n   Continue anyway? (yes/no): ")
        if response.lower() != "yes":
            sys.exit(1)
    else:
        print("\n✓ Configuration matches paper exactly")

    print("=" * 70)


def create_dataset_excel_files():
    """Generate individual Excel files for each dataset with metadata."""

    print("\n" + "=" * 70)
    print("GENERATING DATASET FILES")
    print("=" * 70)

    from sklearn.datasets import load_breast_cancer, load_iris, load_wine

    datasets = {
        "iris": {
            "loader": load_iris,
            "description": "Iris flower classification (Fisher, 1936)",
            "samples": 150,
            "features": 4,
            "classes": 3,
            "source": "UCI ML Repository",
        },
        "wine": {
            "loader": load_wine,
            "description": "Wine recognition (Forina et al., 1991)",
            "samples": 178,
            "features": 13,
            "classes": 3,
            "source": "UCI ML Repository",
        },
        "breast_cancer": {
            "loader": load_breast_cancer,
            "description": "Wisconsin Breast Cancer (Wolberg et al., 1995)",
            "samples": 569,
            "features": 30,
            "classes": 2,
            "source": "UCI ML Repository",
        },
    }

    results_dir = Path("results/professor_consultation")
    results_dir.mkdir(parents=True, exist_ok=True)

    for name, info in datasets.items():
        print(f"\n📊 {name}_dataset.xlsx...")

        data = info["loader"]()
        X = data.data
        y = data.target

        excel_path = results_dir / f"{name}_dataset.xlsx"

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

            # Sheet 1: Raw Data
            df_data = pd.DataFrame(X, columns=data.feature_names)
            df_data["Target"] = y
            df_data["Target_Name"] = [data.target_names[t] for t in y]
            df_data.to_excel(writer, sheet_name="Raw_Data", index=True)

            # Sheet 2: Feature Descriptions
            feature_info = []
            for i, fname in enumerate(data.feature_names):
                feature_info.append(
                    {
                        "Feature_Number": i + 1,
                        "Feature_Name": fname,
                        "Mean": f"{X[:, i].mean():.4f}",
                        "Std": f"{X[:, i].std():.4f}",
                        "Min": f"{X[:, i].min():.4f}",
                        "Max": f"{X[:, i].max():.4f}",
                        "Type": "Continuous",
                    }
                )

            pd.DataFrame(feature_info).to_excel(
                writer, sheet_name="Feature_Descriptions", index=False
            )

            # Sheet 3: Class Distribution
            unique, counts = np.unique(y, return_counts=True)
            class_dist = []
            for cls, count in zip(unique, counts):
                class_dist.append(
                    {
                        "Class_Number": cls,
                        "Class_Name": data.target_names[cls],
                        "Count": count,
                        "Percentage": f"{count/len(y)*100:.2f}%",
                        "Balance": "Balanced" if count / len(y) > 0.25 else "Imbalanced",
                    }
                )

            pd.DataFrame(class_dist).to_excel(writer, sheet_name="Class_Distribution", index=False)

            # Sheet 4: Dataset Summary
            summary = {
                "Property": [
                    "Dataset Name",
                    "Full Name",
                    "Total Samples",
                    "Number of Features",
                    "Number of Classes",
                    "Source",
                    "Year",
                    "Paper Reference",
                    "Missing Values",
                ],
                "Value": [
                    name,
                    info["description"],
                    info["samples"],
                    info["features"],
                    info["classes"],
                    info["source"],
                    "1936" if name == "iris" else "1991" if name == "wine" else "1995",
                    "Section 3.6.1 of paper",
                    "None",
                ],
            }

            pd.DataFrame(summary).to_excel(writer, sheet_name="Dataset_Summary", index=False)

        print(f"  ✓ Created with {info['samples']} samples, {info['features']} features")

    print(f"\n✓ All dataset files saved to: {results_dir}")


def run_paper_experiments(config: dict):
    """Run experiments using EXACT paper configuration."""

    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENTS (20-FOLD CV, PAPER CONFIG)")
    print("=" * 70)

    # Load the FAST experiment implementations from scripts/experiment.py by file path
    import importlib.util

    from sklearn.datasets import load_breast_cancer, load_iris, load_wine
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    exp_path = Path(__file__).parent / "experiment.py"
    if not exp_path.exists():
        raise FileNotFoundError(f"experiment.py not found at: {exp_path}")

    spec = importlib.util.spec_from_file_location("fast_experiment", str(exp_path))
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)

    datasets = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer}

    all_results = []
    n_folds = config["experiment"]["cv_folds"]
    random_state = config["experiment"]["random_state"]

    # Map paper config to FAST experiment config shape
    exp_config = {
        "ga": config["ga"],
        "tree": config["tree"],
        "fitness": {
            "mode": config["fitness"]["mode"],
            "accuracy_weight": config["fitness"]["weights"]["accuracy"],
            "interpretability_weight": config["fitness"]["weights"]["interpretability"],
            "interpretability_weights": config["fitness"]["interpretability_weights"],
        },
        "experiment": config["experiment"],
    }

    print(f"\nConfiguration:")
    print(f"  Population: {config['ga']['population_size']}")
    print(f"  Generations: {config['ga']['n_generations']}")
    print(f"  CV Folds: {n_folds}")
    print(f"  Accuracy Weight: {config['fitness']['weights']['accuracy']}")

    for dataset_name, loader in datasets.items():
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*70}")

        X, y = loader(return_X_y=True)

        # Run GA and CART using the trusted FAST implementations
        ga_res = exp.run_ga_experiment(X, y, dataset_name, exp_config, n_folds=n_folds)
        cart_res = exp.run_cart_experiment(X, y, dataset_name, exp_config, n_folds=n_folds)

        # Aggregate per-fold results into expected DataFrame rows
        for fold in range(1, n_folds + 1):
            i = fold - 1
            ga_acc = ga_res["test_acc"][i] * 100
            cart_acc = cart_res["test_acc"][i] * 100

            ga_nodes = ga_res.get("nodes", [np.nan])[i]
            cart_nodes = cart_res.get("nodes", [np.nan])[i]

            ga_depth = ga_res.get("depth", [np.nan])[i]
            cart_depth = cart_res.get("depth", [np.nan])[i]

            ga_features = ga_res.get("features", [np.nan])[i]
            cart_features = cart_res.get("features", [np.nan])[i]

            all_results.append(
                {
                    "Dataset": dataset_name,
                    "Fold": fold,
                    "Model": "GA",
                    "Accuracy": ga_acc,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1_Score": np.nan,
                    "Nodes": ga_nodes,
                    "Depth": ga_depth,
                    "Features_Used": ga_features,
                }
            )

            all_results.append(
                {
                    "Dataset": dataset_name,
                    "Fold": fold,
                    "Model": "CART",
                    "Accuracy": cart_acc,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1_Score": np.nan,
                    "Nodes": cart_nodes,
                    "Depth": cart_depth,
                    "Features_Used": cart_features,
                }
            )

        print(
            f"  Avg GA={np.mean(ga_res['test_acc'])*100:.2f}% ({np.mean(ga_res.get('nodes', [np.nan])):.1f}n), "
            f"Avg CART={np.mean(cart_res['test_acc'])*100:.2f}% ({np.mean(cart_res.get('nodes', [np.nan])):.1f}n)"
        )

    return pd.DataFrame(all_results)


def compute_paper_statistics(results_df: pd.DataFrame):
    """Compute statistics matching paper's Table 1, 2, 3."""

    print("\n" + "=" * 70)
    print("COMPUTING PAPER STATISTICS")
    print("=" * 70)

    stats_results = []

    for dataset in results_df["Dataset"].unique():
        dataset_df = results_df[results_df["Dataset"] == dataset]

        ga_acc = dataset_df[dataset_df["Model"] == "GA"]["Accuracy"].values
        cart_acc = dataset_df[dataset_df["Model"] == "CART"]["Accuracy"].values

        ga_nodes = dataset_df[dataset_df["Model"] == "GA"]["Nodes"].values
        cart_nodes = dataset_df[dataset_df["Model"] == "CART"]["Nodes"].values

        ga_depth = dataset_df[dataset_df["Model"] == "GA"]["Depth"].values
        cart_depth = dataset_df[dataset_df["Model"] == "CART"]["Depth"].values

        ga_features = dataset_df[dataset_df["Model"] == "GA"]["Features_Used"].values
        cart_features = dataset_df[dataset_df["Model"] == "CART"]["Features_Used"].values

        # Accuracy statistics
        differences = ga_acc - cart_acc
        t_stat, p_value = stats.ttest_rel(ga_acc, cart_acc)

        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(differences)

        # Tree size statistics
        size_t_stat, size_p_value = stats.ttest_rel(ga_nodes, cart_nodes)
        reduction = (cart_nodes.mean() - ga_nodes.mean()) / cart_nodes.mean() * 100

        stats_results.append(
            {
                "Dataset": dataset,
                "n_folds": len(ga_acc),
                # Table 1: Accuracy
                "GA_Accuracy_Mean": ga_acc.mean(),
                "GA_Accuracy_Std": ga_acc.std(ddof=1),
                "CART_Accuracy_Mean": cart_acc.mean(),
                "CART_Accuracy_Std": cart_acc.std(ddof=1),
                "Delta_GA_CART": mean_diff,
                # Table 2: Tree Complexity
                "GA_Nodes_Mean": ga_nodes.mean(),
                "GA_Nodes_Std": ga_nodes.std(ddof=1),
                "CART_Nodes_Mean": cart_nodes.mean(),
                "CART_Nodes_Std": cart_nodes.std(ddof=1),
                "Size_Reduction_%": reduction,
                "GA_Depth_Mean": ga_depth.mean(),
                "GA_Depth_Std": ga_depth.std(ddof=1),
                "CART_Depth_Mean": cart_depth.mean(),
                "CART_Depth_Std": cart_depth.std(ddof=1),
                # Table 3: Statistical Tests
                "t_statistic": t_stat,
                "p_value": p_value,
                "Cohen_d": cohens_d,
                # Additional analyses
                "Shapiro_Wilk_W": shapiro_stat,
                "Shapiro_Wilk_p": shapiro_p,
                "Normal_Distribution": "Yes" if shapiro_p >= 0.05 else "No",
                "Size_t_statistic": size_t_stat,
                "Size_p_value": size_p_value,
                # Feature usage (Table 4)
                "GA_Features_Mean": ga_features.mean(),
                "CART_Features_Mean": cart_features.mean(),
                "Feature_Reduction_%": (1 - ga_features.mean() / cart_features.mean()) * 100,
            }
        )

        print(f"\n{dataset.upper()}:")
        print(
            f"  Accuracy: GA={ga_acc.mean():.2f}±{ga_acc.std(ddof=1):.2f}%, "
            f"CART={cart_acc.mean():.2f}±{cart_acc.std(ddof=1):.2f}%"
        )
        print(f"  t={t_stat:.3f}, p={p_value:.3f}, Cohen's d={cohens_d:.3f}")
        print(
            f"  Tree size: GA={ga_nodes.mean():.1f} nodes, CART={cart_nodes.mean():.1f} nodes "
            f"({reduction:.1f}% reduction)"
        )

    return pd.DataFrame(stats_results)


def create_paper_tables_excel(results_df: pd.DataFrame, stats_df: pd.DataFrame):
    """Generate Excel file with tables matching the paper."""

    print("\n" + "=" * 70)
    print("CREATING PAPER-FORMAT EXCEL FILE")
    print("=" * 70)

    results_dir = Path("results/professor_consultation")
    excel_path = results_dir / "detailed_results.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        # Sheet 1: Table 1 from paper (Classification Accuracy)
        table1_data = []
        for _, row in stats_df.iterrows():
            table1_data.append(
                {
                    "Dataset": row["Dataset"].capitalize(),
                    "GA_Diverse": f"{row['GA_Accuracy_Mean']:.2f} ± {row['GA_Accuracy_Std']:.2f}%",
                    "CART": f"{row['CART_Accuracy_Mean']:.2f} ± {row['CART_Accuracy_Std']:.2f}%",
                    "Δ_GA_CART": f"{row['Delta_GA_CART']:+.2f}%",
                }
            )

        pd.DataFrame(table1_data).to_excel(writer, sheet_name="Table1_Accuracy", index=False)
        print("  ✓ Table 1: Classification Accuracy")

        # Sheet 2: Table 2 from paper (Tree Complexity)
        table2_data = []
        for _, row in stats_df.iterrows():
            table2_data.append(
                {
                    "Dataset": row["Dataset"].capitalize(),
                    "GA_Nodes": f"{row['GA_Nodes_Mean']:.1f} ± {row['GA_Nodes_Std']:.1f}",
                    "CART_Nodes": f"{row['CART_Nodes_Mean']:.1f} ± {row['CART_Nodes_Std']:.1f}",
                    "Reduction": f"{row['Size_Reduction_%']:.0f}%",
                    "GA_Depth": f"{row['GA_Depth_Mean']:.1f} ± {row['GA_Depth_Std']:.1f}",
                    "CART_Depth": f"{row['CART_Depth_Mean']:.1f} ± {row['CART_Depth_Std']:.1f}",
                }
            )

        pd.DataFrame(table2_data).to_excel(writer, sheet_name="Table2_Complexity", index=False)
        print("  ✓ Table 2: Tree Complexity")

        # Sheet 3: Table 3 from paper (Statistical Tests)
        table3_data = []
        for _, row in stats_df.iterrows():
            table3_data.append(
                {
                    "Dataset": row["Dataset"].capitalize(),
                    "t_statistic": f"{row['t_statistic']:.3f}",
                    "p_value": f"{row['p_value']:.3f}",
                    "Cohen_d": f"{row['Cohen_d']:.3f}",
                    "Interpretation": (
                        "Not significant" if row["p_value"] >= 0.05 else "Significant"
                    ),
                }
            )

        pd.DataFrame(table3_data).to_excel(writer, sheet_name="Table3_Stats", index=False)
        print("  ✓ Table 3: Statistical Tests")

        # Sheet 4: Table 4 from paper (Feature Usage)
        table4_data = []
        total_features = {"iris": 4, "wine": 13, "breast_cancer": 30}
        for _, row in stats_df.iterrows():
            dataset = row["Dataset"]
            table4_data.append(
                {
                    "Dataset": dataset.capitalize(),
                    "Total_Features": total_features[dataset],
                    "GA_Features": f"{row['GA_Features_Mean']:.1f}",
                    "CART_Features": f"{row['CART_Features_Mean']:.1f}",
                    "Feature_Reduction": f"{row['Feature_Reduction_%']:.0f}%",
                }
            )

        pd.DataFrame(table4_data).to_excel(writer, sheet_name="Table4_Features", index=False)
        print("  ✓ Table 4: Feature Usage")

        # Sheet 5: All 20-fold raw results
        results_df.to_excel(writer, sheet_name="All_20_Folds", index=False)
        print("  ✓ Sheet: All 20-fold results")

        # Sheet 6: Complete statistics
        stats_df.to_excel(writer, sheet_name="Complete_Statistics", index=False)
        print("  ✓ Sheet: Complete statistics")

        # Sheet 7: Normality tests
        normality_data = []
        for _, row in stats_df.iterrows():
            normality_data.append(
                {
                    "Dataset": row["Dataset"].capitalize(),
                    "Shapiro_Wilk_W": f"{row['Shapiro_Wilk_W']:.4f}",
                    "Shapiro_Wilk_p": f"{row['Shapiro_Wilk_p']:.4f}",
                    "Normal_Distribution": row["Normal_Distribution"],
                    "Test_Recommendation": (
                        "Paired t-test valid"
                        if row["Normal_Distribution"] == "Yes"
                        else "Consider Wilcoxon signed-rank"
                    ),
                }
            )

        pd.DataFrame(normality_data).to_excel(writer, sheet_name="Normality_Tests", index=False)
        print("  ✓ Sheet: Normality tests")

        # Sheet 8: Paper claims verification
        claims_data = []
        for _, row in stats_df.iterrows():
            dataset = row["Dataset"]
            reduction = row["Size_Reduction_%"]
            p_val = row["p_value"]

            # Check if matches paper claims
            if dataset == "iris":
                claim_match = 20 <= reduction <= 30
            elif dataset == "wine":
                claim_match = 25 <= reduction <= 30
            elif dataset == "breast_cancer":
                claim_match = 70 <= reduction <= 80
            else:
                claim_match = False

            claims_data.append(
                {
                    "Dataset": dataset.capitalize(),
                    "Paper_Claim_Reduction": (
                        "24%" if dataset == "iris" else "27%" if dataset == "wine" else "77%"
                    ),
                    "Actual_Reduction": f"{reduction:.1f}%",
                    "Matches_Paper": "Yes" if claim_match else "No",
                    "Paper_Claim_p_value": "> 0.05",
                    "Actual_p_value": f"{p_val:.3f}",
                    "Statistical_Equivalence": "Confirmed" if p_val >= 0.05 else "Not confirmed",
                }
            )

        pd.DataFrame(claims_data).to_excel(writer, sheet_name="Paper_Claims_Check", index=False)
        print("  ✓ Sheet: Paper claims verification")

    print(f"\n✓ Excel file created: {excel_path}")


def create_consultation_package(results_df: pd.DataFrame, stats_df: pd.DataFrame):
    """Create comprehensive consultation package in Arabic/English."""

    results_dir = Path("results/professor_consultation")
    excel_path = results_dir / "statistical_consultation_package.xlsx"

    print("\n" + "=" * 70)
    print("CREATING CONSULTATION PACKAGE")
    print("=" * 70)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        # Sheet 1: Executive Summary
        summary_data = []
        for _, row in stats_df.iterrows():
            dataset = row["Dataset"]
            p_val = row["p_value"]
            reduction = row["Size_Reduction_%"]

            if p_val >= 0.05:
                result_en = "Statistically equivalent accuracy"
                result_ar = "دقة متكافئة إحصائياً"
            else:
                result_en = "Statistically significant difference"
                result_ar = "فرق ذو دلالة إحصائية"

            summary_data.append(
                {
                    "Dataset": dataset.capitalize(),
                    "GA_Accuracy": f"{row['GA_Accuracy_Mean']:.2f}% ± {row['GA_Accuracy_Std']:.2f}%",
                    "CART_Accuracy": f"{row['CART_Accuracy_Mean']:.2f}% ± {row['CART_Accuracy_Std']:.2f}%",
                    "p_value": f"{p_val:.3f}",
                    "Result_English": result_en,
                    "Result_Arabic": result_ar,
                    "Tree_Size_Reduction": f"{reduction:.1f}%",
                    "Paper_Section": "Table 1, 2, 3",
                }
            )

        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Executive_Summary", index=False)
        print("  ✓ Executive Summary")

        # Sheet 2: Questions for Professor (from Arabic doc)
        questions = [
            {
                "Q#": 1,
                "Question_EN": "Is 20-fold CV appropriate for sample sizes 150-569?",
                "Question_AR": "هل 20-fold CV مناسب لأحجام العينات؟",
                "Paper_Section": "3.6.2",
            },
        ]

        pd.DataFrame(questions).to_excel(writer, sheet_name="Questions_للبروفيسور", index=False)
        print("  ✓ Questions for Professor")

        # Sheet 3: Statistical tests summary
        stats_df.to_excel(writer, sheet_name="Complete_Statistics", index=False)
        print("  ✓ Complete Statistics")

        # Sheet 4: Comparison with paper claims
        comparison_data = []
        paper_claims = {
            "iris": {
                "acc_ga": 93.84,
                "acc_cart": 92.41,
                "nodes_ga": 12.5,
                "nodes_cart": 16.4,
                "reduction": 24,
            },
            "wine": {
                "acc_ga": 89.93,
                "acc_cart": 87.22,
                "nodes_ga": 15.1,
                "nodes_cart": 20.7,
                "reduction": 27,
            },
            "breast_cancer": {
                "acc_ga": 92.10,
                "acc_cart": 91.57,
                "nodes_ga": 8.0,
                "nodes_cart": 35.5,
                "reduction": 77,
            },
        }

        for _, row in stats_df.iterrows():
            dataset = row["Dataset"]
            if dataset in paper_claims:
                pc = paper_claims[dataset]
                comparison_data.append(
                    {
                        "Dataset": dataset.capitalize(),
                        "Metric": "GA Accuracy",
                        "Paper_Value": f"{pc['acc_ga']:.2f}%",
                        "Our_Value": f"{row['GA_Accuracy_Mean']:.2f}%",
                        "Difference": f"{row['GA_Accuracy_Mean'] - pc['acc_ga']:.2f}%",
                    }
                )
                comparison_data.append(
                    {
                        "Dataset": dataset.capitalize(),
                        "Metric": "CART Accuracy",
                        "Paper_Value": f"{pc['acc_cart']:.2f}%",
                        "Our_Value": f"{row['CART_Accuracy_Mean']:.2f}%",
                        "Difference": f"{row['CART_Accuracy_Mean'] - pc['acc_cart']:.2f}%",
                    }
                )
                comparison_data.append(
                    {
                        "Dataset": dataset.capitalize(),
                        "Metric": "Tree Size Reduction",
                        "Paper_Value": f"{pc['reduction']}%",
                        "Our_Value": f"{row['Size_Reduction_%']:.0f}%",
                        "Difference": f"{row['Size_Reduction_%'] - pc['reduction']:.0f}%",
                    }
                )

        pd.DataFrame(comparison_data).to_excel(
            writer, sheet_name="Paper_vs_Our_Results", index=False
        )
        print("  ✓ Paper comparison")

    print(f"\n✓ Consultation package created: {excel_path}")


def create_arabic_summary():
    """Create Arabic summary matching the consultation request."""

    results_dir = Path("results/professor_consultation")
    summary_path = results_dir / "consultation_request_summary_AR.txt"

    summary_text = """
╔══════════════════════════════════════════════════════════════════════╗
║              طلب استشارة إحصائية - ملخص النتائج                    ║
╚══════════════════════════════════════════════════════════════════════╝

"""

    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"✓ Arabic summary: {summary_path}")


def main():
    """Main pipeline using paper-exact configuration."""

    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  STATISTICAL CONSULTATION (PAPER-EXACT CONFIG)".center(68) + "█")
    print("█" + "  Using configuration from published paper".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    # Load paper configuration
    config = load_paper_config()

    # Verify configuration
    verify_paper_configuration(config)

    # Create dataset files
    create_dataset_excel_files()

    # Run experiments with paper config
    results_df = run_paper_experiments(config)

    # Compute statistics
    stats_df = compute_paper_statistics(results_df)

    # Create Excel files
    create_paper_tables_excel(results_df, stats_df)
    create_consultation_package(results_df, stats_df)

    # Create Arabic summary
    create_arabic_summary()

    # Final summary
    results_dir = Path("results/professor_consultation")

    print("\n" + "=" * 70)
    print("✅ ALL FILES GENERATED")
    print("=" * 70)

    print("\n📁 Files:")
    for file in sorted(results_dir.glob("*")):
        size_kb = file.stat().st_size / 1024
        print(f"  • {file.name:50s} ({size_kb:6.1f} KB)")

    print("\n" + "=" * 70)
    print("📧 SEND TO PROFESSOR:")
    print("=" * 70)
    print("  1. All 3 dataset .xlsx files")
    print("  2. detailed_results.xlsx (Tables 1-4 from paper)")
    print("  3. statistical_consultation_package.xlsx")
    print("  4. consultation_request_summary_AR.txt")
    print("  5. Your research paper PDF")

    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT:")
    print("=" * 70)
    print("  Results should match paper claims:")
    print("  • Iris: ~24% reduction, p > 0.05")
    print("  • Wine: ~27% reduction, p > 0.05")
    print("  • Breast Cancer: ~77% reduction, p > 0.05")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
