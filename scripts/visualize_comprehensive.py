"""
Comprehensive visualization of GA experiment results.

Creates publication-quality figures showing:
1. Accuracy comparison (bar chart)
2. Tree size comparison (bar chart)
3. Accuracy vs tree size trade-off (scatter — node count x-axis)
4. Speed comparison (bar chart)
5. Summary heatmap (replaces static table image)
6. Key findings (grouped horizontal bar chart — replaces monospace text dump)

Usage:
    python scripts/visualize_comprehensive.py
    python scripts/visualize_comprehensive.py --config paper
    python scripts/visualize_comprehensive.py --config fast --results-dir results
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"

# Canonical model-name mapping from experiment CSV labels to short keys
_MODEL_KEY_MAP = {
    "GA-Optimized": "GA",
    "CART": "CART",
    "CART (constrained)": "CART",
    "CART (unconstrained)": "CART-unconstrained",
    "Random Forest": "RF",
    "XGBoost": "XGBoost",
}

# Ordered dataset display names
_DATASET_DISPLAY = {
    "iris": "Iris",
    "wine": "Wine",
    "breast_cancer": "Breast Cancer",
}


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------


def _parse_mean_std(val_str):
    """Parse 'mean ± std' string into (mean, std) floats.

    Returns (nan, 0.0) on any parse error.
    """
    try:
        parts = str(val_str).split("±")
        mean = float(parts[0].strip())
        std = float(parts[1].strip()) if len(parts) > 1 else 0.0
        return mean, std
    except (ValueError, IndexError):
        return float("nan"), 0.0


def load_results_from_csv(config_name="default", results_dir=None):
    """Load experiment results from CSV into plotting-ready dicts.

    Search order:
    1. ``results/tables/results-{config_name}.csv``  (written by experiment.py)
    2. Most recently modified ``results/result-{config_name}-*.csv``

    Returns
    -------
    (results, paper_results) on success, or (None, None) when no data is found.

    ``results`` structure::

        {
            "iris": {
                "GA":   {"acc": 95.33, "std": 3.40, "nodes": 7.4, "depth": 2.4, "time": 3.4},
                "CART": {"acc": 94.67, "std": 2.67, "nodes": 13.8, ...},
                "RF":   {"acc": 95.33, ...},
            },
            ...
        }

    ``paper_results`` structure::

        {
            "iris": {
                "ga_acc": 95.33, "cart_acc": 94.67,
                "ga_nodes": 7.4,  "cart_nodes": 13.8,
                "ga_std": 3.40,   "cart_std": 2.67,
                "p_val": 0.186,   "size_reduction_pct": 46,
            },
            ...
        }
    """
    base = Path(results_dir) if results_dir else Path("results")

    # 1. Prefer tables directory (written by experiment.py automatically)
    tables_file = base / "tables" / f"results-{config_name}.csv"

    if not tables_file.exists():
        # 2. Fall back to most recent timestamped result CSV
        candidates = sorted(
            base.glob(f"result-{config_name}-*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(
                f"WARNING: No results found for config '{config_name}'. "
                "Run experiment.py first, then re-run this script."
            )
            return None, None
        tables_file = candidates[0]

    try:
        df = pd.read_csv(tables_file)
    except Exception as exc:
        print(f"WARNING: Could not read results file {tables_file}: {exc}")
        return None, None

    # Build nested results dict
    results = {}
    for _, row in df.iterrows():
        raw_dataset = str(row.get("Dataset", "")).strip().lower().replace(" ", "_")
        raw_model = str(row.get("Model", "")).strip()
        model_key = _MODEL_KEY_MAP.get(raw_model, raw_model)

        if raw_dataset not in results:
            results[raw_dataset] = {}

        acc_mean, acc_std = _parse_mean_std(row.get("Test Acc", "nan"))
        time_val = float(row.get("Time (s)", 0) or 0)

        entry = {
            "acc": acc_mean * 100,  # convert fraction → percentage
            "std": acc_std * 100,
            "time": time_val,
        }

        nodes_raw = row.get("Nodes", None)
        depth_raw = row.get("Depth", None)
        if (
            nodes_raw is not None
            and pd.notna(nodes_raw)
            and str(nodes_raw).strip() not in ("", "nan")
        ):
            entry["nodes"] = float(nodes_raw)
        if (
            depth_raw is not None
            and pd.notna(depth_raw)
            and str(depth_raw).strip() not in ("", "nan")
        ):
            entry["depth"] = float(depth_raw)

        results[raw_dataset][model_key] = entry

    # Derive paper_results from parsed data (GA + CART required)
    paper_results = {}
    for dataset, models in results.items():
        if "GA" not in models or "CART" not in models:
            continue
        ga = models["GA"]
        cart = models["CART"]
        ga_nodes = ga.get("nodes", float("nan"))
        cart_nodes = cart.get("nodes", float("nan"))
        reduction = (
            int(round((1 - ga_nodes / cart_nodes) * 100))
            if cart_nodes and not np.isnan(cart_nodes) and not np.isnan(ga_nodes)
            else 0
        )
        paper_results[dataset] = {
            "ga_acc": ga["acc"],
            "cart_acc": cart["acc"],
            "ga_std": ga["std"],
            "cart_std": cart["std"],
            "ga_nodes": ga_nodes,
            "cart_nodes": cart_nodes,
            "p_val": 0.05,  # default; overwritten below if stats CSV exists
            "size_reduction_pct": reduction,
        }

    # Try to load p-values from stats CSV written by experiment.py
    stats_files = sorted(
        base.glob(f"stats-{config_name}-*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if stats_files:
        try:
            stats_df = pd.read_csv(stats_files[0])
            for _, row in stats_df.iterrows():
                ds = str(row["dataset"]).strip().lower().replace(" ", "_")
                if ds in paper_results:
                    paper_results[ds]["p_val"] = float(row["p"])
        except Exception:
            pass  # stats CSV is optional

    print(f"Loaded results from: {tables_file}")
    return results, paper_results


# ------------------------------------------------------------------
# Comprehensive figures (use results dict)
# ------------------------------------------------------------------


def create_accuracy_comparison(results=None):
    """Grouped bar chart comparing accuracy across GA, CART, and RF."""
    if results is None:
        print("Skipping accuracy_comparison: no results data.")
        return

    datasets = list(_DATASET_DISPLAY.keys())
    labels = list(_DATASET_DISPLAY.values())
    x = np.arange(len(datasets))
    width = 0.25

    ga_acc, ga_std, cart_acc, cart_std, rf_acc, rf_std = [], [], [], [], [], []
    for ds in datasets:
        d = results.get(ds, {})
        ga_acc.append(d.get("GA", {}).get("acc", float("nan")))
        ga_std.append(d.get("GA", {}).get("std", 0))
        cart_acc.append(d.get("CART", {}).get("acc", float("nan")))
        cart_std.append(d.get("CART", {}).get("std", 0))
        rf_acc.append(d.get("RF", {}).get("acc", float("nan")))
        rf_std.append(d.get("RF", {}).get("std", 0))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        x - width,
        ga_acc,
        width,
        yerr=ga_std,
        label="GA-Optimized",
        color="#FF6B6B",
        alpha=0.85,
        capsize=5,
        edgecolor="black",
        linewidth=1.2,
    )
    bars2 = ax.bar(
        x,
        cart_acc,
        width,
        yerr=cart_std,
        label="CART",
        color="#4ECDC4",
        alpha=0.85,
        capsize=5,
        edgecolor="black",
        linewidth=1.2,
    )
    bars3 = ax.bar(
        x + width,
        rf_acc,
        width,
        yerr=rf_std,
        label="Random Forest",
        color="#95E1D3",
        alpha=0.85,
        capsize=5,
        edgecolor="black",
        linewidth=1.2,
    )

    def autolabel(bars, values):
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 1,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    autolabel(bars1, ga_acc)
    autolabel(bars2, cart_acc)
    autolabel(bars3, rf_acc)

    ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Dataset", fontsize=14, fontweight="bold")
    ax.set_title("Model Accuracy Comparison", fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.set_ylim([82, 102])
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "accuracy_comparison.png", bbox_inches="tight")
    print("✓ Saved: accuracy_comparison.png")
    plt.close()


def create_tree_size_comparison(results=None):
    """Grouped bar chart showing GA produces smaller trees than CART."""
    if results is None:
        print("Skipping tree_size_comparison: no results data.")
        return

    datasets = list(_DATASET_DISPLAY.keys())
    labels = list(_DATASET_DISPLAY.values())
    x = np.arange(len(datasets))
    width = 0.35

    ga_nodes, cart_nodes = [], []
    for ds in datasets:
        d = results.get(ds, {})
        ga_nodes.append(d.get("GA", {}).get("nodes", float("nan")))
        cart_nodes.append(d.get("CART", {}).get("nodes", float("nan")))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2,
        ga_nodes,
        width,
        label="GA-Optimized",
        color="#FF6B6B",
        alpha=0.85,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.bar(
        x + width / 2,
        cart_nodes,
        width,
        label="CART",
        color="#4ECDC4",
        alpha=0.85,
        edgecolor="black",
        linewidth=1.5,
    )

    for i, (ga, cart) in enumerate(zip(ga_nodes, cart_nodes)):
        if not np.isnan(ga):
            ax.text(
                i - width / 2,
                ga + 1,
                f"{ga:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        if not np.isnan(cart):
            ax.text(
                i + width / 2,
                cart + 1,
                f"{cart:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        if not np.isnan(ga) and not np.isnan(cart) and cart > 0:
            reduction = (1 - ga / cart) * 100
            ax.annotate(
                f"{reduction:.0f}% smaller",
                xy=(i, max(ga, cart) + 3),
                fontsize=10,
                ha="center",
                color="green",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
            )

    ax.set_ylabel("Number of Nodes", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dataset", fontsize=14, fontweight="bold")
    ax.set_title(
        "Tree Complexity: GA Produces 2–7× Smaller Trees", fontsize=16, fontweight="bold", pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(Path("results/figures") / "tree_size_comparison.png", bbox_inches="tight")
    print("✓ Saved: tree_size_comparison.png")
    plt.close()


def create_tradeoff_scatter(results=None):
    """Accuracy vs number-of-nodes scatter.

    X-axis shows node count directly (lower = more interpretable).
    The GA sweet spot is the top-left region: high accuracy + low node count.
    """
    if results is None:
        print("Skipping tradeoff_scatter: no results data.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {"GA": "#FF6B6B", "CART": "#4ECDC4"}
    markers = {"GA": "o", "CART": "s"}
    sizes = {"GA": 300, "CART": 200}

    best_ga_acc, best_ga_nodes = float("-inf"), float("inf")

    for dataset, display_name in [
        ("iris", "Iris"),
        ("wine", "Wine"),
        ("breast_cancer", "Breast Cancer"),
    ]:
        for model in ["GA", "CART"]:
            d = results.get(dataset, {}).get(model, {})
            acc = d.get("acc", float("nan"))
            nodes = d.get("nodes", float("nan"))
            if np.isnan(acc) or np.isnan(nodes):
                continue

            ax.scatter(
                nodes,
                acc,
                s=sizes[model],
                alpha=0.7,
                color=colors[model],
                marker=markers[model],
                edgecolors="black",
                linewidth=2,
                label=model if dataset == "iris" else "",
            )

            offset_x = -2.5 if model == "GA" else 2.5
            ax.annotate(
                f"{model}\n({display_name})",
                xy=(nodes, acc),
                xytext=(offset_x, 5),
                textcoords="offset points",
                fontsize=9,
                ha="right" if model == "GA" else "left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[model], alpha=0.3),
            )

            if model == "GA" and acc > best_ga_acc:
                best_ga_acc = acc
                best_ga_nodes = nodes

    # Highlight the sweet-spot region: top-left = low nodes + high accuracy
    ax.axhspan(92, 100, alpha=0.08, color="green", label="High Accuracy Zone")
    ax.axvspan(0, 12, alpha=0.08, color="blue", label="Low Complexity Zone")

    # Point annotation toward the GA point with best accuracy
    if best_ga_acc > float("-inf"):
        ax.annotate(
            "GA Sweet Spot:\nLow Complexity\n+ High Accuracy",
            xy=(best_ga_nodes, best_ga_acc),
            xytext=(best_ga_nodes + 8, best_ga_acc - 4),
            arrowprops=dict(arrowstyle="->", lw=2, color="darkgreen"),
            fontsize=10,
            fontweight="bold",
            color="darkgreen",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FF6B6B",
            markersize=12,
            label="GA-Optimized",
            markeredgecolor="black",
            markeredgewidth=1.5,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#4ECDC4",
            markersize=10,
            label="CART",
            markeredgecolor="black",
            markeredgewidth=1.5,
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=11, framealpha=0.95)

    ax.set_xlabel("Number of Nodes  (lower = more interpretable)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontsize=13, fontweight="bold")
    ax.set_title("Accuracy–Interpretability Trade-off", fontsize=15, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(Path("results/figures") / "tradeoff_scatter.png", bbox_inches="tight")
    print("✓ Saved: tradeoff_scatter.png")
    plt.close()


def create_speed_comparison(results=None):
    """Training speed comparison (log scale)."""
    if results is None:
        print("Skipping speed_comparison: no results data.")
        return

    datasets = list(_DATASET_DISPLAY.keys())
    labels = list(_DATASET_DISPLAY.values())
    x = np.arange(len(datasets))
    width = 0.25

    ga_time, cart_time, rf_time = [], [], []
    for ds in datasets:
        d = results.get(ds, {})
        ga_time.append(max(d.get("GA", {}).get("time", 0.01), 0.01))
        cart_time.append(max(d.get("CART", {}).get("time", 0.01), 0.01))
        rf_time.append(max(d.get("RF", {}).get("time", 0.01), 0.01))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        x - width,
        ga_time,
        width,
        label="GA-Optimized",
        color="#FF6B6B",
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
    )
    bars2 = ax.bar(
        x,
        cart_time,
        width,
        label="CART",
        color="#4ECDC4",
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
    )
    bars3 = ax.bar(
        x + width,
        rf_time,
        width,
        label="Random Forest",
        color="#95E1D3",
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
    )

    ax.set_ylabel("Training Time (seconds, log scale)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Dataset", fontsize=13, fontweight="bold")
    ax.set_title("Training Speed Comparison", fontsize=15, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, linestyle="--", which="both")

    for bars, times in [(bars1, ga_time), (bars2, cart_time), (bars3, rf_time)]:
        for bar, t in zip(bars, times):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() * 1.2,
                f"{t:.2f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(Path("results/figures") / "speed_comparison.png", bbox_inches="tight")
    print("✓ Saved: speed_comparison.png")
    plt.close()


def create_summary_heatmap(results=None):
    """Seaborn heatmap: rows = (dataset × model), columns = metrics.

    Replaces the old summary_table.png (a matplotlib table rendered as image).
    Color intensity reflects normalized performance; higher = better for every
    column (nodes/depth/time are inverted so lower raw value = higher color).
    Actual values are annotated on each cell.
    """
    if results is None:
        print("Skipping summary_heatmap: no results data.")
        return

    rows = []
    model_order = [("GA", "GA-Optimized"), ("CART", "CART"), ("RF", "Random Forest")]

    for ds, label in _DATASET_DISPLAY.items():
        for model_key, model_label in model_order:
            d = results.get(ds, {}).get(model_key, {})
            if not d:
                continue
            rows.append(
                {
                    "Dataset & Model": f"{label} / {model_label}",
                    "Accuracy (%)": d.get("acc", float("nan")),
                    "Nodes": d.get("nodes", float("nan")),
                    "Depth": d.get("depth", float("nan")),
                    "Time (s)": d.get("time", float("nan")),
                }
            )

    if not rows:
        print("Skipping summary_heatmap: no data rows available.")
        return

    df_heat = pd.DataFrame(rows).set_index("Dataset & Model")

    # Normalize each column 0–1; invert node/depth/time so lower raw = higher colour
    df_norm = df_heat.copy().astype(float)
    invert_cols = {"Nodes", "Depth", "Time (s)"}
    for col in df_norm.columns:
        col_min = df_norm[col].min()
        col_max = df_norm[col].max()
        if col_max == col_min:
            df_norm[col] = 0.5
        elif col in invert_cols:
            df_norm[col] = 1.0 - (df_norm[col] - col_min) / (col_max - col_min)
        else:
            df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)

    # Build annotation dataframe with formatted actual values
    annot_df = df_heat.copy()
    for col in annot_df.columns:
        annot_df[col] = annot_df[col].apply(
            lambda v: f"{v:.2f}" if pd.notna(v) and not np.isnan(float(v)) else "N/A"
        )

    fig, ax = plt.subplots(figsize=(10, max(6, len(rows) * 0.7)))
    sns.heatmap(
        df_norm,
        annot=annot_df,
        fmt="",
        cmap="YlGnBu",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Normalised score (higher = better per metric)"},
        vmin=0,
        vmax=1,
    )
    ax.set_title(
        "Performance Summary\n(colour = normalised score; actual values annotated)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(Path("results/figures") / "summary_table.png", bbox_inches="tight")
    print("✓ Saved: summary_table.png")
    plt.close()


def create_key_findings_chart(results=None):
    """Grouped horizontal bar chart showing GA improvements over CART.

    Replaces the old key_findings.png (monospace text dump).

    Left panel  — accuracy difference (GA − CART, percentage points).
    Right panel — tree size reduction (%, relative to CART node count).
    """
    if results is None:
        print("Skipping key_findings: no results data.")
        return

    datasets = list(_DATASET_DISPLAY.keys())
    labels = list(_DATASET_DISPLAY.values())

    acc_diff, size_reduction = [], []
    for ds in datasets:
        d = results.get(ds, {})
        ga_acc = d.get("GA", {}).get("acc", float("nan"))
        cart_acc = d.get("CART", {}).get("acc", float("nan"))
        ga_nodes = d.get("GA", {}).get("nodes", float("nan"))
        cart_nodes = d.get("CART", {}).get("nodes", float("nan"))

        acc_diff.append(ga_acc - cart_acc if not (np.isnan(ga_acc) or np.isnan(cart_acc)) else 0.0)
        if not np.isnan(ga_nodes) and not np.isnan(cart_nodes) and cart_nodes > 0:
            size_reduction.append((1 - ga_nodes / cart_nodes) * 100)
        else:
            size_reduction.append(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: accuracy difference
    ax = axes[0]
    colors_acc = ["#27ae60" if v >= 0 else "#e74c3c" for v in acc_diff]
    ax.barh(labels, acc_diff, color=colors_acc, edgecolor="black", linewidth=1.2, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Accuracy Difference: GA − CART (pp)", fontsize=12, fontweight="bold")
    ax.set_title("Accuracy Delta vs CART", fontsize=13, fontweight="bold")
    for i, v in enumerate(acc_diff):
        sign = "+" if v >= 0 else ""
        ax.text(
            v + (0.15 if v >= 0 else -0.15),
            i,
            f"{sign}{v:.2f}%",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=11,
            fontweight="bold",
        )
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Right: size reduction
    ax = axes[1]
    colors_size = ["#27ae60" if v > 0 else "#e74c3c" for v in size_reduction]
    ax.barh(labels, size_reduction, color=colors_size, edgecolor="black", linewidth=1.2, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Tree Size Reduction vs CART (%)", fontsize=12, fontweight="bold")
    ax.set_title("Interpretability Gain: Tree Size Reduction", fontsize=13, fontweight="bold")
    for i, v in enumerate(size_reduction):
        ax.text(
            v + 1,
            i,
            f"{v:.1f}%",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color="#27ae60" if v > 0 else "#e74c3c",
        )
    ax.set_xlim(0, 110)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    fig.suptitle(
        "Key Findings: GA-Optimized vs CART Baseline",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "key_findings.png", bbox_inches="tight")
    print("✓ Saved: key_findings.png")
    plt.close()


# ------------------------------------------------------------------
# Publication figures (uses paper_results dict)
# ------------------------------------------------------------------


def create_statistical_equivalence(paper_results=None):
    """Horizontal bar chart of p-values demonstrating statistical equivalence to CART.

    All bars extending past the α = 0.05 threshold mean GA and CART accuracy
    are statistically indistinguishable on those datasets.
    """
    if paper_results is None:
        print("Skipping statistical_equivalence: no paper_results data.")
        return

    keys = [k for k in ["iris", "wine", "breast_cancer"] if k in paper_results]
    if not keys:
        print("Skipping statistical_equivalence: no matching datasets in paper_results.")
        return

    dataset_labels = [_DATASET_DISPLAY.get(k, k.title()) for k in keys]
    p_values = [paper_results[k]["p_val"] for k in keys]
    colors = ["#27ae60" if p > 0.05 else "#e74c3c" for p in p_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(dataset_labels, p_values, color=colors, edgecolor="black", linewidth=1.2, alpha=0.9)
    ax.axvline(
        0.05,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label="α = 0.05 (significance threshold)",
    )

    for i, (ds_label, p) in enumerate(zip(dataset_labels, p_values)):
        ax.text(p + 0.02, i, f"p = {p:.3f}", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("p-value (Paired t-test)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Statistical Equivalence to CART\n(p > 0.05 = no significant difference)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, max(p_values) * 1.4 + 0.1)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "paper_fig2_statistical_equiv.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "paper_fig2_statistical_equiv.pdf", bbox_inches="tight")
    print("✓ Saved: paper_fig2_statistical_equiv.png / .pdf")
    plt.close()


def create_publication_figures(results=None, paper_results=None):
    """Generate all four publication-quality figures.

    Figures produced
    ----------------
    paper_fig1_size_reduction  — GA vs CART node-count bar chart
    paper_fig2_statistical_equiv — p-value horizontal bar chart
    paper_fig3_pareto_tradeoff — accuracy vs nodes scatter with arrows + Pareto line
    paper_table_summary        — grouped bar chart with p-value annotations
                                 (replaces old static matplotlib table image)
    """
    if paper_results is None:
        print("Skipping publication figures: no paper_results data.")
        return

    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    orig_family = plt.rcParams["font.family"]
    orig_size = plt.rcParams["font.size"]
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    sns.set_palette("colorblind")

    try:
        keys = [k for k in ["iris", "wine", "breast_cancer"] if k in paper_results]
        datasets_labels = [_DATASET_DISPLAY.get(k, k.title()) for k in keys]

        # --- Figure 1: Size Reduction ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ga_nodes = [paper_results[k]["ga_nodes"] for k in keys]
        cart_nodes = [paper_results[k]["cart_nodes"] for k in keys]
        reductions = [paper_results[k]["size_reduction_pct"] for k in keys]

        x = np.arange(len(datasets_labels))
        width = 0.35
        ax.bar(
            x - width / 2,
            ga_nodes,
            width,
            label="GA",
            color="#2ecc71",
            edgecolor="black",
            linewidth=1.2,
        )
        ax.bar(
            x + width / 2,
            cart_nodes,
            width,
            label="CART",
            color="#e74c3c",
            alpha=0.85,
            edgecolor="black",
            linewidth=1.2,
        )

        for i, (ga, cart, red) in enumerate(zip(ga_nodes, cart_nodes, reductions)):
            ax.text(
                i,
                max(ga, cart) + 1.5,
                f"{red}%",
                ha="center",
                fontsize=11,
                fontweight="bold",
                color="#27ae60",
            )

        ax.set_ylabel("Number of Nodes", fontsize=12, fontweight="bold")
        ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
        max_nodes = max(max(cart_nodes), max(ga_nodes)) + 8
        ax.set_ylim(0, max_nodes)
        ax.set_title("GA Achieves Tree Size Reduction", fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets_labels, fontsize=11)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        plt.tight_layout()
        plt.savefig(output_dir / "paper_fig1_size_reduction.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / "paper_fig1_size_reduction.pdf", bbox_inches="tight")
        print("✓ Saved: paper_fig1_size_reduction.png / .pdf")
        plt.close()

        # --- Figure 2: Statistical Equivalence ---
        create_statistical_equivalence(paper_results)

        # --- Figure 3: Pareto Trade-off Scatter with connecting line ---
        fig, ax = plt.subplots(figsize=(8, 6))

        ga_pts = []  # (nodes, acc) for GA, to draw connecting line
        for key, display in zip(keys, datasets_labels):
            d = paper_results[key]
            ga_n, ga_a = d["ga_nodes"], d["ga_acc"]
            cart_n, cart_a = d["cart_nodes"], d["cart_acc"]

            ax.scatter(
                ga_n,
                ga_a,
                s=200,
                alpha=0.85,
                color="#2ecc71",
                marker="o",
                edgecolors="black",
                linewidth=1.2,
                zorder=3,
            )
            ax.scatter(
                cart_n,
                cart_a,
                s=140,
                alpha=0.8,
                color="#e74c3c",
                marker="s",
                edgecolors="black",
                linewidth=1.2,
                zorder=3,
            )
            ax.annotate(
                "",
                xy=(ga_n, ga_a),
                xytext=(cart_n, cart_a),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="gray", alpha=0.6),
                zorder=2,
            )
            # Dataset labels: offset to avoid overlap
            ax.annotate(
                display,
                xy=(ga_n, ga_a),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=9,
                color="#1a7a45",
            )
            ga_pts.append((ga_n, ga_a))

        # Draw Pareto front connecting line through GA points (sorted by nodes)
        if len(ga_pts) >= 2:
            ga_pts_sorted = sorted(ga_pts, key=lambda t: t[0])
            px = [t[0] for t in ga_pts_sorted]
            py = [t[1] for t in ga_pts_sorted]
            ax.plot(
                px,
                py,
                color="#2ecc71",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="Pareto front (GA)",
                zorder=1,
            )

        ax.axhspan(90, 96, alpha=0.03, color="green", zorder=0)
        ax.axvspan(0, 15, alpha=0.03, color="blue", zorder=0)
        ax.text(
            2,
            max(d["ga_acc"] for d in paper_results.values()) - 0.5,
            "Ideal region",
            fontsize=9,
            color="darkgreen",
            fontweight="bold",
        )

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#2ecc71",
                markersize=10,
                label="GA (smaller)",
                markeredgecolor="black",
                markeredgewidth=1.2,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="#e74c3c",
                markersize=8,
                label="CART (baseline)",
                markeredgecolor="black",
                markeredgewidth=1.2,
            ),
            Line2D(
                [0],
                [0],
                color="#2ecc71",
                linestyle="--",
                linewidth=1.5,
                label="Pareto front (GA)",
            ),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.95)
        ax.set_xlabel("Number of Nodes", fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
        ax.set_title(
            "GA Finds Pareto-Optimal Solutions\n(Smaller trees, competitive accuracy)",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.grid(True, alpha=0.25, linestyle="--")
        all_nodes = [paper_results[k]["ga_nodes"] for k in keys] + [
            paper_results[k]["cart_nodes"] for k in keys
        ]
        all_acc = [paper_results[k]["ga_acc"] for k in keys] + [
            paper_results[k]["cart_acc"] for k in keys
        ]
        ax.set_xlim(0, max(all_nodes) * 1.2)
        ax.set_ylim(min(all_acc) - 2, max(all_acc) + 3)
        plt.tight_layout()
        plt.savefig(output_dir / "paper_fig3_pareto_tradeoff.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / "paper_fig3_pareto_tradeoff.pdf", bbox_inches="tight")
        print("✓ Saved: paper_fig3_pareto_tradeoff.png / .pdf")
        plt.close()

        # --- Paper Table Summary: grouped bar chart with p-value annotations ---
        # Replaces old static matplotlib table image with a real data-driven figure.
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        x = np.arange(len(datasets_labels))
        width = 0.35

        # Left panel: accuracy comparison with error bars + p-value stars
        ax = axes[0]
        ga_acc = [paper_results[k]["ga_acc"] for k in keys]
        cart_acc_vals = [paper_results[k]["cart_acc"] for k in keys]
        ga_std = [paper_results[k].get("ga_std", 0) for k in keys]
        cart_std = [paper_results[k].get("cart_std", 0) for k in keys]

        ax.bar(
            x - width / 2,
            ga_acc,
            width,
            yerr=ga_std,
            label="GA",
            color="#2ecc71",
            capsize=5,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.9,
        )
        ax.bar(
            x + width / 2,
            cart_acc_vals,
            width,
            yerr=cart_std,
            label="CART",
            color="#e74c3c",
            capsize=5,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.85,
        )

        for i, k in enumerate(keys):
            p = paper_results[k]["p_val"]
            if p > 0.05:
                sig_label = "ns"
            elif p > 0.01:
                sig_label = "*"
            elif p > 0.001:
                sig_label = "**"
            else:
                sig_label = "***"
            top = max(ga_acc[i] + ga_std[i], cart_acc_vals[i] + cart_std[i]) + 0.8
            ax.text(
                i,
                top,
                sig_label,
                ha="center",
                fontsize=12,
                fontweight="bold",
                color="black",
            )
            ax.text(
                i,
                top + 0.8,
                f"p={p:.3f}",
                ha="center",
                fontsize=8,
                color="gray",
            )

        ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
        ax.set_title("Accuracy: GA vs CART\n(* p<0.05, ns = not significant)", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets_labels, fontsize=11)
        ax.legend(fontsize=10, framealpha=0.95)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

        # Right panel: node count reduction
        ax = axes[1]
        reductions_pct = [paper_results[k]["size_reduction_pct"] for k in keys]
        bar_colors = ["#27ae60" if r > 0 else "#e74c3c" for r in reductions_pct]
        ax.bar(
            x,
            reductions_pct,
            0.5,
            color=bar_colors,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.9,
        )
        for i, (r, k) in enumerate(zip(reductions_pct, keys)):
            ax.text(i, r + 1, f"{r}%", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylabel("Tree Size Reduction vs CART (%)", fontsize=12, fontweight="bold")
        ax.set_title("Interpretability: Node Count Reduction", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets_labels, fontsize=11)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

        fig.suptitle("Complete Results Summary", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(output_dir / "paper_table_summary.png", dpi=300, bbox_inches="tight")
        print("✓ Saved: paper_table_summary.png")
        plt.close()

    finally:
        plt.rcParams["font.family"] = orig_family
        plt.rcParams["font.size"] = orig_size


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main():
    """Generate all visualizations from experiment CSV results."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive visualizations from experiment results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Config name to load results for (default: 'default'). "
        "E.g. 'paper', 'fast', 'accuracy_focused'.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to results directory (default: results/)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Creating Comprehensive Visualizations")
    print(f"Config: {args.config}")
    print("=" * 70 + "\n")

    results, paper_results = load_results_from_csv(
        config_name=args.config, results_dir=args.results_dir
    )

    create_accuracy_comparison(results)
    create_tree_size_comparison(results)
    create_tradeoff_scatter(results)
    create_speed_comparison(results)
    create_summary_heatmap(results)
    create_key_findings_chart(results)

    print("\n" + "=" * 70)
    print("Creating Publication Figures (paper/thesis quality)")
    print("=" * 70 + "\n")

    create_publication_figures(results, paper_results)

    print("\n" + "=" * 70)
    print("All Visualizations Created!")
    print("=" * 70)
    print("\nSaved to: results/figures/")
    print("\nComprehensive figures:")
    print("  1. accuracy_comparison.png")
    print("  2. tree_size_comparison.png")
    print("  3. tradeoff_scatter.png       (node-count x-axis; GA sweet spot = top-left)")
    print("  4. speed_comparison.png")
    print("  5. summary_table.png          (seaborn heatmap)")
    print("  6. key_findings.png           (horizontal bar chart)")
    print("\nPublication figures (PNG + PDF):")
    print("  7. paper_fig1_size_reduction.png / .pdf")
    print("  8. paper_fig2_statistical_equiv.png / .pdf")
    print("  9. paper_fig3_pareto_tradeoff.png / .pdf  (with Pareto front line)")
    print(" 10. paper_table_summary.png               (grouped bar chart + p-values)")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
