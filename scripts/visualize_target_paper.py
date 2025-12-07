"""
Create publication-quality figures for research paper/thesis.
Highlights the key achievements: 48% size reduction and p=0.64 equivalence.
to see the same results in this visualizer, run:
    python scripts/experiment.py --config configs/target.yaml
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Ensure output directory exists
os.makedirs("results/figures", exist_ok=True)

# Set publication style
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 300
sns.set_palette("colorblind")

# Your actual results
results = {
    "iris": {
        "ga_acc": 94.55,
        "cart_acc": 92.41,
        "ga_nodes": 7.4,
        "cart_nodes": 16.4,
        "p_val": 0.186,
    },
    "wine": {
        "ga_acc": 88.19,
        "cart_acc": 87.22,
        "ga_nodes": 10.7,
        "cart_nodes": 20.7,
        "p_val": 0.683,
    },
    "breast_cancer": {
        "ga_acc": 91.05,
        "cart_acc": 91.57,
        "ga_nodes": 6.5,
        "cart_nodes": 35.5,
        "p_val": 0.640,
    },
}

# FIGURE 1: Size Reduction
fig, ax = plt.subplots(figsize=(8, 5))

datasets = ["Iris", "Wine", "Breast\nCancer"]
ga_nodes = [7.4, 10.7, 6.5]
cart_nodes = [16.4, 20.7, 35.5]
reductions = [55, 48, 82]

x = np.arange(len(datasets))
width = 0.35

bars_ga = ax.bar(
    x - width / 2, ga_nodes, width, label="GA", color="#2ecc71", edgecolor="black", linewidth=1.2
)
bars_cart = ax.bar(
    x + width / 2,
    cart_nodes,
    width,
    label="CART",
    color="#e74c3c",
    alpha=0.85,
    edgecolor="black",
    linewidth=1.2,
)

# Add reduction percentages (no emoji, no special symbols)
for i, (ga, cart, red) in enumerate(zip(ga_nodes, cart_nodes, reductions)):
    y_pos = max(ga, cart) + 2
    color = "#27ae60" if 46 <= red <= 49 else "#2c3e50"
    ax.text(i, y_pos, f"{red}%", ha="center", fontsize=11, fontweight="bold", color=color)

# Optional: highlight a horizontal band (example kept neutral)
# If you prefer a specific band for the target zone, adjust the ymin/ymax accordingly.
# Example commented out (uncomment if desired):
# ax.axhspan(0, 50, alpha=0.03, color='gray', zorder=0)

ax.set_ylabel("Number of Nodes", fontsize=12, fontweight="bold")
ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
ax.set_title("GA Achieves 46–82% Tree Size Reduction", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.set_ylim(0, 40)

plt.tight_layout()
plt.savefig("results/figures/paper_fig1_size_reduction.png", dpi=300, bbox_inches="tight")
plt.savefig("results/figures/paper_fig1_size_reduction.pdf", bbox_inches="tight")
print("Figure 1 saved (Size Reduction)")

# FIGURE 2: Statistical Equivalence - The p-value Plot
fig, ax = plt.subplots(figsize=(8, 5))

datasets = ["Iris", "Wine", "Breast Cancer"]
p_values = [0.186, 0.683, 0.640]
colors = ["#3498db", "#3498db", "#27ae60"]  # Green for the target achievement

bars = ax.barh(datasets, p_values, color=colors, edgecolor="black", linewidth=1.2, alpha=0.9)

# Significance threshold line
ax.axvline(
    0.05, color="red", linestyle="--", linewidth=1.8, label="α = 0.05 (significance threshold)"
)

# Target line example (adjust if you have a different target)
ax.axvline(
    0.55, color="orange", linestyle=":", linewidth=1.8, alpha=0.8, label="Target p-value (0.55)"
)

# Annotations (plain text, no emojis)
for i, (dataset, p) in enumerate(zip(datasets, p_values)):
    label = f"p = {p:.3f}"
    # Example target labels (kept plain): uncomment or change logic as desired
    # if dataset == 'Breast Cancer' and p >= 0.55:
    #     label += '  Target'
    ax.text(p + 0.03, i, label, va="center", fontsize=11, fontweight="bold")

ax.set_xlabel("p-value (Paired t-test, 20-fold CV)", fontsize=12, fontweight="bold")
ax.set_title(
    "Statistical Equivalence to CART\n(All p > 0.05 = No Significant Difference)",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(0, 0.75)
ax.grid(axis="x", alpha=0.25, linestyle="--")

plt.tight_layout()
plt.savefig("results/figures/paper_fig2_statistical_equiv.png", dpi=300, bbox_inches="tight")
plt.savefig("results/figures/paper_fig2_statistical_equiv.pdf", bbox_inches="tight")
print("Figure 2 saved (Statistical Equivalence)")

# FIGURE 3: Accuracy-Interpretability Trade-off Scatter
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each dataset's GA and CART points (no emoji)
for dataset, data in results.items():
    # GA point
    ax.scatter(
        data["ga_nodes"],
        data["ga_acc"],
        s=200,
        alpha=0.85,
        color="#2ecc71",
        marker="o",
        edgecolors="black",
        linewidth=1.2,
        label=f"GA - {dataset}",
    )
    # CART point
    ax.scatter(
        data["cart_nodes"],
        data["cart_acc"],
        s=140,
        alpha=0.8,
        color="#e74c3c",
        marker="s",
        edgecolors="black",
        linewidth=1.2,
        label=f"CART - {dataset}",
    )

    # Connect with arrow (gray)
    ax.annotate(
        "",
        xy=(data["ga_nodes"], data["ga_acc"]),
        xytext=(data["cart_nodes"], data["cart_acc"]),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="gray", alpha=0.6),
    )

# Ideal region (subtle shading)
ax.axhspan(90, 95, alpha=0.03, color="green", zorder=0)
ax.axvspan(0, 15, alpha=0.03, color="blue", zorder=0)
ax.text(8.0, 94.5, "Ideal region", fontsize=10, color="darkgreen", fontweight="bold")

ax.set_xlabel("Tree Size (Number of Nodes)", fontsize=12, fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
ax.set_title(
    "GA Finds Pareto-Optimal Solutions\n(Smaller trees, competitive accuracy)",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.grid(True, alpha=0.25, linestyle="--")
ax.set_xlim(0, 40)
ax.set_ylim(85, 96)

# Custom legend
legend_elements = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#2ecc71",
        markersize=10,
        label="GA (Smaller)",
        markeredgecolor="black",
        markeredgewidth=1.2,
    ),
    plt.Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor="#e74c3c",
        markersize=8,
        label="CART (Baseline)",
        markeredgecolor="black",
        markeredgewidth=1.2,
    ),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.95)

plt.tight_layout()
plt.savefig("results/figures/paper_fig3_pareto_tradeoff.png", dpi=300, bbox_inches="tight")
plt.savefig("results/figures/paper_fig3_pareto_tradeoff.pdf", bbox_inches="tight")
print("Figure 3 saved (Pareto Trade-off)")

# FIGURE 4: Comparison Table as Image (for slides)
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")

table_data = [
    ["Iris", "94.55 ± 8.07%", "92.41 ± 10.43%", "0.186", "7.4", "16.4", "55%"],
    ["Wine", "88.19 ± 10.39%", "87.22 ± 10.70%", "0.683", "10.7", "20.7", "48%"],
    ["Breast Cancer", "91.05 ± 5.60%", "91.57 ± 3.92%", "0.640", "6.5", "35.5", "82%"],
]

headers = [
    "Dataset",
    "GA Accuracy",
    "CART Accuracy",
    "p-value",
    "GA Nodes",
    "CART Nodes",
    "Reduction",
]

table = ax.table(
    cellText=table_data,
    colLabels=headers,
    cellLoc="center",
    loc="center",
    colWidths=[0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.12],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for col_idx in range(len(headers)):
    cell = table[(0, col_idx)]
    cell.set_facecolor("#34495e")
    cell.set_text_props(weight="bold", color="white")

# Highlight achievements (plain coloring, no emoji)
for row_idx in range(1, len(table_data) + 1):
    # p-value column - green if > 0.05
    cell_p = table[(row_idx, 3)]
    try:
        pval = float(table_data[row_idx - 1][3].split()[0])
    except Exception:
        pval = None
    if pval is not None and pval > 0.05:
        cell_p.set_facecolor("#d5f4e6")

    # Reduction column - gold if 46-49%, otherwise green tint
    cell_red = table[(row_idx, 6)]
    reduction = int(table_data[row_idx - 1][6].replace("%", "").split()[0])
    if 46 <= reduction <= 49:
        cell_red.set_facecolor("#f9e79f")  # gold for exact target
    else:
        cell_red.set_facecolor("#d5f4e6")

ax.set_title("Complete Results Summary (20-fold CV)", fontsize=14, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig("results/figures/paper_table_summary.png", dpi=300, bbox_inches="tight")
print("Summary table saved")

print("\n" + "=" * 70)
print("All paper-quality figures generated.")
print("=" * 70)
print("\nFiles created:")
print("  1. paper_fig1_size_reduction.png/pdf (Size reduction)")
print("  2. paper_fig2_statistical_equiv.png/pdf (p-value plot)")
print("  3. paper_fig3_pareto_tradeoff.png/pdf (Trade-off visualization)")
print("  4. paper_table_summary.png (Results table)")
print("\nUse these for your paper/presentation.")
