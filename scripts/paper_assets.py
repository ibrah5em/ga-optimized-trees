#!/usr/bin/env python
"""Every number, table and figure in ``paper/gecco`` — computed, never typed.

Reads only committed evidence under ``paper/evidence/`` and writes
``paper/gecco/generated/``. The paper ``\\input``s the generated files, so a
number in the PDF can always be traced to a CSV and to this script. An evidence
set that does not exist yet is skipped and its macros are left undefined, which
makes LaTeX fail loudly instead of printing a stale value.

    python scripts/paper_assets.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ga_trees.evaluation.statistics import (  # noqa: E402
    compare_all_to_reference,
    corrected_fold_equivalence,
    equivalence_test,
    friedman_nemenyi,
)

EVIDENCE = ROOT / "paper" / "evidence"
OUT = ROOT / "paper" / "gecco" / "generated"
FRONTIER = EVIDENCE / "frontier-2026-08-07"
REPAIR = EVIDENCE / "frontier-repair-2026-09-29"
GOSDT = EVIDENCE / "gosdt-2026-09-29"
K3 = EVIDENCE / "k3-2026-09-29"
VIOLATIONS = EVIDENCE / "constraint-violations-2026-09-29" / "violations.csv"

GA, GA_ARCH, RS, CART, GOSDT_NAME = (
    "GA (NSGA-II)",
    "GA (NSGA-II, archived)",
    "Random Search",
    "CART (ccp path)",
    "GOSDT (reg path)",
)
K2_THRESHOLD = 0.60

# Reference categorical palette (dataviz skill, light mode), first three slots —
# the only ones validated for all-pairs use. CART is the reference and is drawn
# in neutral ink with its own marker rather than a fourth hue.
COLORS = {GA: "#2a78d6", RS: "#eb6834", GOSDT_NAME: "#1baf7a", CART: "#52514e"}
MARKERS = {GA: "o", RS: "s", GOSDT_NAME: "D", CART: "^"}
LABELS = {GA: "GA (NSGA-II)", RS: "Random search", CART: "CART ccp path", GOSDT_NAME: "GOSDT"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

macros = {}


def macro(name: str, value) -> None:
    if not name.isalpha():
        raise ValueError(f"LaTeX macro names must be letters only: {name}")
    macros[name] = value


def fmt(value: float, digits: int = 3, sign: bool = False) -> str:
    text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    return text.replace("-", "$-$") if not sign else text.replace("-", "$-$").replace("+", "$+$")


def pval(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def dataset_means(folds: pd.DataFrame, value: str = "hypervolume") -> pd.DataFrame:
    return folds.groupby(["dataset", "method"])[value].mean().unstack()


def comparisons(means: pd.DataFrame, reference: str):
    scores = {m: means[m].tolist() for m in means.columns}
    return {c.method_b: c for c in compare_all_to_reference(scores, reference=reference)}


# ---------------------------------------------------------------------------
# K1 / K2 — the pre-registered frontier run
# ---------------------------------------------------------------------------


def frontier_section():
    folds = pd.read_csv(FRONTIER / "folds.csv")
    points = pd.read_csv(FRONTIER / "points.csv")
    seeds = json.load(open(FRONTIER / "seeds.json"))
    means = dataset_means(folds)
    reference = folds.groupby("dataset").reference_nodes.first()
    normalised = means.div(reference, axis=0)

    tests = comparisons(means, GA)
    k1 = tests[RS]
    macro("NDatasets", len(means))
    macro("NFoldsFrontier", folds.fold.nunique())
    macro("KoneDiff", fmt(k1.mean_difference, 2, sign=True))
    macro("KoneP", pval(k1.p_value))
    macro("KonePholm", pval(k1.p_adjusted))
    macro("KoneDz", fmt(k1.effect_size, 2, sign=True))
    macro("KoneWins", int((means[GA] > means[RS]).sum()))
    macro("ArchDiff", fmt(tests[GA_ARCH].mean_difference, 2, sign=True))
    macro("ArchPholm", pval(tests[GA_ARCH].p_adjusted))
    macro("CartDiff", fmt(tests[CART].mean_difference, 2, sign=True))
    macro("CartPholm", pval(tests[CART].p_adjusted))
    rate = float((means[GA] > means[CART]).mean())
    macro("KtwoRate", f"{rate:.0%}".replace("%", "\\%"))
    macro("KtwoWins", int((means[GA] > means[CART]).sum()))
    macro("KtwoThreshold", f"{K2_THRESHOLD:.0%}".replace("%", "\\%"))
    macro("BudgetMismatch", int(seeds["protocol"]["budget_mismatched_folds"]))
    evals = folds[folds.method == GA].n_evaluations.mean()
    macro("MeanEvaluations", f"{evals:,.0f}".replace(",", "{,}"))

    friedman = friedman_nemenyi({m: means[m].tolist() for m in means.columns})
    macro("FriedmanP", pval(friedman.p_value))
    for method, key in ((GA, "GA"), (GA_ARCH, "Arch"), (RS, "RS"), (CART, "Cart")):
        macro(f"Rank{key}", f"{friedman.average_ranks[method]:.2f}")

    # Delivered-front statistics.
    per_fold = folds.groupby("method")[["n_candidates", "n_points"]].mean()
    macro("DeliveredGA", f"{per_fold.loc[GA, 'n_candidates']:.1f}")
    macro("DeliveredRS", f"{per_fold.loc[RS, 'n_candidates']:.1f}")

    # Exploratory: where does the GA lose to CART? (post hoc)
    largest = points.groupby(["dataset", "method", "fold"]).nodes.max()
    largest = largest.groupby(["dataset", "method"]).mean().unstack()
    best = dataset_means(folds, "best_accuracy")
    gap = normalised[GA] - normalised[CART]
    best_gap = best[GA] - best[CART]
    rho = spearmanr(gap, best_gap)
    macro("MedianMaxNodesGA", f"{largest[GA].median():.1f}")
    macro("MedianMaxNodesRS", f"{largest[RS].median():.1f}")
    macro("MedianMaxNodesCart", f"{largest[CART].median():.1f}")
    macro("RhoBestGap", f"{rho.statistic:.2f}")
    macro("RhoBestGapP", pval(rho.pvalue))
    macro("BestAccGA", f"{best[GA].mean():.3f}")
    macro("BestAccCart", f"{best[CART].mean():.3f}")

    return folds, points, means, normalised, largest, best


def frontier_table(normalised, largest, extra=None):
    """Per-dataset normalised hypervolume, best method in bold."""
    columns = [GA, RS, CART] + ([GOSDT_NAME] if extra is not None else [])
    table = normalised.copy()
    if extra is not None:
        table[GOSDT_NAME] = extra
    lines = [
        "\\begin{tabular}{l" + "r" * len(columns) + "rr}",
        "\\toprule",
        "Dataset & "
        + " & ".join(LABELS[c].replace("CART ccp path", "CART") for c in columns)
        + " & \\multicolumn{2}{c}{Largest tree} \\\\",
        " & " * len(columns) + " & GA & CART \\\\",
        "\\midrule",
    ]
    for dataset, row in table.iterrows():
        top = max(row[c] for c in columns if not np.isnan(row[c]))
        cells = []
        for c in columns:
            value = row[c]
            text = "--" if np.isnan(value) else f"{value:.3f}"
            cells.append(f"\\textbf{{{text}}}" if value == top else text)
        name = dataset.replace("_", "\\_")
        cells += [f"{largest.loc[dataset, GA]:.0f}", f"{largest.loc[dataset, CART]:.0f}"]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_frontier.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Sensitivity: constraint repair
# ---------------------------------------------------------------------------


def repair_section(primary_means):
    if not (REPAIR / "folds.csv").exists():
        return None
    folds = pd.read_csv(REPAIR / "folds.csv")
    means = dataset_means(folds)
    tests = comparisons(means, GA)
    macro("RepairKoneDiff", fmt(tests[RS].mean_difference, 2, sign=True))
    macro("RepairKonePholm", pval(tests[RS].p_adjusted))
    macro("RepairKoneWins", int((means[GA] > means[RS]).sum()))
    macro("RepairKtwoRate", f"{float((means[GA] > means[CART]).mean()):.0%}".replace("%", "\\%"))
    macro("RepairKtwoWins", int((means[GA] > means[CART]).sum()))
    # Same folds, same seeds: the GA change is the only difference.
    shift = means[GA] - primary_means[GA]
    macro("RepairGAShift", fmt(float(shift.mean()), 2, sign=True))
    return means


def violations_section():
    table = pd.read_csv(VIOLATIONS)
    ga = table[table.method == GA]
    rs = table[table.method == RS]
    macro("ViolDatasets", len(ga))
    macro("ViolMin", f"{100 * ga.evaluated_share_violating.min():.0f}")
    macro("ViolMax", f"{100 * ga.evaluated_share_violating.max():.0f}")
    macro("ViolRS", f"{100 * rs.evaluated_share_violating.max():.0f}")
    macro("ViolDelivered", int(ga.delivered_violating_nodes.sum()))
    macro("ViolDeliveredOf", int(ga.delivered_internal_nodes.sum()))


# ---------------------------------------------------------------------------
# Exploratory: GOSDT
# ---------------------------------------------------------------------------


def gosdt_section():
    if not (GOSDT / "gosdt-folds.csv").exists():
        return None
    folds = pd.read_csv(GOSDT / "gosdt-folds.csv")
    points = pd.read_csv(GOSDT / "gosdt-points.csv")
    reference = folds.groupby("dataset").reference_nodes.first()
    means = dataset_means(folds).div(reference, axis=0)
    macro("GosdtFolds", folds.fold.nunique())
    wins_cart = int((means[GOSDT_NAME] > means[CART]).sum())
    wins_ga = int((means[GA] > means[GOSDT_NAME]).sum())
    macro("GosdtBeatsCart", wins_cart)
    macro("GABeatsGosdt", wins_ga)
    tests = comparisons(dataset_means(folds), GA)
    macro("GosdtDiff", fmt(tests[GOSDT_NAME].mean_difference, 2, sign=True))
    macro("GosdtPholm", pval(tests[GOSDT_NAME].p_adjusted))
    timing = points.groupby(["dataset", "fold"]).agg(
        t=("fit_seconds", "first"), n=("timeouts", "first")
    )
    macro("GosdtSeconds", f"{timing.t.mean():.1f}")
    macro("GosdtTimeouts", int(timing.n.sum()))
    macro("GosdtFits", int(len(timing) * 10))
    return means[GOSDT_NAME], points


# ---------------------------------------------------------------------------
# K3 / H2 — point-estimate run
# ---------------------------------------------------------------------------


def k3_section():
    files = sorted(K3.glob("folds*.csv"))
    if not files:
        return None
    folds = pd.read_csv(files[0])
    k3_ga, k3_cart = "GA-Optimized", "CART (pruned)"
    splits = folds.fold.nunique()
    rows = []
    for dataset, block in folds.groupby("dataset"):
        wide = block.pivot_table(index="fold", columns="method", values="test_accuracy")
        diff = (wide[k3_ga] - wide[k3_cart]).dropna()
        mean, low, high, equivalent = corrected_fold_equivalence(diff, 1.0 / (splits - 1))
        leaves = block.groupby("method").leaves.mean()
        rows.append(
            {
                "dataset": dataset,
                "diff": mean,
                "low": low,
                "high": high,
                "equivalent": equivalent,
                "ga_leaves": leaves[k3_ga],
                "cart_leaves": leaves[k3_cart],
            }
        )
    table = pd.DataFrame(rows).set_index("dataset")
    n = len(table)
    primary = int((table["diff"] < -0.02).sum())
    secondary = int((~table["equivalent"]).sum())
    macro("KthreePrimary", primary)
    macro("KthreeSecondary", secondary)
    macro("KthreeN", n)
    macro("KthreeFires", "fires" if primary / n > 0.30 else "does not fire")

    means = dataset_means(folds, "test_accuracy")
    h2 = equivalence_test(means[k3_ga].tolist(), means[k3_cart].tolist())
    macro("HtwoDiff", fmt(h2.mean_difference, 3, sign=True))
    macro("HtwoLow", fmt(h2.ci_low, 3, sign=True))
    macro("HtwoHigh", fmt(h2.ci_high, 3, sign=True))
    macro("HtwoP", pval(h2.p_value))
    macro("HtwoVerdict", "equivalent" if h2.equivalent else "not equivalent")

    tests = comparisons(means, k3_ga)
    for method, key in (
        ("Random Search", "RS"),
        (k3_cart, "Cart"),
        ("CART (unconstrained)", "CartFull"),
        ("Random Forest", "RF"),
    ):
        if method in tests:
            macro(f"AccDiff{key}", fmt(tests[method].mean_difference, 3, sign=True))
            macro(f"AccPholm{key}", pval(tests[method].p_adjusted))
    for method, key in ((k3_ga, "GA"), (k3_cart, "Cart"), ("Random Search", "RS")):
        sub = folds[folds.method == method]
        per_dataset = sub.groupby("dataset")[
            ["test_accuracy", "leaves", "mean_path_length", "features_used"]
        ]
        values = per_dataset.mean().mean()
        macro(f"Acc{key}", f"{values.test_accuracy:.3f}")
        macro(f"Leaves{key}", f"{values.leaves:.1f}")
        macro(f"Path{key}", f"{values.mean_path_length:.2f}")
        macro(f"Features{key}", f"{values.features_used:.1f}")
    return table


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figure_hv_differences(normalised):
    import matplotlib.pyplot as plt

    diff_rs = normalised[GA] - normalised[RS]
    diff_cart = normalised[GA] - normalised[CART]
    order = diff_cart.sort_values().index
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    _style(ax)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.scatter(
        diff_rs[order],
        y + 0.15,
        s=16,
        color=COLORS[RS],
        marker=MARKERS[RS],
        label="GA $-$ random search",
        zorder=3,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.scatter(
        diff_cart[order],
        y - 0.15,
        s=18,
        color=COLORS[CART],
        marker=MARKERS[CART],
        label="GA $-$ CART ccp path",
        zorder=3,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(y, [d.replace("_", " ") for d in order], fontsize=6.5, color=INK)
    ax.set_xlabel("Normalised hypervolume, GA minus baseline", fontsize=7, color=INK)
    ax.legend(
        fontsize=6.5,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.45, 1.0),
        ncol=2,
        handletextpad=0.2,
        columnspacing=1.0,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_hv_diff.pdf")
    plt.close(fig)


def figure_frontiers(points, gosdt_points=None, datasets=("breast_w", "vehicle")):
    """Test-fold frontiers on one fold, one panel per dataset."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4, 1.8))
    for ax, dataset in zip(axes, datasets):
        _style(ax)
        ax.grid(axis="both", color=GRID, linewidth=0.6)
        sub = points[(points.dataset == dataset) & (points.fold == 1)]
        methods = [CART, RS, GA]
        if gosdt_points is not None:
            g = gosdt_points[(gosdt_points.dataset == dataset) & (gosdt_points.fold == 1)]
            sub = pd.concat([sub, g[sub.columns]])
            methods.insert(1, GOSDT_NAME)
        for method in methods:
            pts = sub[sub.method == method].sort_values("nodes")
            if pts.empty:
                continue
            ax.step(pts.nodes, pts.accuracy, where="post", color=COLORS[method], linewidth=1.2)
            ax.scatter(
                pts.nodes,
                pts.accuracy,
                s=14,
                color=COLORS[method],
                marker=MARKERS[method],
                label=LABELS[method],
                zorder=3,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_xscale("log")
        ax.set_title(dataset.replace("_", " "), fontsize=7, color=INK)
        ax.set_xlabel("Nodes (log)", fontsize=7, color=INK)
    axes[0].set_ylabel("Test accuracy", fontsize=7, color=INK)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=6,
        frameon=False,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig_frontiers.pdf")
    plt.close(fig)


def figure_k3(table):
    import matplotlib.pyplot as plt

    order = table["diff"].sort_values().index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    _style(ax)
    ax.axvspan(-0.02, 0.02, color=GRID, zorder=0, label="$\\pm$2-point margin")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    t = table.loc[order]
    ax.hlines(y, t["low"], t["high"], color=COLORS[GA], linewidth=1.2)
    ax.scatter(
        t["diff"],
        y,
        s=16,
        color=COLORS[GA],
        zorder=3,
        edgecolor="white",
        linewidth=0.5,
        label="GA $-$ tuned CART (90% CI)",
    )
    ax.set_yticks(y, [d.replace("_", " ") for d in order], fontsize=6.5, color=INK)
    ax.set_xlabel("Test accuracy difference", fontsize=7, color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_k3.pdf")
    plt.close(fig)


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})
    OUT.mkdir(parents=True, exist_ok=True)

    folds, points, means, normalised, largest, best = frontier_section()
    repair_section(means)
    violations_section()
    gosdt = gosdt_section()
    frontier_table(normalised, largest, extra=None if gosdt is None else gosdt[0])
    figure_hv_differences(normalised)
    figure_frontiers(points, None if gosdt is None else gosdt[1])
    table = k3_section()
    if table is not None:
        figure_k3(table)

    lines = ["% Generated by scripts/paper_assets.py — do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(macros)} macros -> {OUT / 'numbers.tex'}")
    for key in ("KoneDiff", "KonePholm", "KoneWins", "KtwoRate", "MedianMaxNodesGA"):
        print(f"  {key} = {macros.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
