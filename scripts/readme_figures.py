#!/usr/bin/env python
"""Draw the README figures: the GA loop and one run's fitness curve.

Both images use a dark theme on GitHub's dark page colour so they sit flush in
the README. The fitness curve comes from the same seeded run as the README's
Python example, so the picture and the code agree. Rerunning the script on the
same versions reproduces the PNGs byte for byte.

    python scripts/readme_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ga_trees import FitnessCalculator, GAConfig, GAEngine, Mutation, TreeInitializer  # noqa: E402
from ga_trees.data import DatasetLoader  # noqa: E402

OUT = ROOT / "docs" / "assets" / "readme"

# GitHub's dark page and its neutral steps, so the figures blend into the page.
SURFACE = "#0d1117"
CARD = "#161b22"
BORDER = "#30363d"
GRID = "#21262d"
INK = "#f0f6fc"
SECONDARY = "#c9d1d9"
MUTED = "#8b949e"
# Dark-mode steps of the first two categorical slots; validated against SURFACE
# (contrast >= 3:1, CVD separation well above the floor).
BLUE = "#3987e5"
ORANGE = "#d95926"


def _save(fig, name: str, tight: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / name,
        dpi=200,
        facecolor=SURFACE,
        metadata={"Software": None},
        bbox_inches="tight" if tight else None,
        pad_inches=0.3,
    )
    plt.close(fig)
    print(f"wrote {(OUT / name).relative_to(ROOT)}")


def how_it_works() -> None:
    """The evolutionary loop as a left-to-right pipeline with a loop-back arc."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    steps = [
        ("Population", "random trees,\nsplits from the data"),
        ("Evaluate", "fitness on a\nheld-out split"),
        ("Select", "tournament\nplus elitism"),
        ("Crossover", "swap subtrees\nbetween parents"),
        ("Mutate", "threshold, feature,\nprune, grow"),
    ]
    width, height, y = 1.62, 1.05, 1.55
    centres = [0.95 + i * 2.02 for i in range(len(steps))]

    for x, (title, detail) in zip(centres, steps):
        ax.add_patch(
            FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=CARD,
                edgecolor=BORDER,
                linewidth=1.2,
            )
        )
        ax.text(
            x, y + 0.2, title, ha="center", va="center", color=INK, fontsize=11.5, fontweight="bold"
        )
        ax.text(
            x,
            y - 0.2,
            detail,
            ha="center",
            va="center",
            color=SECONDARY,
            fontsize=8.8,
            linespacing=1.35,
        )

    arrow = dict(arrowstyle="-|>", mutation_scale=14, color=MUTED, linewidth=1.4)
    for left, right in zip(centres, centres[1:]):
        ax.add_patch(
            FancyArrowPatch((left + width / 2 + 0.04, y), (right - width / 2 - 0.04, y), **arrow)
        )

    # Offspring go back to Evaluate: the generation loop.
    ax.add_patch(
        FancyArrowPatch(
            (centres[4], y - height / 2 - 0.04),
            (centres[1], y - height / 2 - 0.04),
            connectionstyle="arc3,rad=-0.28",
            arrowstyle="-|>",
            mutation_scale=15,
            color=BLUE,
            linewidth=2,
        )
    )
    ax.text(
        (centres[1] + centres[4]) / 2,
        0.58,
        "next generation",
        ha="center",
        va="center",
        color=SECONDARY,
        fontsize=9.5,
    )

    # After the last generation the best tree seen comes out of Evaluate.
    ax.add_patch(
        FancyArrowPatch(
            (centres[1], y + height / 2 + 0.04),
            (centres[1], 3.05),
            arrowstyle="-|>",
            mutation_scale=15,
            color=ORANGE,
            linewidth=2,
        )
    )
    ax.text(
        centres[1] + 0.14,
        3.25,
        "best tree, after the last generation",
        ha="left",
        va="center",
        color=INK,
        fontsize=10,
        fontweight="bold",
    )

    _save(fig, "how-it-works.png", tight=True)


def evolution_curve() -> None:
    """Best and mean fitness per generation for the README example's run."""
    data = DatasetLoader().load_dataset("breast_cancer", test_size=0.2)
    X_train, y_train = data["X_train"], data["y_train"]
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=0
    )
    n_features = X_fit.shape[1]
    engine = GAEngine(
        GAConfig(population_size=80, n_generations=40, random_state=42),
        TreeInitializer(
            n_features, n_classes=2, max_depth=5, min_samples_split=10, min_samples_leaf=5
        ),
        FitnessCalculator(accuracy_weight=0.9, interpretability_weight=0.1).calculate_fitness,
        Mutation(
            n_features,
            feature_ranges={i: (X_fit[:, i].min(), X_fit[:, i].max()) for i in range(n_features)},
            X=X_fit,
            min_samples_leaf=5,
        ),
    )
    best_tree = engine.evolve(X_fit, y_fit, X_val=X_val, y_val=y_val, verbose=False)
    history = engine.get_history()
    best = np.asarray(history["best_fitness"])
    mean = np.asarray(history["avg_fitness"])
    generations = np.arange(1, len(best) + 1)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=6)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)

    for values, colour, label, short in (
        (best, BLUE, "best", "best"),
        (mean, ORANGE, "population mean", "mean"),
    ):
        ax.plot(
            generations,
            values,
            color=colour,
            linewidth=2,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=label,
        )
        ax.scatter(
            generations[-1],
            values[-1],
            s=64,
            color=colour,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
        )
        ax.annotate(
            f"{short}  {values[-1]:.3f}",
            (generations[-1], values[-1]),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            color=SECONDARY,
            fontsize=9.5,
        )

    ax.set_xlim(0.5, generations[-1] + 7)
    ax.set_xlabel("Generation", color=MUTED, fontsize=9.5, labelpad=8)
    ax.set_ylabel("Fitness", color=MUTED, fontsize=9.5, labelpad=8)
    legend = ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=9.5,
        handlelength=1.6,
        bbox_to_anchor=(0.86, 0.02),
    )
    for text in legend.get_texts():
        text.set_color(SECONDARY)

    fig.text(
        0.065,
        0.93,
        "One run on breast cancer",
        color=INK,
        fontsize=13,
        fontweight="bold",
        ha="left",
    )
    fig.text(
        0.065,
        0.865,
        f"80 trees × 40 generations, fitness scored on a held-out split. "
        f"Final tree: {best_tree.get_num_nodes()} nodes, depth {best_tree.get_depth()}.",
        color=SECONDARY,
        fontsize=9.5,
        ha="left",
    )
    fig.subplots_adjust(left=0.065, right=0.97, top=0.8, bottom=0.14)
    _save(fig, "evolution.png")


def main() -> int:
    how_it_works()
    evolution_curve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
