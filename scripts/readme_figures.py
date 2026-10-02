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
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
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


# A tree glyph is a nested tuple: a leaf is a colour string, an internal node is
# (left, right) or (left, right, ring_colour) to highlight that split.
Glyph = Union[str, tuple]


def _leaves(tree: Glyph) -> int:
    return 1 if isinstance(tree, str) else _leaves(tree[0]) + _leaves(tree[1])


def _depth(tree: Glyph) -> int:
    return 0 if isinstance(tree, str) else 1 + max(_depth(tree[0]), _depth(tree[1]))


def draw_tree(ax, tree: Glyph, x: float, y: float, width: float, height: float, alpha=1.0):
    """Draw a small decision-tree glyph with its root at (x, y), growing downwards.

    Leaves are spread evenly across *width*; internal nodes sit over the middle of
    their leaves. Leaves are squares in a class colour, splits are light circles.
    """
    n_leaves, depth = _leaves(tree), max(_depth(tree), 1)
    step_x = width / max(n_leaves - 1, 1)
    step_y = height / depth
    cursor = [x - width / 2 if n_leaves > 1 else x]
    node = 0.055

    def place(sub, level):
        top = y - level * step_y
        if isinstance(sub, str):
            cx = cursor[0]
            cursor[0] += step_x
            ax.add_patch(
                FancyBboxPatch(
                    (cx - node, top - node),
                    2 * node,
                    2 * node,
                    boxstyle="round,pad=0,rounding_size=0.015",
                    facecolor=sub,
                    edgecolor="none",
                    alpha=alpha,
                    zorder=4,
                )
            )
            return cx, top
        left = place(sub[0], level + 1)
        right = place(sub[1], level + 1)
        cx = (left[0] + right[0]) / 2
        for child in (left, right):
            ax.plot(
                [cx, child[0]],
                [top, child[1]],
                color=MUTED,
                linewidth=1.3,
                alpha=alpha,
                zorder=2,
                solid_capstyle="round",
            )
        ring = sub[2] if len(sub) > 2 else None
        ax.add_patch(
            Circle(
                (cx, top),
                node * 1.05,
                facecolor=SECONDARY,
                edgecolor=ring or "none",
                linewidth=2.2 if ring else 0,
                alpha=alpha,
                zorder=3,
            )
        )
        return cx, top

    place(tree, 0)


def how_it_works() -> None:
    """The evolutionary loop as a pipeline of cards, each showing what happens to the trees."""
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 5.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Leaf colours, one per class.
    b, o = BLUE, ORANGE
    small = ((b, o), o)
    deeper = ((b, (o, b)), (o, b))

    steps = [
        ("Population", "random trees,\nsplits from the data"),
        ("Evaluate", "fitness on a\nheld-out split"),
        ("Select", "tournament\nplus elitism"),
        ("Crossover", "swap subtrees\nbetween parents"),
        ("Mutate", "threshold, feature,\nprune, grow"),
    ]
    width, bottom, top = 1.75, 0.95, 2.95
    row_y = (bottom + top) / 2
    icon_y = top - 0.2  # root of each card's glyph
    icon_h = 0.5
    centres = [1.0 + i * 2.1 for i in range(len(steps))]

    for x, (title, detail) in zip(centres, steps):
        ax.add_patch(
            FancyBboxPatch(
                (x - width / 2, bottom),
                width,
                top - bottom,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=CARD,
                edgecolor=BORDER,
                linewidth=1.2,
            )
        )
        ax.text(
            x, 1.72, title, ha="center", va="center", color=INK, fontsize=11.5, fontweight="bold"
        )
        ax.text(
            x,
            1.3,
            detail,
            ha="center",
            va="center",
            color=SECONDARY,
            fontsize=8.6,
            linespacing=1.35,
        )

    pop, ev, sel, cx_, mut = centres
    # Population: a handful of different random trees.
    draw_tree(ax, small, pop - 0.52, icon_y, 0.3, icon_h)
    draw_tree(ax, deeper, pop, icon_y, 0.46, icon_h)
    draw_tree(ax, (o, (b, o)), pop + 0.52, icon_y, 0.3, icon_h)
    # Evaluate: one tree and its fitness, as a filled meter.
    draw_tree(ax, deeper, ev - 0.25, icon_y, 0.46, icon_h)
    ax.add_patch(
        FancyBboxPatch(
            (ev + 0.18, icon_y - icon_h),
            0.12,
            icon_h,
            boxstyle="round,pad=0,rounding_size=0.03",
            facecolor=BORDER,
            edgecolor="none",
            zorder=2,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (ev + 0.18, icon_y - icon_h),
            0.12,
            0.78 * icon_h,  # a good but not perfect score
            boxstyle="round,pad=0,rounding_size=0.03",
            facecolor=b,
            edgecolor="none",
            zorder=3,
        )
    )
    # Select: the tournament winner at full strength, the others faded.
    draw_tree(ax, small, sel - 0.52, icon_y, 0.3, icon_h, alpha=0.3)
    draw_tree(ax, deeper, sel, icon_y, 0.46, icon_h)
    draw_tree(ax, (o, (b, o)), sel + 0.52, icon_y, 0.3, icon_h, alpha=0.3)
    # Crossover: each parent carries a subtree in the other's colours.
    draw_tree(ax, ((b, b), (o, o)), cx_ - 0.4, icon_y, 0.44, icon_h)
    draw_tree(ax, ((o, o), (b, b)), cx_ + 0.4, icon_y, 0.44, icon_h)
    ax.text(cx_, icon_y - icon_h / 2, "\u2194", ha="center", va="center", color=MUTED, fontsize=13)
    # Mutate: one split changed, ringed in orange.
    draw_tree(ax, ((b, (o, b, o)), (o, b)), mut, icon_y, 0.56, icon_h)

    arrow = dict(arrowstyle="-|>", mutation_scale=14, color=MUTED, linewidth=1.4)
    for left, right in zip(centres, centres[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (left + width / 2 + 0.05, row_y), (right - width / 2 - 0.05, row_y), **arrow
            )
        )

    # Offspring go back to Evaluate: the generation loop.
    ax.add_patch(
        FancyArrowPatch(
            (mut, bottom - 0.05),
            (ev, bottom - 0.05),
            connectionstyle="arc3,rad=-0.25",
            arrowstyle="-|>",
            mutation_scale=15,
            color=b,
            linewidth=2,
        )
    )
    ax.text(
        (ev + mut) / 2,
        0.55,
        "next generation",
        ha="center",
        va="center",
        color=SECONDARY,
        fontsize=9.5,
    )

    # After the last generation the best tree comes out of Evaluate, straight up into
    # its own card, so the arrow and its label can't drift apart.
    card_w, card_bottom, card_top = 2.6, 3.65, 4.75
    ax.add_patch(
        FancyArrowPatch(
            (ev, top + 0.05),
            (ev, card_bottom - 0.04),
            arrowstyle="-|>",
            mutation_scale=15,
            color=o,
            linewidth=2,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (ev - card_w / 2, card_bottom),
            card_w,
            card_top - card_bottom,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=CARD,
            edgecolor=o,
            linewidth=1.6,
        )
    )
    draw_tree(ax, deeper, ev - 0.82, card_top - 0.24, 0.5, 0.6)
    ax.text(
        ev - 0.42,
        card_top - 0.38,
        "Best tree",
        ha="left",
        va="center",
        color=INK,
        fontsize=11.5,
        fontweight="bold",
    )
    ax.text(
        ev - 0.42,
        card_bottom + 0.3,
        "after the last\ngeneration",
        ha="left",
        va="center",
        color=SECONDARY,
        fontsize=8.6,
        linespacing=1.3,
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
