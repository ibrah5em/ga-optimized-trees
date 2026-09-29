#!/usr/bin/env python
"""How often evolved trees break min_samples_split / min_samples_leaf.

Counts, for every tree the searcher *evaluates* on one outer fold, the internal
nodes whose split the sample-count constraints forbid (``count_violations``),
and the same for the delivered front. Used to size Phase 2 item 4.

    python scripts/measure_constraint_violations.py wdbc vehicle --output out.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ga_trees.benchmark import frontiers  # noqa: E402
from ga_trees.ga.repair import count_violations  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper.yaml"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from experiment import load_dataset

    config = yaml.safe_load(open(args.config))
    seen = []
    original = frontiers._CountingObjective.__call__

    def recording(self, tree, X, y):
        seen.append((count_violations(tree, X), (tree.get_num_nodes() - 1) // 2))
        return original(self, tree, X, y)

    frontiers._CountingObjective.__call__ = recording
    rows = []
    for name in args.datasets:
        X, y = load_dataset(name)
        train, _ = next(StratifiedKFold(10, shuffle=True, random_state=0).split(X, y))
        ga = frontiers.ParetoGAFrontier(config["ga"], config["tree"], config["fitness"])
        rs = frontiers.RandomSearchFrontier(config["ga"], config["tree"], config["fitness"])
        for label, method in (("GA (NSGA-II)", ga), ("Random Search", rs)):
            seen.clear()
            if method is rs:
                rs.budget = budget
            delivered, spent = method.build(X[train], y[train], seed=1)
            if method is ga:
                budget = spent
            evaluated = np.array(seen)
            delivered_v = [count_violations(t, X[train]) for t in delivered]
            rows.append(
                {
                    "dataset": name,
                    "method": label,
                    "evaluated_trees": len(evaluated),
                    "evaluated_share_violating": float((evaluated[:, 0] > 0).mean()),
                    "evaluated_internal_share_violating": float(
                        evaluated[:, 0].sum() / max(1, evaluated[:, 1].sum())
                    ),
                    "delivered_trees": len(delivered),
                    "delivered_violating_nodes": int(sum(delivered_v)),
                    "delivered_internal_nodes": int(
                        sum((t.get_num_nodes() - 1) // 2 for t in delivered)
                    ),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
