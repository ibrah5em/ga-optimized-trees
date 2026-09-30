"""Deterministic seed derivation for reproducible experiments.

Every reported number must be traceable to the seed that produced it. A single
experiment-wide seed is not enough for cross-validated runs: ``GAEngine.evolve``
seeds ``random`` and ``numpy.random`` globally when ``GAConfig.random_state`` is
set, so passing the same value to every fold makes each fold repeat the same
search. Folds need seeds that differ from one another but are fixed across
invocations.

That is what :func:`derive_fold_seed` provides — a pure function of
``(base_seed, dataset, method, fold)``.
"""

import hashlib
from typing import Dict, Iterable, List

# numpy.random.seed accepts [0, 2**32 - 1]. Staying inside signed 32-bit range
# keeps derived seeds portable to RNG APIs that are stricter than numpy's.
MAX_SEED = 2**31 - 1

DEFAULT_METHOD = "ga"


def derive_fold_seed(
    base_seed: int, dataset_name: str, fold: int, method: str = DEFAULT_METHOD
) -> int:
    """Derive a stable, distinct seed for one (dataset, method, fold) cell.

    Uses blake2b rather than the built-in ``hash()``. Python salts string
    hashing per process unless ``PYTHONHASHSEED`` is pinned, so a ``hash()``
    based derivation would silently produce different seeds on every
    invocation — precisely the failure this function exists to prevent.

    Including ``method`` keeps a baseline from accidentally sharing the GA's
    random stream on the same fold, which would couple their results.

    Parameters
    ----------
    base_seed : int
        Experiment-wide seed, normally ``config["experiment"]["random_state"]``.
    dataset_name : str
        Dataset identifier, so the same fold index differs across datasets.
    fold : int
        Fold number.
    method : str, optional
        Method identifier, e.g. ``"ga"`` or ``"random_search"``.

    Returns
    -------
    int
        A seed in ``[0, MAX_SEED)``.

    Raises
    ------
    ValueError
        If ``base_seed`` or ``fold`` is negative, or ``dataset_name`` is empty.
    """
    if base_seed < 0:
        raise ValueError(f"base_seed must be >= 0, got {base_seed}.")
    if fold < 0:
        raise ValueError(f"fold must be >= 0, got {fold}.")
    if not dataset_name:
        raise ValueError("dataset_name must be a non-empty string.")

    key = "{}|{}|{}|{}".format(base_seed, dataset_name, method, fold).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") % MAX_SEED


def build_seed_manifest(
    base_seed: int,
    dataset_names: Iterable[str],
    n_folds: int,
    methods: Iterable[str] = (DEFAULT_METHOD,),
) -> Dict[str, object]:
    """Build the full seed manifest for an experiment, for writing to JSON.

    Emitting this alongside results is what makes "seeded" a checkable claim
    rather than an assertion: the manifest is regenerable from ``base_seed``
    alone, so a reader can confirm the seeds actually used.
    """
    if n_folds <= 0:
        raise ValueError(f"n_folds must be > 0, got {n_folds}.")

    folds: Dict[str, Dict[str, List[int]]] = {}
    for dataset_name in dataset_names:
        folds[dataset_name] = {
            method: [
                derive_fold_seed(base_seed, dataset_name, fold, method)
                for fold in range(1, n_folds + 1)
            ]
            for method in methods
        }

    derivation = (
        "blake2b('{base_seed}|{dataset}|{method}|{fold}'.encode(), digest_size=8) " "% (2**31 - 1)"
    )
    return {
        "base_seed": base_seed,
        "n_folds": n_folds,
        "derivation": derivation,
        "folds": folds,
    }
