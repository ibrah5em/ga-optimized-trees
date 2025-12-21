import os

from ga_trees.configs.bayesian_config import BayesianConfig


def test_from_dict_and_validate():
    cfg = {"n_samples": 100, "confidence_level": 0.9, "dirichlet_prior_alpha": [1.0, 1.0]}
    bc = BayesianConfig.from_dict(cfg)
    ok, errs = bc.validate(n_classes=2)
    assert ok, errs


def test_from_file_default(tmp_path):
    # Use the repo example config file
    here = os.path.join(os.getcwd(), "configs", "bayesian_default.yaml")
    bc = BayesianConfig.from_file(here)
    ok, errs = bc.validate()
    assert ok, errs
