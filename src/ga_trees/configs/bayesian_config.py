"""Bayesian configuration dataclass and helpers.

Provides a centralized container for Bayesian hyperparameters and sampling
options used by E-BDT (Evolved Bayesian Decision Trees).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import yaml


@dataclass
class BayesianConfig:
    """Centralized Bayesian configuration container.

    Fields:
    - dirichlet_prior_alpha: Optional prior concentration vector for multi-class
      Dirichlet prior. If None, a symmetric prior (ones) may be used.
    - beta_prior_a, beta_prior_b: Beta prior params for binary classification.
    - n_samples: Monte Carlo samples for predictive uncertainty.
    - confidence_level: Interval coverage (e.g. 0.95).
    - calibration_weight: Weight in fitness for calibration (MCE).
    - confidence_weight: Weight in fitness for rewarding calibrated uncertainty.
    """

    dirichlet_prior_alpha: Optional[List[float]] = None
    beta_prior_a: Optional[float] = 1.0
    beta_prior_b: Optional[float] = 1.0

    # Sampling configuration
    n_samples: int = 200
    confidence_level: float = 0.95

    # Fitness optimization weights
    calibration_weight: float = 0.0
    confidence_weight: float = 0.0

    # Generic extra options
    extras: Dict[str, Any] = field(default_factory=dict)

    def validate(self, n_classes: Optional[int] = None) -> Tuple[bool, List[str]]:
        """Validate configuration consistency.

        Optionally accepts `n_classes` to validate prior shapes.
        Returns (is_valid, list_of_errors).
        """
        errors: List[str] = []

        if not isinstance(self.n_samples, int) or self.n_samples <= 0:
            errors.append("n_samples must be a positive integer")

        if not (0.0 < float(self.confidence_level) < 1.0):
            errors.append("confidence_level must be in (0, 1)")

        if float(self.calibration_weight) < 0.0:
            errors.append("calibration_weight must be >= 0")
        if float(self.confidence_weight) < 0.0:
            errors.append("confidence_weight must be >= 0")

        if n_classes is not None and self.dirichlet_prior_alpha is not None:
            try:
                if len(self.dirichlet_prior_alpha) != n_classes:
                    errors.append("dirichlet_prior_alpha length does not match n_classes")
                if any(a < 0 for a in self.dirichlet_prior_alpha):
                    errors.append("dirichlet_prior_alpha must have non-negative entries")
            except Exception:
                errors.append("dirichlet_prior_alpha is invalid")

        # Beta priors only meaningful for binary problems; warn if n_classes > 2
        if n_classes is not None and n_classes > 2 and (self.beta_prior_a is not None or self.beta_prior_b is not None):
            # not an error, but a mismatch
            errors.append("beta_prior parameters provided but n_classes > 2 (use dirichlet_prior_alpha)")

        return (len(errors) == 0, errors)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "BayesianConfig":
        """Create a BayesianConfig from a plain dict (e.g., parsed YAML/JSON)."""
        # Map keys forgivingly
        kwargs: Dict[str, Any] = {}
        if "dirichlet_prior_alpha" in cfg:
            kwargs["dirichlet_prior_alpha"] = list(cfg["dirichlet_prior_alpha"]) if cfg["dirichlet_prior_alpha"] is not None else None
        if "beta_prior_a" in cfg:
            kwargs["beta_prior_a"] = float(cfg["beta_prior_a"]) if cfg["beta_prior_a"] is not None else None
        if "beta_prior_b" in cfg:
            kwargs["beta_prior_b"] = float(cfg["beta_prior_b"]) if cfg["beta_prior_b"] is not None else None

        if "n_samples" in cfg:
            kwargs["n_samples"] = int(cfg["n_samples"])
        if "confidence_level" in cfg:
            kwargs["confidence_level"] = float(cfg["confidence_level"])

        if "calibration_weight" in cfg:
            kwargs["calibration_weight"] = float(cfg["calibration_weight"])
        if "confidence_weight" in cfg:
            kwargs["confidence_weight"] = float(cfg["confidence_weight"])

        # Capture any extra keys
        extras = {k: v for k, v in cfg.items() if k not in {
            "dirichlet_prior_alpha", "beta_prior_a", "beta_prior_b", "n_samples", "confidence_level", "calibration_weight", "confidence_weight"
        }}
        kwargs["extras"] = extras

        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str) -> "BayesianConfig":
        """Load configuration from a YAML or JSON file path."""
        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf8") as fh:
                cfg = json.load(fh)
        else:
            with open(path, "r", encoding="utf8") as fh:
                cfg = yaml.safe_load(fh)
        return cls.from_dict(cfg or {})

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dict for storage/inspection."""
        return {
            "dirichlet_prior_alpha": list(self.dirichlet_prior_alpha) if self.dirichlet_prior_alpha is not None else None,
            "beta_prior_a": self.beta_prior_a,
            "beta_prior_b": self.beta_prior_b,
            "n_samples": self.n_samples,
            "confidence_level": self.confidence_level,
            "calibration_weight": self.calibration_weight,
            "confidence_weight": self.confidence_weight,
            **self.extras,
        }
