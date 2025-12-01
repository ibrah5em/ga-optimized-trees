"""
Evolved Bayesian Decision Trees (E-BDT) - Core Genotype Extension

This module extends the existing TreeGenotype with Bayesian capabilities:
- Probabilistic thresholds with uncertainty quantification
- Dirichlet-distributed leaf predictions
- Confidence intervals for all decisions
- Full backward compatibility with deterministic mode

Author: GA-Trees Framework
Version: 2.0 (Bayesian Extension)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Literal, Dict
import numpy as np
import copy
from scipy import stats
from scipy.special import digamma

# Import base classes
from ga_trees.genotype.tree_genotype import (
    Node as BaseNode, 
    TreeGenotype as BaseTreeGenotype
)


@dataclass
class BayesianNode(BaseNode):
    """
    Extended Node class with Bayesian probabilistic parameters.
    
    Fully backward compatible - behaves like BaseNode when bayesian=False.
    """
    
    # Bayesian mode flag
    bayesian_mode: bool = False
    
    # === THRESHOLD DISTRIBUTION PARAMETERS ===
    # For internal nodes: P(threshold | data)
    threshold_mean: Optional[float] = None
    threshold_std: Optional[float] = None
    threshold_dist_type: Literal['normal', 'laplace', 'cauchy'] = 'normal'
    
    # Confidence in threshold placement
    threshold_confidence: float = 1.0  # [0, 1] - higher = more certain
    
    # === LEAF DISTRIBUTION PARAMETERS ===
    # For leaf nodes: P(class | data) ~ Dirichlet(alpha)
    leaf_alpha: Optional[np.ndarray] = None  # Dirichlet concentration params
    leaf_samples_count: int = 0  # Number of training samples
    
    # Posterior predictive distribution
    leaf_class_probs: Optional[np.ndarray] = None  # Expected class probabilities
    leaf_class_uncertainty: Optional[np.ndarray] = None  # Uncertainty per class
    
    # === UNCERTAINTY METRICS ===
    # Total entropy of predictions
    prediction_entropy: float = 0.0
    
    # Aleatoric uncertainty (data noise)
    aleatoric_uncertainty: float = 0.0
    
    # Epistemic uncertainty (model uncertainty)
    epistemic_uncertainty: float = 0.0
    
    def __post_init__(self):
        """Initialize Bayesian parameters if needed."""
        super().__post_init__() if hasattr(super(), '__post_init__') else None
        
        if self.bayesian_mode:
            # Initialize Bayesian parameters for internal nodes
            if self.is_internal():
                if self.threshold_mean is None and self.threshold is not None:
                    self.threshold_mean = self.threshold
                    self.threshold_std = 0.1 * abs(self.threshold)  # 10% default uncertainty
            
            # Initialize Bayesian parameters for leaf nodes
            elif self.is_leaf() and self.prediction is not None:
                if self.leaf_alpha is None:
                    # Initialize with uniform prior + observations
                    n_classes = 3 if isinstance(self.prediction, (int, float)) else len(self.prediction)
                    self.leaf_alpha = np.ones(n_classes)
                    if isinstance(self.prediction, int):
                        self.leaf_alpha[self.prediction] += 1
    
    def get_threshold_sample(self, n_samples: int = 1) -> Union[float, np.ndarray]:
        """
        Sample threshold from posterior distribution.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Sampled threshold(s)
        """
        if not self.bayesian_mode or not self.is_internal():
            return self.threshold
        
        if self.threshold_dist_type == 'normal':
            samples = np.random.normal(
                self.threshold_mean, 
                self.threshold_std, 
                size=n_samples
            )
        elif self.threshold_dist_type == 'laplace':
            samples = np.random.laplace(
                self.threshold_mean,
                self.threshold_std,
                size=n_samples
            )
        elif self.threshold_dist_type == 'cauchy':
            samples = np.random.standard_cauchy(size=n_samples) * self.threshold_std + self.threshold_mean
        else:
            samples = np.full(n_samples, self.threshold_mean)
        
        return samples[0] if n_samples == 1 else samples
    
    def get_threshold_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute confidence interval for threshold.
        
        Args:
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            (lower_bound, upper_bound)
        """
        if not self.bayesian_mode or not self.is_internal():
            return (self.threshold, self.threshold)
        
        if self.threshold_dist_type == 'normal':
            z = stats.norm.ppf((1 + confidence) / 2)
            margin = z * self.threshold_std
        elif self.threshold_dist_type == 'laplace':
            # Laplace quantiles
            q_lower = (1 - confidence) / 2
            q_upper = (1 + confidence) / 2
            margin = self.threshold_std * np.log(2 / (1 - confidence))
        else:
            # Conservative estimate
            margin = 2 * self.threshold_std
        
        return (
            self.threshold_mean - margin,
            self.threshold_mean + margin
        )
    
    def update_leaf_distribution(self, y_samples: np.ndarray):
        """
        Update leaf Dirichlet distribution with new observations.
        
        Args:
            y_samples: Class labels of samples reaching this leaf
        """
        if not self.bayesian_mode or not self.is_leaf():
            return
        
        self.leaf_samples_count = len(y_samples)
        
        if self.leaf_alpha is None:
            # Initialize with uniform prior
            n_classes = len(np.unique(y_samples))
            self.leaf_alpha = np.ones(n_classes)
        
        # Bayesian update: posterior = prior + counts
        unique, counts = np.unique(y_samples, return_counts=True)
        for cls, count in zip(unique, counts):
            if cls < len(self.leaf_alpha):
                self.leaf_alpha[int(cls)] += count
        
        # Update expected probabilities
        self._update_leaf_statistics()
    
    def _update_leaf_statistics(self):
        """Compute statistics from Dirichlet posterior."""
        if self.leaf_alpha is None:
            return
        
        alpha_0 = np.sum(self.leaf_alpha)
        
        # Expected class probabilities (Dirichlet mean)
        self.leaf_class_probs = self.leaf_alpha / alpha_0
        
        # Uncertainty per class (Dirichlet variance)
        self.leaf_class_uncertainty = np.sqrt(
            (self.leaf_alpha * (alpha_0 - self.leaf_alpha)) / 
            (alpha_0**2 * (alpha_0 + 1))
        )
        
        # Prediction entropy (measure of uncertainty)
        probs = self.leaf_class_probs + 1e-10  # Avoid log(0)
        self.prediction_entropy = -np.sum(probs * np.log(probs))
        
        # Aleatoric uncertainty (inherent data randomness)
        # Higher when probabilities are balanced
        self.aleatoric_uncertainty = self.prediction_entropy
        
        # Epistemic uncertainty (lack of data)
        # Higher when we have fewer samples
        # Using expected KL divergence from uniform
        n_classes = len(self.leaf_alpha)
        self.epistemic_uncertainty = np.log(n_classes) - (
            np.sum(digamma(self.leaf_alpha + 1) - digamma(alpha_0 + 1))
        )
        self.epistemic_uncertainty = max(0, self.epistemic_uncertainty)
    
    def get_prediction_with_confidence(self) -> Dict[str, Union[int, float, np.ndarray]]:
        """
        Get prediction with confidence metrics.
        
        Returns:
            Dictionary with:
                - prediction: Most likely class
                - confidence: Probability of predicted class
                - class_probs: Full probability distribution
                - uncertainty: Total uncertainty
                - ci_lower/ci_upper: Confidence intervals per class
        """
        if not self.bayesian_mode or not self.is_leaf():
            return {
                'prediction': self.prediction,
                'confidence': 1.0,
                'class_probs': None,
                'uncertainty': 0.0
            }
        
        if self.leaf_class_probs is None:
            self._update_leaf_statistics()
        
        pred_class = int(np.argmax(self.leaf_class_probs))
        
        # Confidence intervals using Beta distribution approximation
        # For each class, approximate marginal as Beta(alpha_i, sum(alpha_-i))
        ci_lower = []
        ci_upper = []
        for i, alpha_i in enumerate(self.leaf_alpha):
            alpha_others = np.sum(self.leaf_alpha) - alpha_i
            # 95% credible interval
            ci_lower.append(stats.beta.ppf(0.025, alpha_i, alpha_others))
            ci_upper.append(stats.beta.ppf(0.975, alpha_i, alpha_others))
        
        return {
            'prediction': pred_class,
            'confidence': float(self.leaf_class_probs[pred_class]),
            'class_probs': self.leaf_class_probs.copy(),
            'class_uncertainty': self.leaf_class_uncertainty.copy(),
            'uncertainty': self.prediction_entropy,
            'aleatoric': self.aleatoric_uncertainty,
            'epistemic': self.epistemic_uncertainty,
            'ci_lower': np.array(ci_lower),
            'ci_upper': np.array(ci_upper),
            'n_samples': self.leaf_samples_count
        }
    
    def sample_threshold(self, n_samples: int = 1, method: str = 'posterior') -> Union[float, np.ndarray]:
        """
        Sample thresholds from distribution for Monte Carlo predictions.
        
        Supports multiple sampling strategies:
        - 'posterior': Sample from learned posterior distribution
        - 'prior': Sample from prior (for sensitivity analysis)
        - 'map': Return MAP estimate (threshold_mean)
        
        Args:
            n_samples: Number of threshold samples
            method: Sampling method ('posterior', 'prior', 'map')
            
        Returns:
            Sampled threshold(s)
            
        Examples:
            >>> node = BayesianNode(bayesian_mode=True, threshold_mean=0.5, threshold_std=0.1)
            >>> samples = node.sample_threshold(n_samples=100)
            >>> print(f"Mean: {samples.mean():.3f}, Std: {samples.std():.3f}")
        """
        if not self.bayesian_mode or not self.is_internal():
            return self.threshold
        
        if method == 'map':
            # Maximum A Posteriori estimate
            return self.threshold_mean if n_samples == 1 else np.full(n_samples, self.threshold_mean)
        
        elif method == 'prior':
            # Sample from prior (less informative)
            std = self.threshold_prior_std if hasattr(self, 'threshold_prior_std') else self.threshold_std * 2
            if self.threshold_dist_type == 'normal':
                return np.random.normal(self.threshold_mean, std, size=n_samples)
            else:
                return self.get_threshold_sample(n_samples)
        
        else:  # 'posterior'
            # Sample from posterior distribution
            return self.get_threshold_sample(n_samples)
    
    def sample_leaf_distribution(self, n_samples: int = 1, 
                                 return_probs: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Sample class probability vectors from Dirichlet posterior.
        
        This is the core method for uncertainty quantification in leaves.
        
        Args:
            n_samples: Number of probability vectors to sample
            return_probs: If True, return both predictions AND probability vectors
            
        Returns:
            - If return_probs=False: Array of sampled class labels [n_samples]
            - If return_probs=True: Tuple of (predictions, probability_vectors)
              where probability_vectors has shape [n_samples, n_classes]
        
        Examples:
            >>> node = BayesianNode(bayesian_mode=True)
            >>> node.leaf_alpha = np.array([5, 3, 2])  # More evidence for class 0
            >>> predictions, probs = node.sample_leaf_distribution(100, return_probs=True)
            >>> print(f"Class 0 probability: {probs[:, 0].mean():.3f}")
            >>> print(f"Most frequent prediction: {np.bincount(predictions).argmax()}")
        """
        if not self.bayesian_mode or self.leaf_alpha is None:
            pred = self.prediction if isinstance(self.prediction, int) else np.argmax(self.prediction)
            if return_probs:
                probs = np.zeros((n_samples, len(self.leaf_alpha) if self.leaf_alpha is not None else 2))
                probs[:, pred] = 1.0
                return np.full(n_samples, pred), probs
            return np.full(n_samples, pred)
        
        # Sample probability vectors from Dirichlet posterior
        prob_samples = np.random.dirichlet(self.leaf_alpha, size=n_samples)
        
        # Sample class predictions from each probability vector
        predictions = np.array([
            np.random.choice(len(probs), p=probs) 
            for probs in prob_samples
        ])
        
        if return_probs:
            return predictions, prob_samples
        return predictions
    
    def get_soft_decision_prob(self, feature_value: float, 
                               n_samples: int = 1000) -> Dict[str, float]:
        """
        Compute probabilistic routing instead of hard left/right splits.
        
        Instead of deterministic routing (feature <= threshold),
        compute P(go_left | feature_value, threshold_distribution).
        
        This enables soft decision trees where samples are probabilistically
        routed through multiple paths.
        
        Args:
            feature_value: Value of the split feature
            n_samples: Monte Carlo samples for probability estimation
            
        Returns:
            Dictionary with:
                - 'prob_left': P(route to left child)
                - 'prob_right': P(route to right child)
                - 'entropy': Uncertainty in routing decision
                - 'confidence': 1 - entropy (normalized)
        
        Examples:
            >>> node = BayesianNode(bayesian_mode=True, feature_idx=0,
            ...                     threshold_mean=0.5, threshold_std=0.1)
            >>> # Feature value close to threshold = high uncertainty
            >>> probs = node.get_soft_decision_prob(0.51)
            >>> print(f"P(left)={probs['prob_left']:.2f}, entropy={probs['entropy']:.3f}")
        """
        if not self.bayesian_mode or not self.is_internal():
            # Deterministic fallback
            goes_left = feature_value <= self.threshold
            return {
                'prob_left': 1.0 if goes_left else 0.0,
                'prob_right': 0.0 if goes_left else 1.0,
                'entropy': 0.0,
                'confidence': 1.0
            }
        
        # Sample thresholds from posterior
        threshold_samples = self.sample_threshold(n_samples=n_samples, method='posterior')
        
        # Compute routing probabilities via Monte Carlo
        goes_left = (feature_value <= threshold_samples).astype(float)
        prob_left = np.mean(goes_left)
        prob_right = 1.0 - prob_left
        
        # Compute routing uncertainty (entropy)
        if prob_left > 0 and prob_right > 0:
            routing_entropy = -(
                prob_left * np.log2(prob_left) + 
                prob_right * np.log2(prob_right)
            )
        else:
            routing_entropy = 0.0
        
        # Confidence = how certain we are about routing (inverse of entropy)
        max_entropy = 1.0  # Maximum entropy for binary decision
        confidence = 1.0 - (routing_entropy / max_entropy)
        
        return {
            'prob_left': float(prob_left),
            'prob_right': float(prob_right),
            'entropy': float(routing_entropy),
            'confidence': float(confidence),
            'threshold_mean': float(self.threshold_mean),
            'threshold_std': float(self.threshold_std)
        }
    
    def sample_prediction(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample predictions from Dirichlet posterior.
        
        Alias for sample_leaf_distribution for backward compatibility.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Array of sampled class labels
        """
        return self.sample_leaf_distribution(n_samples=n_samples, return_probs=False)
    
    def to_deterministic(self) -> BaseNode:
        """
        Convert Bayesian node to deterministic node.
        
        Returns:
            BaseNode with deterministic parameters
        """
        det_node = BaseNode(
            node_type=self.node_type,
            feature_idx=self.feature_idx,
            threshold=self.threshold_mean if self.bayesian_mode else self.threshold,
            operator=self.operator,
            left_child=None,  # Will be set later
            right_child=None,
            prediction=int(np.argmax(self.leaf_class_probs)) if (
                self.bayesian_mode and self.leaf_class_probs is not None
            ) else self.prediction,
            depth=self.depth,
            node_id=self.node_id,
            samples_count=self.samples_count
        )
        
        # Recursively convert children
        if self.left_child:
            det_node.left_child = self.left_child.to_deterministic() if isinstance(
                self.left_child, BayesianNode
            ) else self.left_child
        if self.right_child:
            det_node.right_child = self.right_child.to_deterministic() if isinstance(
                self.right_child, BayesianNode
            ) else self.right_child
        
        return det_node
    
    def copy(self) -> 'BayesianNode':
        """Create deep copy of Bayesian node."""
        return copy.deepcopy(self)


@dataclass
class BayesianTreeGenotype(BaseTreeGenotype):
    """
    Extended TreeGenotype with Bayesian capabilities.
    
    Fully backward compatible with deterministic trees.
    
    Attributes:
        mode: Operation mode ('deterministic' or 'bayesian')
        bayesian_config: Bayesian-specific hyperparameters
        mean_calibration_error: Probability reliability metric
        prediction_uncertainty: Quantified confidence intervals
    """
    
    # === MODE CONFIGURATION ===
    mode: Literal['deterministic', 'bayesian'] = 'deterministic'
    bayesian_mode: bool = False  # Kept for backward compatibility
    
    # === BAYESIAN CONFIGURATION ===
    bayesian_config: Dict[str, Any] = field(default_factory=lambda: {
        # Prior parameters
        'threshold_prior_std': 0.1,
        'leaf_prior_alpha': 1.0,
        'threshold_dist_type': 'normal',
        
        # Uncertainty quantification
        'n_mc_samples': 100,
        'confidence_threshold': 0.8,
        'uncertainty_threshold': 0.5,
        
        # Soft decision trees
        'enable_soft_routing': False,
        'soft_routing_samples': 1000,
        
        # Calibration
        'temperature': 1.0,
        'calibration_method': 'temperature_scaling',
        
        # Risk sensitivity
        'risk_averse': False,
        'epistemic_penalty': 0.05,
    })
    
    # Prior parameters (kept for backward compatibility)
    threshold_prior_std: float = 0.1
    leaf_prior_alpha: float = 1.0
    
    # === EVALUATION METRICS ===
    # Calibration metrics
    mean_calibration_error: float = 0.0  # Expected Calibration Error (ECE)
    max_calibration_error: float = 0.0   # Maximum Calibration Error (MCE)
    brier_score: float = 0.0             # Brier score (lower is better)
    
    # Prediction uncertainty metrics
    prediction_uncertainty: Dict[str, float] = field(default_factory=lambda: {
        'mean_entropy': 0.0,
        'mean_aleatoric': 0.0,
        'mean_epistemic': 0.0,
        'mean_confidence': 0.0,
        'fraction_uncertain': 0.0,
    })
    
    # Uncertainty-aware fitness components
    uncertainty_penalty: float = 0.0
    confidence_weight: float = 0.0
    
    # Calibration parameters
    temperature: float = 1.0
    
    def __post_init__(self):
        """Initialize Bayesian tree with proper mode handling."""
        super().__post_init__()
        
        # Sync mode and bayesian_mode
        if self.mode == 'bayesian':
            self.bayesian_mode = True
        elif self.bayesian_mode:
            self.mode = 'bayesian'
        
        # Sync config with direct attributes
        if self.bayesian_mode:
            self.bayesian_config['threshold_prior_std'] = self.threshold_prior_std
            self.bayesian_config['leaf_prior_alpha'] = self.leaf_prior_alpha
            self.bayesian_config['temperature'] = self.temperature
            
            # Convert all nodes to BayesianNodes if needed
            self.root = self._convert_to_bayesian_nodes(self.root)
    
    def _convert_to_bayesian_nodes(self, node: BaseNode) -> BayesianNode:
        """Recursively convert tree to Bayesian nodes."""
        if isinstance(node, BayesianNode):
            return node
        
        # Create Bayesian node from base node
        bay_node = BayesianNode(
            bayesian_mode=True,
            node_type=node.node_type,
            feature_idx=node.feature_idx,
            threshold=node.threshold,
            threshold_mean=node.threshold,
            threshold_std=self.threshold_prior_std * abs(node.threshold) if node.threshold else 0.1,
            operator=node.operator,
            prediction=node.prediction,
            depth=node.depth,
            node_id=node.node_id,
            samples_count=node.samples_count
        )
        
        # Initialize leaf distributions
        if bay_node.is_leaf() and bay_node.prediction is not None:
            bay_node.leaf_alpha = np.ones(self.n_classes) * self.leaf_prior_alpha
            if isinstance(bay_node.prediction, int):
                bay_node.leaf_alpha[bay_node.prediction] += 1
            bay_node._update_leaf_statistics()
        
        # Recursively convert children
        if node.left_child:
            bay_node.left_child = self._convert_to_bayesian_nodes(node.left_child)
        if node.right_child:
            bay_node.right_child = self._convert_to_bayesian_nodes(node.right_child)
        
        return bay_node
    
    def fit_bayesian_parameters(self, X: np.ndarray, y: np.ndarray):
        """
        Fit Bayesian parameters to data.
        
        Updates threshold distributions and leaf Dirichlet parameters
        based on training data.
        
        Args:
            X: Training features
            y: Training labels
        """
        if not self.bayesian_mode:
            return
        
        # Recursively fit parameters
        self._fit_node_bayesian(self.root, X, y)
    
    def _fit_node_bayesian(self, node: BayesianNode, X: np.ndarray, y: np.ndarray):
        """Recursively fit Bayesian parameters for each node."""
        if node is None or len(X) == 0:
            return
        
        node.samples_count = len(X)
        
        if node.is_leaf():
            # Update leaf Dirichlet distribution
            node.update_leaf_distribution(y)
        else:
            # Update threshold distribution based on local data
            feature_vals = X[:, node.feature_idx]
            
            # Estimate optimal threshold distribution
            # Use empirical split quality to estimate uncertainty
            candidate_thresholds = np.percentile(feature_vals, [25, 50, 75])
            
            # Compute information gain for each candidate
            gains = []
            for thresh in candidate_thresholds:
                left_mask = feature_vals <= thresh
                if np.sum(left_mask) > 0 and np.sum(~left_mask) > 0:
                    left_entropy = self._entropy(y[left_mask])
                    right_entropy = self._entropy(y[~left_mask])
                    p_left = np.sum(left_mask) / len(y)
                    gain = self._entropy(y) - (p_left * left_entropy + (1-p_left) * right_entropy)
                    gains.append(gain)
                else:
                    gains.append(0)
            
            if gains:
                best_idx = np.argmax(gains)
                node.threshold_mean = candidate_thresholds[best_idx]
                
                # Uncertainty based on gain variance
                gain_std = np.std(gains) if len(gains) > 1 else 0.1
                node.threshold_std = self.threshold_prior_std * (1 + gain_std)
                node.threshold_confidence = max(gains) / (self._entropy(y) + 1e-10)
            
            # Recursively fit children
            left_mask = X[:, node.feature_idx] <= node.threshold_mean
            if node.left_child and np.sum(left_mask) > 0:
                self._fit_node_bayesian(node.left_child, X[left_mask], y[left_mask])
            
            if node.right_child and np.sum(~left_mask) > 0:
                self._fit_node_bayesian(node.right_child, X[~left_mask], y[~left_mask])
    
    def _entropy(self, y: np.ndarray) -> float:
        """Compute entropy of labels."""
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log(probs + 1e-10))
    
    def predict_with_uncertainty(self, X: np.ndarray, 
                                 n_samples: int = 100,
                                 use_soft_routing: bool = False) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty quantification using Monte Carlo.
        
        Supports two routing strategies:
        - Hard routing (default): Deterministic path through tree
        - Soft routing: Probabilistic routing at each node
        
        Args:
            X: Test features
            n_samples: Number of Monte Carlo samples for uncertainty
            use_soft_routing: Enable probabilistic routing
            
        Returns:
            Dictionary with:
                - predictions: Point predictions
                - confidences: Confidence per prediction
                - class_probs: Probability distribution per sample
                - uncertainties: Total uncertainty per sample
                - aleatoric: Data uncertainty
                - epistemic: Model uncertainty
                - prediction_samples: All MC samples (if n_samples > 1)
        """
        n_test = len(X)
        
        # Point predictions
        predictions = np.zeros(n_test, dtype=int)
        confidences = np.zeros(n_test)
        class_probs = np.zeros((n_test, self.n_classes))
        uncertainties = np.zeros(n_test)
        aleatoric = np.zeros(n_test)
        epistemic = np.zeros(n_test)
        
        # Monte Carlo samples for each test instance
        all_samples = np.zeros((n_test, n_samples), dtype=int) if n_samples > 1 else None
        
        for i, x in enumerate(X):
            if use_soft_routing and self.bayesian_mode:
                # Soft routing: average predictions over all paths
                pred_info = self._soft_predict_single(x, n_samples)
            else:
                # Hard routing: single deterministic path
                leaf = self._find_leaf(self.root, x)
                
                if isinstance(leaf, BayesianNode) and self.bayesian_mode:
                    pred_info = leaf.get_prediction_with_confidence()
                    
                    # Monte Carlo samples from leaf distribution
                    if n_samples > 1:
                        samples = leaf.sample_leaf_distribution(n_samples)
                        if all_samples is not None:
                            all_samples[i] = samples
                else:
                    # Deterministic fallback
                    pred = leaf.prediction if isinstance(leaf.prediction, int) else np.argmax(leaf.prediction)
                    pred_info = {
                        'prediction': pred,
                        'confidence': 1.0,
                        'class_probs': np.zeros(self.n_classes),
                        'uncertainty': 0.0,
                        'aleatoric': 0.0,
                        'epistemic': 0.0
                    }
                    pred_info['class_probs'][pred] = 1.0
            
            predictions[i] = pred_info['prediction']
            confidences[i] = pred_info['confidence']
            class_probs[i] = pred_info['class_probs']
            uncertainties[i] = pred_info['uncertainty']
            aleatoric[i] = pred_info.get('aleatoric', 0)
            epistemic[i] = pred_info.get('epistemic', 0)
        
        result = {
            'predictions': predictions,
            'confidences': confidences,
            'class_probs': class_probs,
            'uncertainties': uncertainties,
            'aleatoric': aleatoric,
            'epistemic': epistemic
        }
        
        if all_samples is not None:
            result['prediction_samples'] = all_samples
        
        return result
    
    def _soft_predict_single(self, x: np.ndarray, n_samples: int) -> Dict[str, Any]:
        """
        Soft prediction for single instance using probabilistic routing.
        
        Instead of following one path, we compute weighted average over
        all possible paths, weighted by routing probabilities.
        
        Args:
            x: Single instance features
            n_samples: Monte Carlo samples
            
        Returns:
            Prediction info dictionary
        """
        # Collect predictions from all leaves with path probabilities
        leaf_predictions = []
        leaf_probs = []
        leaf_path_probs = []
        
        def traverse_soft(node: BayesianNode, path_prob: float = 1.0):
            """Recursively traverse with soft routing."""
            if node.is_leaf():
                # Reached a leaf - record prediction and path probability
                if node.leaf_class_probs is not None:
                    leaf_predictions.append(node.get_prediction_with_confidence())
                    leaf_probs.append(node.leaf_class_probs)
                    leaf_path_probs.append(path_prob)
                return
            
            # Get soft routing probabilities
            feat_val = x[node.feature_idx]
            routing = node.get_soft_decision_prob(feat_val)
            
            # Recursively traverse both children with updated path probabilities
            if node.left_child and routing['prob_left'] > 0.01:
                traverse_soft(node.left_child, path_prob * routing['prob_left'])
            if node.right_child and routing['prob_right'] > 0.01:
                traverse_soft(node.right_child, path_prob * routing['prob_right'])
        
        # Traverse tree with soft routing
        traverse_soft(self.root)
        
        if not leaf_predictions:
            # Fallback to hard routing
            return self._find_leaf(self.root, x).get_prediction_with_confidence()
        
        # Aggregate predictions weighted by path probabilities
        leaf_path_probs = np.array(leaf_path_probs)
        leaf_path_probs /= leaf_path_probs.sum()  # Normalize
        
        # Weighted average of class probabilities
        weighted_probs = np.zeros(self.n_classes)
        for prob_vec, weight in zip(leaf_probs, leaf_path_probs):
            weighted_probs += prob_vec * weight
        
        # Compute aggregate uncertainty
        # Aleatoric: weighted average of leaf uncertainties
        aleatoric_unc = sum(
            pred['aleatoric'] * weight 
            for pred, weight in zip(leaf_predictions, leaf_path_probs)
        )
        
        # Epistemic: disagreement between leaves
        epistemic_unc = sum(
            pred['epistemic'] * weight 
            for pred, weight in zip(leaf_predictions, leaf_path_probs)
        )
        
        # Total uncertainty from aggregated probabilities
        total_unc = -np.sum(weighted_probs * np.log(weighted_probs + 1e-10))
        
        return {
            'prediction': int(np.argmax(weighted_probs)),
            'confidence': float(np.max(weighted_probs)),
            'class_probs': weighted_probs,
            'uncertainty': float(total_unc),
            'aleatoric': float(aleatoric_unc),
            'epistemic': float(epistemic_unc),
            'n_paths': len(leaf_predictions)
        }
    
    def compute_calibration_metrics(self, X: np.ndarray, y: np.ndarray, 
                                    n_bins: int = 10) -> Dict[str, float]:
        """
        Compute comprehensive calibration metrics.
        
        Updates tree's calibration attributes:
        - mean_calibration_error (ECE)
        - max_calibration_error (MCE)
        - brier_score
        
        Args:
            X: Features
            y: True labels
            n_bins: Number of calibration bins
            
        Returns:
            Dictionary with all calibration metrics
        """
        if not self.bayesian_mode:
            return {
                'ece': 0.0,
                'mce': 0.0,
                'brier': 0.0,
                'reliability': 1.0
            }
        
        # Get predictions with probabilities
        pred_result = self.predict_with_uncertainty(X)
        y_probs = pred_result['class_probs']
        confidences = pred_result['confidences']
        predictions = pred_result['predictions']
        
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(y, y_probs, n_bins)
        self.mean_calibration_error = ece
        
        # Maximum Calibration Error (MCE)
        mce = self._compute_mce(y, y_probs, n_bins)
        self.max_calibration_error = mce
        
        # Brier Score
        y_one_hot = np.zeros_like(y_probs)
        y_one_hot[np.arange(len(y)), y] = 1
        brier = np.mean(np.sum((y_probs - y_one_hot)**2, axis=1))
        self.brier_score = brier
        
        # Reliability score (1 - ECE)
        reliability = 1.0 - ece
        
        return {
            'ece': ece,
            'mce': mce,
            'brier': brier,
            'reliability': reliability,
            'mean_confidence': float(np.mean(confidences))
        }
    
    def _compute_ece(self, y_true: np.ndarray, y_probs: np.ndarray, n_bins: int) -> float:
        """Compute Expected Calibration Error."""
        confidences = np.max(y_probs, axis=1)
        predictions = np.argmax(y_probs, axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
            if np.sum(in_bin) > 0:
                avg_conf = np.mean(confidences[in_bin])
                avg_acc = np.mean(accuracies[in_bin])
                bin_weight = np.sum(in_bin) / len(y_true)
                ece += bin_weight * abs(avg_conf - avg_acc)
        
        return ece
    
    def _compute_mce(self, y_true: np.ndarray, y_probs: np.ndarray, n_bins: int) -> float:
        """Compute Maximum Calibration Error."""
        confidences = np.max(y_probs, axis=1)
        predictions = np.argmax(y_probs, axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        bins = np.linspace(0, 1, n_bins + 1)
        max_error = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
            if np.sum(in_bin) > 0:
                avg_conf = np.mean(confidences[in_bin])
                avg_acc = np.mean(accuracies[in_bin])
                error = abs(avg_conf - avg_acc)
                max_error = max(max_error, error)
        
        return max_error
    
    def update_uncertainty_metrics(self, X: np.ndarray):
        """
        Update tree-level uncertainty metrics.
        
        Computes aggregate uncertainty statistics across all predictions.
        
        Args:
            X: Features for uncertainty estimation
        """
        if not self.bayesian_mode:
            return
        
        pred_result = self.predict_with_uncertainty(X)
        
        self.prediction_uncertainty = {
            'mean_entropy': float(np.mean(pred_result['uncertainties'])),
            'mean_aleatoric': float(np.mean(pred_result['aleatoric'])),
            'mean_epistemic': float(np.mean(pred_result['epistemic'])),
            'mean_confidence': float(np.mean(pred_result['confidences'])),
            'fraction_uncertain': float(
                np.mean(pred_result['uncertainties'] > 
                       self.bayesian_config.get('uncertainty_threshold', 0.5))
            ),
            'std_entropy': float(np.std(pred_result['uncertainties'])),
        }
    
    def _find_leaf(self, node: BayesianNode, x: np.ndarray) -> BayesianNode:
        """Navigate to leaf for instance."""
        if node.is_leaf():
            return node
        
        threshold = node.threshold_mean if self.bayesian_mode else node.threshold
        
        if x[node.feature_idx] <= threshold:
            return self._find_leaf(node.left_child, x)
        else:
            return self._find_leaf(node.right_child, x)
    
    def to_deterministic(self) -> BaseTreeGenotype:
        """
        Convert Bayesian tree to deterministic tree.
        
        Returns:
            BaseTreeGenotype with point estimates
        """
        det_root = self.root.to_deterministic() if isinstance(
            self.root, BayesianNode
        ) else self.root
        
        return BaseTreeGenotype(
            root=det_root,
            n_features=self.n_features,
            n_classes=self.n_classes,
            task_type=self.task_type,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features
        )
    
    def get_uncertainty_map(self) -> Dict[int, Dict[str, float]]:
        """
        Get uncertainty metrics for all nodes.
        
        Returns:
            Dictionary mapping node_id to uncertainty metrics
        """
        uncertainty_map = {}
        
        for node in self.get_all_nodes():
            if isinstance(node, BayesianNode) and self.bayesian_mode:
                if node.is_leaf():
                    uncertainty_map[node.node_id] = {
                        'type': 'leaf',
                        'entropy': node.prediction_entropy,
                        'aleatoric': node.aleatoric_uncertainty,
                        'epistemic': node.epistemic_uncertainty,
                        'n_samples': node.leaf_samples_count
                    }
                else:
                    ci_lower, ci_upper = node.get_threshold_confidence_interval()
                    uncertainty_map[node.node_id] = {
                        'type': 'internal',
                        'threshold_std': node.threshold_std,
                        'confidence': node.threshold_confidence,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper
                    }
        
        return uncertainty_map
    
    def validate_bayesian(self) -> Tuple[bool, List[str]]:
        """
        Validate Bayesian tree structural constraints.
        
        Checks Bayesian-specific requirements beyond standard tree validation:
        - All nodes have proper Bayesian parameters
        - Threshold distributions are valid
        - Leaf Dirichlet parameters are positive
        - Uncertainty metrics are computed
        - Mode consistency
        
        Returns:
            Tuple of (is_valid, list_of_errors)
            
        Examples:
            >>> tree = BayesianTreeGenotype(mode='bayesian', ...)
            >>> valid, errors = tree.validate_bayesian()
            >>> if not valid:
            ...     print("Errors:", errors)
        """
        errors = []
        
        # First run standard validation
        base_valid, base_errors = self.validate()
        if not base_valid:
            errors.extend(base_errors)
        
        # Check mode consistency
        if self.mode == 'bayesian' and not self.bayesian_mode:
            errors.append("Mode is 'bayesian' but bayesian_mode=False")
        if self.bayesian_mode and self.mode != 'bayesian':
            errors.append(f"bayesian_mode=True but mode='{self.mode}'")
        
        if not self.bayesian_mode:
            # If not in Bayesian mode, skip Bayesian checks
            return (len(errors) == 0, errors)
        
        # Check all nodes are BayesianNodes
        for node in self.get_all_nodes():
            if not isinstance(node, BayesianNode):
                errors.append(f"Node {node.node_id} is not a BayesianNode (mode is bayesian)")
            
            if not hasattr(node, 'bayesian_mode') or not node.bayesian_mode:
                errors.append(f"Node {node.node_id} has bayesian_mode=False")
        
        # Validate internal nodes
        for node in self.get_internal_nodes():
            if not isinstance(node, BayesianNode):
                continue
            
            # Check threshold distribution parameters
            if node.threshold_mean is None:
                errors.append(f"Internal node {node.node_id} missing threshold_mean")
            
            if node.threshold_std is None:
                errors.append(f"Internal node {node.node_id} missing threshold_std")
            elif node.threshold_std < 0:
                errors.append(f"Internal node {node.node_id} has negative threshold_std")
            elif node.threshold_std == 0:
                errors.append(f"Internal node {node.node_id} has zero threshold_std (no uncertainty)")
            
            # Check threshold confidence
            if not (0 <= node.threshold_confidence <= 1):
                errors.append(f"Internal node {node.node_id} threshold_confidence out of [0,1]")
            
            # Check distribution type
            if node.threshold_dist_type not in ['normal', 'laplace', 'cauchy']:
                errors.append(f"Internal node {node.node_id} has invalid threshold_dist_type")
        
        # Validate leaf nodes
        for node in self.get_all_leaves():
            if not isinstance(node, BayesianNode):
                continue
            
            # Check Dirichlet parameters
            if node.leaf_alpha is None:
                errors.append(f"Leaf node {node.node_id} missing leaf_alpha")
            else:
                if len(node.leaf_alpha) != self.n_classes:
                    errors.append(
                        f"Leaf node {node.node_id} leaf_alpha has wrong size "
                        f"(expected {self.n_classes}, got {len(node.leaf_alpha)})"
                    )
                
                if np.any(node.leaf_alpha <= 0):
                    errors.append(f"Leaf node {node.node_id} has non-positive leaf_alpha values")
                
                # Check if statistics are computed
                if node.leaf_class_probs is None:
                    errors.append(f"Leaf node {node.node_id} missing leaf_class_probs (call _update_leaf_statistics)")
            
            # Check uncertainty metrics
            if node.prediction_entropy < 0:
                errors.append(f"Leaf node {node.node_id} has negative prediction_entropy")
            
            if node.aleatoric_uncertainty < 0:
                errors.append(f"Leaf node {node.node_id} has negative aleatoric_uncertainty")
            
            if node.epistemic_uncertainty < 0:
                errors.append(f"Leaf node {node.node_id} has negative epistemic_uncertainty")
        
        # Check tree-level Bayesian config
        required_config_keys = [
            'threshold_prior_std', 'leaf_prior_alpha', 'n_mc_samples',
            'confidence_threshold', 'uncertainty_threshold'
        ]
        for key in required_config_keys:
            if key not in self.bayesian_config:
                errors.append(f"Missing required config key: {key}")
        
        # Validate config values
        if self.bayesian_config.get('threshold_prior_std', 0) <= 0:
            errors.append("threshold_prior_std must be positive")
        
        if self.bayesian_config.get('leaf_prior_alpha', 0) <= 0:
            errors.append("leaf_prior_alpha must be positive")
        
        if self.bayesian_config.get('n_mc_samples', 0) < 10:
            errors.append("n_mc_samples should be at least 10 for reliable estimates")
        
        return (len(errors) == 0, errors)
    
    def copy(self) -> 'BayesianTreeGenotype':
        """
        Create deep copy of Bayesian tree with proper parameter duplication.
        
        Ensures all Bayesian parameters are properly copied:
        - Threshold distributions
        - Dirichlet parameters
        - Uncertainty metrics
        - Configuration dictionaries
        
        Returns:
            Deep copy of the tree
        """
        # Use deepcopy to handle nested structures
        tree_copy = copy.deepcopy(self)
        
        # Explicitly copy mutable config dictionary
        tree_copy.bayesian_config = copy.deepcopy(self.bayesian_config)
        tree_copy.prediction_uncertainty = copy.deepcopy(self.prediction_uncertainty)
        
        # Recursively copy all node Bayesian parameters
        tree_copy.root = self._deep_copy_bayesian_node(self.root)
        
        return tree_copy
    
    def _deep_copy_bayesian_node(self, node: BayesianNode) -> BayesianNode:
        """Recursively deep copy Bayesian node with all parameters."""
        if node is None:
            return None
        
        # Create new node with copied parameters
        node_copy = copy.copy(node)  # Shallow copy first
        
        # Deep copy mutable attributes
        if node.leaf_alpha is not None:
            node_copy.leaf_alpha = node.leaf_alpha.copy()
        if node.leaf_class_probs is not None:
            node_copy.leaf_class_probs = node.leaf_class_probs.copy()
        if node.leaf_class_uncertainty is not None:
            node_copy.leaf_class_uncertainty = node.leaf_class_uncertainty.copy()
        if isinstance(node.prediction, np.ndarray):
            node_copy.prediction = node.prediction.copy()
        
        # Recursively copy children
        if node.left_child:
            node_copy.left_child = self._deep_copy_bayesian_node(node.left_child)
        if node.right_child:
            node_copy.right_child = self._deep_copy_bayesian_node(node.right_child)
        
        return node_copy


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_bayesian_leaf_node(
    prediction: int,
    n_classes: int,
    depth: int = 0,
    prior_alpha: float = 1.0
) -> BayesianNode:
    """Factory function for Bayesian leaf nodes."""
    node = BayesianNode(
        bayesian_mode=True,
        node_type='leaf',
        prediction=prediction,
        depth=depth,
        leaf_alpha=np.ones(n_classes) * prior_alpha
    )
    node.leaf_alpha[prediction] += 1
    node._update_leaf_statistics()
    return node


def create_bayesian_internal_node(
    feature_idx: int,
    threshold: float,
    left_child: BayesianNode,
    right_child: BayesianNode,
    depth: int = 0,
    threshold_std: float = 0.1
) -> BayesianNode:
    """Factory function for Bayesian internal nodes."""
    return BayesianNode(
        bayesian_mode=True,
        node_type='internal',
        feature_idx=feature_idx,
        threshold=threshold,
        threshold_mean=threshold,
        threshold_std=threshold_std * abs(threshold),
        operator='<=',
        left_child=left_child,
        right_child=right_child,
        depth=depth
    )