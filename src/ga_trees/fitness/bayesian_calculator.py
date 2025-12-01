"""
Bayesian Fitness Calculator with Uncertainty-Aware Objectives

Extends existing fitness calculation with:
- Confidence-weighted accuracy
- Uncertainty penalties
- Calibration metrics
- Risk-sensitive evaluation

Backward compatible with deterministic fitness.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from scipy.stats import entropy

from ga_trees.fitness.calculator import (
    FitnessCalculator as BaseFitnessCalculator,
    TreePredictor,
    InterpretabilityCalculator
)


class BayesianTreePredictor(TreePredictor):
    """Extended predictor for Bayesian trees."""
    
    @staticmethod
    def predict_with_uncertainty(
        tree, 
        X: np.ndarray,
        return_probs: bool = True,
        n_mc_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Predict with uncertainty quantification.
        
        Args:
            tree: BayesianTreeGenotype
            X: Features
            return_probs: Whether to return full probability distributions
            n_mc_samples: Monte Carlo samples for uncertainty estimation
            
        Returns:
            Dictionary with predictions and uncertainty metrics
        """
        if hasattr(tree, 'predict_with_uncertainty'):
            return tree.predict_with_uncertainty(X, n_samples=n_mc_samples)
        else:
            # Fallback to deterministic prediction
            y_pred = TreePredictor.predict(tree, X)
            return {
                'predictions': y_pred,
                'confidences': np.ones(len(y_pred)),
                'class_probs': None,
                'uncertainties': np.zeros(len(y_pred))
            }
    
    @staticmethod
    def predict_proba(tree, X: np.ndarray) -> np.ndarray:
        """
        Get class probability predictions.
        
        Args:
            tree: Tree genotype
            X: Features
            
        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        result = BayesianTreePredictor.predict_with_uncertainty(tree, X)
        
        if result['class_probs'] is not None:
            return result['class_probs']
        else:
            # One-hot encode deterministic predictions
            predictions = result['predictions']
            n_samples = len(predictions)
            n_classes = len(np.unique(predictions))
            probs = np.zeros((n_samples, n_classes))
            probs[np.arange(n_samples), predictions.astype(int)] = 1.0
            return probs


class BayesianFitnessCalculator(BaseFitnessCalculator):
    """
    Extended fitness calculator with Bayesian objectives.
    
    Adds uncertainty-aware fitness components while maintaining
    backward compatibility with deterministic trees.
    """
    
    def __init__(
        self,
        mode: str = 'weighted_sum',
        accuracy_weight: float = 0.7,
        interpretability_weight: float = 0.2,
        uncertainty_weight: float = 0.1,
        interpretability_weights: Optional[Dict[str, float]] = None,
        # Bayesian-specific parameters
        confidence_reward: float = 0.1,
        uncertainty_penalty: float = 0.1,
        calibration_weight: float = 0.05,
        epistemic_penalty: float = 0.05,
        # Risk sensitivity
        risk_averse: bool = False,
        confidence_threshold: float = 0.8,
    ):
        """
        Initialize Bayesian fitness calculator.
        
        Args:
            mode: 'weighted_sum' or 'pareto'
            accuracy_weight: Weight for accuracy term
            interpretability_weight: Weight for interpretability
            uncertainty_weight: Weight for uncertainty penalty
            interpretability_weights: Sub-weights for interpretability
            confidence_reward: Bonus for high-confidence predictions
            uncertainty_penalty: Penalty for high uncertainty
            calibration_weight: Weight for probability calibration
            epistemic_penalty: Penalty for epistemic uncertainty
            risk_averse: If True, penalize uncertain predictions more
            confidence_threshold: Threshold for flagging low confidence
        """
        super().__init__(
            mode=mode,
            accuracy_weight=accuracy_weight,
            interpretability_weight=interpretability_weight,
            interpretability_weights=interpretability_weights
        )
        
        self.uncertainty_weight = uncertainty_weight
        self.confidence_reward = confidence_reward
        self.uncertainty_penalty = uncertainty_penalty
        self.calibration_weight = calibration_weight
        self.epistemic_penalty = epistemic_penalty
        self.risk_averse = risk_averse
        self.confidence_threshold = confidence_threshold
        
        self.bayesian_predictor = BayesianTreePredictor()
    
    def calculate_fitness(
        self, 
        tree, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> float:
        """
        Calculate fitness with uncertainty awareness.
        
        Extends base fitness with Bayesian components:
        - Confidence-weighted accuracy
        - Uncertainty penalties
        - Calibration metrics
        
        Args:
            tree: Tree genotype (Bayesian or deterministic)
            X: Features
            y: Labels
            
        Returns:
            Fitness score
        """
        # Check if Bayesian tree
        is_bayesian = hasattr(tree, 'bayesian_mode') and tree.bayesian_mode
        
        if is_bayesian:
            # Fit Bayesian parameters if needed
            if hasattr(tree, 'fit_bayesian_parameters'):
                tree.fit_bayesian_parameters(X, y)
        
        # Get predictions with uncertainty
        pred_result = self.bayesian_predictor.predict_with_uncertainty(tree, X)
        y_pred = pred_result['predictions']
        confidences = pred_result['confidences']
        
        # === BASE ACCURACY ===
        base_accuracy = accuracy_score(y, y_pred)
        
        # === CONFIDENCE-WEIGHTED ACCURACY ===
        if is_bayesian and confidences is not None:
            # Weight correct predictions by confidence
            correct_mask = (y_pred == y)
            confidence_weighted_acc = np.sum(
                confidences[correct_mask]
            ) / len(y)
            
            # Blend with base accuracy
            accuracy = (
                0.7 * base_accuracy + 
                0.3 * confidence_weighted_acc
            )
        else:
            accuracy = base_accuracy
        
        # === INTERPRETABILITY ===
        interpretability = self.interp_calc.calculate_composite_score(
            tree, self.interpretability_weights
        )
        
        # === UNCERTAINTY METRICS ===
        uncertainty_score = 0.0
        calibration_score = 0.0
        
        if is_bayesian:
            # 1. Overall uncertainty penalty
            uncertainties = pred_result.get('uncertainties', np.zeros(len(y)))
            mean_uncertainty = np.mean(uncertainties)
            uncertainty_score = 1.0 - mean_uncertainty  # Higher is better
            
            # 2. Epistemic uncertainty penalty (prefer data-informed models)
            epistemic = pred_result.get('epistemic', np.zeros(len(y)))
            mean_epistemic = np.mean(epistemic)
            epistemic_score = np.exp(-mean_epistemic)  # Penalize high epistemic
            
            # 3. Calibration (are probabilities well-calibrated?)
            probs = pred_result.get('class_probs')
            if probs is not None:
                try:
                    # Expected Calibration Error
                    ece = self._expected_calibration_error(y, probs)
                    calibration_score = 1.0 - ece
                except:
                    calibration_score = 0.5
            
            # 4. Confidence-based rewards/penalties
            low_confidence_mask = confidences < self.confidence_threshold
            low_conf_penalty = np.mean(low_confidence_mask)
            
            if self.risk_averse:
                # Heavily penalize low confidence predictions
                uncertainty_score *= (1.0 - 2 * low_conf_penalty)
        
        # === COMPOSITE FITNESS ===
        fitness_components = {
            'accuracy': accuracy,
            'interpretability': interpretability,
            'uncertainty': uncertainty_score,
            'calibration': calibration_score,
        }
        
        if is_bayesian:
            # Bayesian-aware fitness
            fitness = (
                self.accuracy_weight * accuracy +
                self.interpretability_weight * interpretability +
                self.uncertainty_weight * uncertainty_score +
                self.calibration_weight * calibration_score
            )
            
            # Additional epistemic penalty
            if epistemic_score < 1.0:
                fitness -= self.epistemic_penalty * (1.0 - epistemic_score)
        else:
            # Standard deterministic fitness
            fitness = (
                self.accuracy_weight * accuracy +
                self.interpretability_weight * interpretability
            )
        
        # Store detailed metrics
        tree.accuracy_ = accuracy
        tree.interpretability_ = interpretability
        
        if is_bayesian:
            tree.uncertainty_score_ = uncertainty_score
            tree.calibration_score_ = calibration_score
            tree.mean_confidence_ = np.mean(confidences)
        
        return max(0.0, fitness)  # Ensure non-negative
    
    def _expected_calibration_error(
        self, 
        y_true: np.ndarray, 
        y_probs: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Measures how well-calibrated probability predictions are.
        
        Args:
            y_true: True labels
            y_probs: Predicted probabilities
            n_bins: Number of calibration bins
            
        Returns:
            ECE score (lower is better, 0 = perfect calibration)
        """
        # Get confidences and predictions
        confidences = np.max(y_probs, axis=1)
        predictions = np.argmax(y_probs, axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        for i in range(n_bins):
            # Find samples in this bin
            bin_lower = bins[i]
            bin_upper = bins[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if np.sum(in_bin) > 0:
                # Average confidence in bin
                avg_confidence = np.mean(confidences[in_bin])
                # Average accuracy in bin
                avg_accuracy = np.mean(accuracies[in_bin])
                # Bin weight
                bin_weight = np.sum(in_bin) / len(y_true)
                
                # Add weighted absolute difference
                ece += bin_weight * abs(avg_confidence - avg_accuracy)
        
        return ece
    
    def evaluate_calibration(
        self,
        tree,
        X: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Detailed calibration analysis.
        
        Args:
            tree: Tree genotype
            X: Features
            y: Labels
            n_bins: Number of calibration bins
            
        Returns:
            Dictionary with calibration metrics
        """
        pred_result = self.bayesian_predictor.predict_with_uncertainty(tree, X)
        
        if pred_result['class_probs'] is None:
            return {'ece': 0.0, 'brier': 0.0, 'log_loss': 0.0}
        
        y_probs = pred_result['class_probs']
        y_pred = pred_result['predictions']
        
        # Expected Calibration Error
        ece = self._expected_calibration_error(y, y_probs, n_bins)
        
        # Brier Score
        y_one_hot = np.zeros_like(y_probs)
        y_one_hot[np.arange(len(y)), y] = 1
        brier = np.mean(np.sum((y_probs - y_one_hot)**2, axis=1))
        
        # Log Loss
        try:
            logloss = log_loss(y, y_probs)
        except:
            logloss = float('inf')
        
        return {
            'ece': ece,
            'brier': brier,
            'log_loss': logloss,
            'mean_confidence': np.mean(pred_result['confidences'])
        }
    
    def identify_uncertain_regions(
        self,
        tree,
        X: np.ndarray,
        uncertainty_threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Identify samples with high uncertainty.
        
        Useful for active learning or flagging for human review.
        
        Args:
            tree: Tree genotype
            X: Features
            uncertainty_threshold: Threshold for flagging
            
        Returns:
            Dictionary with uncertainty analysis
        """
        pred_result = self.bayesian_predictor.predict_with_uncertainty(tree, X)
        
        uncertainties = pred_result.get('uncertainties', np.zeros(len(X)))
        confidences = pred_result['confidences']
        
        # Flag high uncertainty samples
        high_uncertainty_mask = uncertainties > uncertainty_threshold
        low_confidence_mask = confidences < self.confidence_threshold
        
        return {
            'high_uncertainty_indices': np.where(high_uncertainty_mask)[0],
            'low_confidence_indices': np.where(low_confidence_mask)[0],
            'review_recommended': np.where(
                high_uncertainty_mask | low_confidence_mask
            )[0],
            'mean_uncertainty': np.mean(uncertainties),
            'mean_confidence': np.mean(confidences),
            'fraction_uncertain': np.mean(high_uncertainty_mask)
        }


class UncertaintyAwareInterpretabilityCalculator(InterpretabilityCalculator):
    """
    Extended interpretability calculator considering uncertainty.
    
    Adds uncertainty-based interpretability metrics.
    """
    
    @staticmethod
    def calculate_composite_score(tree, weights: Dict[str, float]) -> float:
        """
        Extended interpretability score with uncertainty.
        
        Args:
            tree: Tree genotype
            weights: Component weights
            
        Returns:
            Interpretability score [0, 1]
        """
        # Base interpretability
        base_score = InterpretabilityCalculator.calculate_composite_score(
            tree, weights
        )
        
        # Add uncertainty-based components for Bayesian trees
        if hasattr(tree, 'bayesian_mode') and tree.bayesian_mode:
            # Bonus for low epistemic uncertainty (more confident model)
            if hasattr(tree, 'get_uncertainty_map'):
                uncertainty_map = tree.get_uncertainty_map()
                
                # Average epistemic uncertainty across leaves
                leaf_uncertainties = [
                    info['epistemic'] 
                    for info in uncertainty_map.values()
                    if info['type'] == 'leaf'
                ]
                
                if leaf_uncertainties:
                    mean_epistemic = np.mean(leaf_uncertainties)
                    epistemic_bonus = np.exp(-mean_epistemic)  # [0, 1]
                    
                    # Add small bonus (5% weight)
                    base_score = 0.95 * base_score + 0.05 * epistemic_bonus
        
        return base_score


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compare_bayesian_vs_deterministic(
    bayesian_tree,
    deterministic_tree,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Compare Bayesian and deterministic versions of a tree.
    
    Args:
        bayesian_tree: BayesianTreeGenotype
        deterministic_tree: Deterministic TreeGenotype
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Comparison metrics
    """
    bay_calc = BayesianFitnessCalculator()
    det_calc = BaseFitnessCalculator()
    
    # Bayesian predictions
    bay_pred = BayesianTreePredictor.predict_with_uncertainty(
        bayesian_tree, X_test
    )
    bay_acc = accuracy_score(y_test, bay_pred['predictions'])
    
    # Deterministic predictions
    det_pred = TreePredictor.predict(deterministic_tree, X_test)
    det_acc = accuracy_score(y_test, det_pred)
    
    # Calibration
    bay_calib = bay_calc.evaluate_calibration(bayesian_tree, X_test, y_test)
    
    return {
        'bayesian_accuracy': bay_acc,
        'deterministic_accuracy': det_acc,
        'accuracy_diff': bay_acc - det_acc,
        'mean_confidence': bay_pred['confidences'].mean(),
        'mean_uncertainty': bay_pred['uncertainties'].mean(),
        'ece': bay_calib['ece'],
        'brier_score': bay_calib['brier']
    }