"""
Bayesian Decision Trees - Complete Usage Examples

This script demonstrates all probabilistic methods and features:
1. Sample thresholds and leaf distributions
2. Soft routing with probabilistic decisions
3. Calibration metrics and uncertainty quantification
4. Validation and mode switching
5. Medical domain example with high confidence requirements

Save as: scripts/bayesian_examples.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import Bayesian components
from ga_trees.genotype.bayesian_tree_genotype import (
    BayesianTreeGenotype,
    BayesianNode,
    create_bayesian_leaf_node,
    create_bayesian_internal_node
)
from ga_trees.fitness.bayesian_calculator import (
    BayesianFitnessCalculator,
    BayesianTreePredictor
)


# ============================================================================
# EXAMPLE 1: Basic Bayesian Node - Probabilistic Methods
# ============================================================================

def example_1_bayesian_node_basics():
    """Demonstrate core probabilistic methods on a single node."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Bayesian Node - Probabilistic Sampling")
    print("="*70)
    
    # Create a Bayesian internal node
    print("\n1.1 Threshold Sampling")
    print("-" * 50)
    
    node = BayesianNode(
        bayesian_mode=True,
        node_type='internal',
        feature_idx=0,
        threshold_mean=0.5,
        threshold_std=0.1,
        threshold_dist_type='normal'
    )
    
    # Sample thresholds
    samples = node.sample_threshold(n_samples=1000, method='posterior')
    print(f"Threshold distribution:")
    print(f"  Mean: {samples.mean():.4f} (expected: {node.threshold_mean})")
    print(f"  Std:  {samples.std():.4f} (expected: {node.threshold_std})")
    print(f"  95% CI: [{np.percentile(samples, 2.5):.4f}, {np.percentile(samples, 97.5):.4f}]")
    
    # Get confidence interval
    ci_lower, ci_upper = node.get_threshold_confidence_interval(confidence=0.95)
    print(f"  Analytical CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Create a Bayesian leaf node
    print("\n1.2 Leaf Distribution Sampling")
    print("-" * 50)
    
    leaf = BayesianNode(
        bayesian_mode=True,
        node_type='leaf',
        leaf_alpha=np.array([10, 5, 2])  # More evidence for class 0
    )
    leaf._update_leaf_statistics()
    
    # Sample predictions
    predictions, probs = leaf.sample_leaf_distribution(
        n_samples=1000, 
        return_probs=True
    )
    
    print(f"Leaf prediction distribution:")
    print(f"  Expected probs: {leaf.leaf_class_probs}")
    print(f"  Sampled probs:  {probs.mean(axis=0)}")
    print(f"  Class counts: {np.bincount(predictions)}")
    
    # Get prediction with confidence
    pred_info = leaf.get_prediction_with_confidence()
    print(f"\nPrediction info:")
    print(f"  Prediction: Class {pred_info['prediction']}")
    print(f"  Confidence: {pred_info['confidence']:.4f}")
    print(f"  Total uncertainty: {pred_info['uncertainty']:.4f}")
    print(f"  Aleatoric (data): {pred_info['aleatoric']:.4f}")
    print(f"  Epistemic (model): {pred_info['epistemic']:.4f}")
    
    # Demonstrate soft decision
    print("\n1.3 Soft Routing Probabilities")
    print("-" * 50)
    
    # Test different feature values
    test_values = [0.45, 0.50, 0.55, 0.70]
    for val in test_values:
        routing = node.get_soft_decision_prob(val, n_samples=10000)
        print(f"\nFeature value = {val:.2f}:")
        print(f"  P(left)  = {routing['prob_left']:.4f}")
        print(f"  P(right) = {routing['prob_right']:.4f}")
        print(f"  Entropy  = {routing['entropy']:.4f}")
        print(f"  Confidence = {routing['confidence']:.4f}")


# ============================================================================
# EXAMPLE 2: Build and Validate Bayesian Tree
# ============================================================================

def example_2_build_and_validate():
    """Build a simple Bayesian tree and validate it."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Build and Validate Bayesian Tree")
    print("="*70)
    
    # Create a simple 3-node Bayesian tree
    print("\n2.1 Building Tree")
    print("-" * 50)
    
    # Leaves
    left_leaf = create_bayesian_leaf_node(
        prediction=0,
        n_classes=2,
        depth=1,
        prior_alpha=1.0
    )
    right_leaf = create_bayesian_leaf_node(
        prediction=1,
        n_classes=2,
        depth=1,
        prior_alpha=1.0
    )
    
    # Root
    root = create_bayesian_internal_node(
        feature_idx=0,
        threshold=0.5,
        left_child=left_leaf,
        right_child=right_leaf,
        depth=0,
        threshold_std=0.1
    )
    
    # Create tree
    tree = BayesianTreeGenotype(
        root=root,
        n_features=4,
        n_classes=2,
        mode='bayesian',  # Using new mode parameter
        bayesian_config={
            'threshold_prior_std': 0.1,
            'leaf_prior_alpha': 1.0,
            'n_mc_samples': 100,
            'confidence_threshold': 0.8,
            'uncertainty_threshold': 0.5,
            'enable_soft_routing': True
        }
    )
    
    print(f"✓ Created tree: {tree}")
    print(f"  Mode: {tree.mode}")
    print(f"  Bayesian mode: {tree.bayesian_mode}")
    print(f"  Depth: {tree.get_depth()}")
    print(f"  Nodes: {tree.get_num_nodes()}")
    
    # Validate Bayesian structure
    print("\n2.2 Validation")
    print("-" * 50)
    
    valid, errors = tree.validate_bayesian()
    
    if valid:
        print("✓ Tree is valid!")
    else:
        print("✗ Validation errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Test copy
    print("\n2.3 Deep Copy Test")
    print("-" * 50)
    
    tree_copy = tree.copy()
    print(f"✓ Created copy")
    print(f"  Original root ID: {id(tree.root)}")
    print(f"  Copy root ID: {id(tree_copy.root)}")
    print(f"  Are different objects: {id(tree.root) != id(tree_copy.root)}")
    
    # Modify copy
    tree_copy.root.threshold_mean = 0.7
    print(f"\nModified copy threshold_mean to 0.7")
    print(f"  Original: {tree.root.threshold_mean}")
    print(f"  Copy: {tree_copy.root.threshold_mean}")
    print(f"  ✓ Independent copies confirmed")


# ============================================================================
# EXAMPLE 3: Mode Switching and Compatibility
# ============================================================================

def example_3_mode_switching():
    """Demonstrate mode switching between deterministic and Bayesian."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Mode Switching (Backward Compatibility)")
    print("="*70)
    
    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print("\n3.1 Create Bayesian Tree")
    print("-" * 50)
    
    # Create simple tree
    from ga_trees.ga.engine import TreeInitializer
    
    initializer = TreeInitializer(
        n_features=4,
        n_classes=3,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    det_tree = initializer.create_random_tree(X_train, y_train)
    
    # Convert to Bayesian
    bay_tree = BayesianTreeGenotype(
        root=det_tree.root,
        n_features=det_tree.n_features,
        n_classes=det_tree.n_classes,
        mode='bayesian',
        max_depth=det_tree.max_depth,
        min_samples_split=det_tree.min_samples_split,
        min_samples_leaf=det_tree.min_samples_leaf
    )
    
    # Fit Bayesian parameters
    bay_tree.fit_bayesian_parameters(X_train, y_train)
    
    print(f"✓ Converted to Bayesian")
    print(f"  Mode: {bay_tree.mode}")
    
    # Make predictions
    print("\n3.2 Bayesian Predictions")
    print("-" * 50)
    
    bay_pred = bay_tree.predict_with_uncertainty(X_test, n_samples=100)
    
    print(f"Prediction statistics:")
    print(f"  Accuracy: {accuracy_score(y_test, bay_pred['predictions']):.4f}")
    print(f"  Mean confidence: {bay_pred['confidences'].mean():.4f}")
    print(f"  Mean uncertainty: {bay_pred['uncertainties'].mean():.4f}")
    print(f"  Mean aleatoric: {bay_pred['aleatoric'].mean():.4f}")
    print(f"  Mean epistemic: {bay_pred['epistemic'].mean():.4f}")
    
    # Convert back to deterministic
    print("\n3.3 Convert Back to Deterministic")
    print("-" * 50)
    
    det_tree_converted = bay_tree.to_deterministic()
    
    print(f"✓ Converted back to deterministic")
    print(f"  Type: {type(det_tree_converted)}")
    print(f"  Nodes: {det_tree_converted.get_num_nodes()}")
    
    # Compare predictions
    from ga_trees.fitness.calculator import TreePredictor
    predictor = TreePredictor()
    det_pred = predictor.predict(det_tree_converted, X_test)
    
    print(f"\nPrediction comparison:")
    print(f"  Bayesian accuracy: {accuracy_score(y_test, bay_pred['predictions']):.4f}")
    print(f"  Deterministic accuracy: {accuracy_score(y_test, det_pred):.4f}")
    print(f"  Agreement: {np.mean(bay_pred['predictions'] == det_pred):.4f}")


# ============================================================================
# EXAMPLE 4: Soft Routing vs Hard Routing
# ============================================================================

def example_4_soft_vs_hard_routing():
    """Compare soft and hard routing strategies."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Soft vs Hard Routing")
    print("="*70)
    
    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create Bayesian tree
    from ga_trees.ga.engine import TreeInitializer
    
    initializer = TreeInitializer(
        n_features=4,
        n_classes=3,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    base_tree = initializer.create_random_tree(X_train, y_train)
    
    bay_tree = BayesianTreeGenotype(
        root=base_tree.root,
        n_features=4,
        n_classes=3,
        mode='bayesian',
        max_depth=4,
        bayesian_config={
            'threshold_prior_std': 0.15,  # Higher uncertainty
            'enable_soft_routing': True
        }
    )
    
    bay_tree.fit_bayesian_parameters(X_train, y_train)
    
    print("\n4.1 Hard Routing (Standard)")
    print("-" * 50)
    
    hard_pred = bay_tree.predict_with_uncertainty(
        X_test, 
        n_samples=100, 
        use_soft_routing=False
    )
    
    print(f"Hard routing results:")
    print(f"  Accuracy: {accuracy_score(y_test, hard_pred['predictions']):.4f}")
    print(f"  Mean confidence: {hard_pred['confidences'].mean():.4f}")
    print(f"  Mean uncertainty: {hard_pred['uncertainties'].mean():.4f}")
    
    print("\n4.2 Soft Routing (Probabilistic)")
    print("-" * 50)
    
    soft_pred = bay_tree.predict_with_uncertainty(
        X_test,
        n_samples=100,
        use_soft_routing=True
    )
    
    print(f"Soft routing results:")
    print(f"  Accuracy: {accuracy_score(y_test, soft_pred['predictions']):.4f}")
    print(f"  Mean confidence: {soft_pred['confidences'].mean():.4f}")
    print(f"  Mean uncertainty: {soft_pred['uncertainties'].mean():.4f}")
    
    print("\n4.3 Comparison")
    print("-" * 50)
    
    print(f"Prediction agreement: {np.mean(hard_pred['predictions'] == soft_pred['predictions']):.4f}")
    print(f"Confidence diff: {abs(hard_pred['confidences'].mean() - soft_pred['confidences'].mean()):.4f}")
    
    # Find samples where routing matters
    diff_mask = hard_pred['predictions'] != soft_pred['predictions']
    if np.any(diff_mask):
        print(f"\nRouting changed predictions for {np.sum(diff_mask)} samples")
        print(f"  Hard routing uncertainty: {hard_pred['uncertainties'][diff_mask].mean():.4f}")
        print(f"  Soft routing uncertainty: {soft_pred['uncertainties'][diff_mask].mean():.4f}")


# ============================================================================
# EXAMPLE 5: Calibration Metrics
# ============================================================================

def example_5_calibration_metrics():
    """Demonstrate calibration metrics computation."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Calibration Metrics")
    print("="*70)
    
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create and train tree
    from ga_trees.ga.engine import TreeInitializer
    
    initializer = TreeInitializer(
        n_features=X_train.shape[1],
        n_classes=2,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5
    )
    
    base_tree = initializer.create_random_tree(X_train, y_train)
    
    bay_tree = BayesianTreeGenotype(
        root=base_tree.root,
        n_features=X_train.shape[1],
        n_classes=2,
        mode='bayesian',
        max_depth=5
    )
    
    bay_tree.fit_bayesian_parameters(X_train, y_train)
    
    print("\n5.1 Compute Calibration Metrics")
    print("-" * 50)
    
    calib_metrics = bay_tree.compute_calibration_metrics(X_test, y_test, n_bins=10)
    
    print(f"Calibration metrics:")
    print(f"  ECE (Expected Calibration Error): {calib_metrics['ece']:.4f}")
    print(f"  MCE (Maximum Calibration Error):  {calib_metrics['mce']:.4f}")
    print(f"  Brier Score: {calib_metrics['brier']:.4f}")
    print(f"  Reliability: {calib_metrics['reliability']:.4f}")
    
    # Update uncertainty metrics
    print("\n5.2 Update Uncertainty Metrics")
    print("-" * 50)
    
    bay_tree.update_uncertainty_metrics(X_test)
    
    print(f"Uncertainty metrics:")
    for key, value in bay_tree.prediction_uncertainty.items():
        print(f"  {key}: {value:.4f}")
    
    # Identify uncertain samples
    print("\n5.3 Identify Uncertain Samples")
    print("-" * 50)
    
    from ga_trees.fitness.bayesian_calculator import BayesianFitnessCalculator
    
    calc = BayesianFitnessCalculator(uncertainty_weight=0.1)
    uncertain_info = calc.identify_uncertain_regions(
        bay_tree,
        X_test,
        uncertainty_threshold=0.5
    )
    
    print(f"Uncertain samples analysis:")
    print(f"  High uncertainty samples: {len(uncertain_info['high_uncertainty_indices'])}")
    print(f"  Low confidence samples: {len(uncertain_info['low_confidence_indices'])}")
    print(f"  Recommended for review: {len(uncertain_info['review_recommended'])}")
    print(f"  Fraction uncertain: {uncertain_info['fraction_uncertain']:.2%}")


# ============================================================================
# EXAMPLE 6: Medical Domain - High Confidence Requirements
# ============================================================================

def example_6_medical_domain():
    """Medical diagnosis example with strict confidence requirements."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Medical Domain - Breast Cancer Diagnosis")
    print("="*70)
    
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    feature_names = load_breast_cancer().feature_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nDataset: Breast Cancer")
    print(f"  Classes: 0=Malignant, 1=Benign")
    print(f"  Train: {len(y_train)} samples")
    print(f"  Test: {len(y_test)} samples")
    
    # Create conservative Bayesian tree
    print("\n6.1 Create Medical-Grade Tree")
    print("-" * 50)
    
    from ga_trees.ga.engine import TreeInitializer
    
    initializer = TreeInitializer(
        n_features=X_train.shape[1],
        n_classes=2,
        max_depth=4,  # Shallow for interpretability
        min_samples_split=20,  # Conservative
        min_samples_leaf=10
    )
    
    base_tree = initializer.create_random_tree(X_train, y_train)
    
    medical_tree = BayesianTreeGenotype(
        root=base_tree.root,
        n_features=X_train.shape[1],
        n_classes=2,
        mode='bayesian',
        max_depth=4,
        bayesian_config={
            'threshold_prior_std': 0.05,  # Low uncertainty
            'leaf_prior_alpha': 2.0,  # Stronger prior
            'confidence_threshold': 0.90,  # Very high required
            'uncertainty_threshold': 0.3,  # Strict
            'risk_averse': True
        }
    )
    
    medical_tree.fit_bayesian_parameters(X_train, y_train)
    
    print(f"✓ Medical tree created")
    print(f"  Depth: {medical_tree.get_depth()}")
    print(f"  Nodes: {medical_tree.get_num_nodes()}")
    
    # Make predictions
    print("\n6.2 Diagnose Test Cases")
    print("-" * 50)
    
    pred_result = medical_tree.predict_with_uncertainty(X_test, n_samples=1000)
    
    accuracy = accuracy_score(y_test, pred_result['predictions'])
    print(f"\nOverall Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Mean confidence: {pred_result['confidences'].mean():.4f}")
    print(f"  Mean uncertainty: {pred_result['uncertainties'].mean():.4f}")
    
    # Identify cases requiring expert review
    print("\n6.3 Flag Cases for Expert Review")
    print("-" * 50)
    
    high_risk = pred_result['predictions'] == 0  # Malignant
    low_conf = pred_result['confidences'] < 0.90
    high_unc = pred_result['uncertainties'] > 0.3
    
    review_needed = high_risk & (low_conf | high_unc)
    
    print(f"\nReview recommendations:")
    print(f"  Total malignant predictions: {np.sum(high_risk)}")
    print(f"  Low confidence cases: {np.sum(low_conf)}")
    print(f"  High uncertainty cases: {np.sum(high_unc)}")
    print(f"  REQUIRES EXPERT REVIEW: {np.sum(review_needed)}")
    
    # Show example uncertain case
    if np.any(review_needed):
        idx = np.where(review_needed)[0][0]
        print(f"\nExample case requiring review (index {idx}):")
        print(f"  Prediction: {'MALIGNANT' if pred_result['predictions'][idx] == 0 else 'BENIGN'}")
        print(f"  True label: {'MALIGNANT' if y_test[idx] == 0 else 'BENIGN'}")
        print(f"  Confidence: {pred_result['confidences'][idx]:.4f}")
        print(f"  Total uncertainty: {pred_result['uncertainties'][idx]:.4f}")
        print(f"  Aleatoric (data noise): {pred_result['aleatoric'][idx]:.4f}")
        print(f"  Epistemic (model unsure): {pred_result['epistemic'][idx]:.4f}")
        print(f"  Class probabilities: {pred_result['class_probs'][idx]}")
    
    # Compute calibration
    print("\n6.4 Calibration Assessment")
    print("-" * 50)
    
    calib = medical_tree.compute_calibration_metrics(X_test, y_test)
    print(f"Calibration quality:")
    print(f"  ECE: {calib['ece']:.4f} {'✓ GOOD' if calib['ece'] < 0.1 else '⚠ NEEDS CALIBRATION'}")
    print(f"  Reliability: {calib['reliability']:.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "🌳 "*35)
    print("BAYESIAN DECISION TREES - COMPLETE EXAMPLES")
    print("🌳 "*35)
    
    examples = [
        ("Probabilistic Sampling", example_1_bayesian_node_basics),
        ("Build & Validate", example_2_build_and_validate),
        ("Mode Switching", example_3_mode_switching),
        ("Soft vs Hard Routing", example_4_soft_vs_hard_routing),
        ("Calibration Metrics", example_5_calibration_metrics),
        ("Medical Domain", example_6_medical_domain)
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n\n{'='*70}")
        print(f"Running Example {i}/{len(examples)}: {name}")
        print(f"{'='*70}")
        
        try:
            func()
            print(f"\n✓ Example {i} completed successfully")
        except Exception as e:
            print(f"\n✗ Example {i} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n\n" + "="*70)
    print("ALL EXAMPLES COMPLETED!")
    print("="*70)
    print("\n📚 Key Takeaways:")
    print("  • Bayesian nodes provide probabilistic thresholds & predictions")
    print("  • Soft routing averages over multiple paths")
    print("  • Calibration metrics ensure probability reliability")
    print("  • Uncertainty quantification enables risk-aware decisions")
    print("  • Medical-grade models flag uncertain cases for expert review")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()