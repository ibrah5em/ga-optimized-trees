# Credit Scoring Example

Financial application minimizing false positives.

## Scenario

**Binary classification**: Approve/reject loan application
**Priority**: Minimize false positives (bad loans approved)
**Requirement**: Transparent model for regulatory compliance

## Custom Fitness for Credit Scoring

```python
def credit_fitness(tree, X, y):
    """
    Fitness for credit scoring:
    - Minimize false positives (costly)
    - Maintain reasonable recall
    - Ensure interpretability for compliance
    """
    from ga_trees.fitness.calculator import TreePredictor
    from sklearn.metrics import confusion_matrix

    predictor = TreePredictor()
    predictor.fit_leaf_predictions(tree, X, y)
    y_pred = predictor.predict(tree, X)

    # Get confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # Financial metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True positive rate

    # Cost-based scoring
    cost_fp = 1000  # Cost of approving bad loan
    cost_fn = 100  # Cost of rejecting good applicant
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    max_cost = len(y) * cost_fp
    cost_score = 1.0 - (total_cost / max_cost)

    # Interpretability (regulatory requirement)
    node_penalty = 1.0 - min(tree.get_num_nodes() / 20.0, 1.0)

    # Combined fitness
    fitness = (
        0.50 * (1.0 - fpr)  # Minimize false positives (primary)
        + 0.25 * tpr  # Maintain true positives
        + 0.15 * cost_score  # Overall financial performance
        + 0.10 * node_penalty  # Regulatory compliance
    )

    tree.accuracy_ = (tp + tn) / len(y)
    tree.interpretability_ = node_penalty

    return fitness
```
