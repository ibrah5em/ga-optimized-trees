"""Model explainability with LIME and SHAP."""

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class TreeExplainer:
    """Explain GA tree predictions."""

    @staticmethod
    def explain_with_shap(tree, X, feature_names=None):
        """Use SHAP to explain tree."""
        if not SHAP_AVAILABLE:
            print("SHAP not available! Install: pip install shap")
            return

        from ga_trees.fitness.calculator import TreePredictor

        predictor = TreePredictor()

        # Create prediction function
        def predict_fn(X_sample):
            return np.array([predictor.predict(tree, X_sample)])

        # SHAP explainer
        explainer = shap.Explainer(predict_fn, X)
        shap_values = explainer(X)

        # Visualize
        shap.summary_plot(shap_values, X, feature_names=feature_names)

    @staticmethod
    def explain_with_lime(tree, X, instance_idx, feature_names=None):
        """Use LIME to explain single prediction."""
        if not LIME_AVAILABLE:
            print("LIME not available! Install: pip install lime")
            return

        from ga_trees.fitness.calculator import TreePredictor

        predictor = TreePredictor()

        # Create explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X, feature_names=feature_names, mode="classification"
        )

        # Explain instance
        exp = explainer.explain_instance(X[instance_idx], lambda x: predictor.predict(tree, x))

        exp.show_in_notebook()
