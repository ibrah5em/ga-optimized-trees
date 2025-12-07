"""Beautiful tree visualization using graphviz."""

try:
    import graphviz

    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("Warning: graphviz not installed. Install with: pip install graphviz")


class TreeVisualizer:
    """Visualize decision trees with graphviz."""

    @staticmethod
    def tree_to_graphviz(tree, feature_names=None, class_names=None):
        """Convert tree to graphviz format."""
        if not GRAPHVIZ_AVAILABLE:
            print("Graphviz not available!")
            return None

        dot = graphviz.Digraph(comment="Decision Tree")
        dot.attr(rankdir="TB")
        dot.attr("node", shape="box", style="rounded,filled", fontname="helvetica")

        def add_node(node, parent_id=None, edge_label=""):
            node_id = str(node.node_id)

            if node.is_leaf():
                # Leaf node
                label = f"Class: {node.prediction}\\nSamples: {node.samples_count}"
                dot.node(node_id, label, fillcolor="#90EE90")
            else:
                # Internal node
                feature_name = (
                    feature_names[node.feature_idx] if feature_names else f"X[{node.feature_idx}]"
                )
                label = f"{feature_name} <= {node.threshold:.3f}\\nSamples: {node.samples_count}"
                dot.node(node_id, label, fillcolor="#87CEEB")

            if parent_id is not None:
                dot.edge(parent_id, node_id, label=edge_label)

            if node.left_child:
                add_node(node.left_child, node_id, "True")
            if node.right_child:
                add_node(node.right_child, node_id, "False")

        add_node(tree.root)
        return dot

    @staticmethod
    def visualize_tree(
        tree, feature_names=None, class_names=None, save_path="results/figures/tree_viz"
    ):
        """Visualize and save tree."""
        dot = TreeVisualizer.tree_to_graphviz(tree, feature_names, class_names)
        if dot:
            dot.render(save_path, format="png", cleanup=True)
            print(f"✓ Tree visualization saved to: {save_path}.png")
