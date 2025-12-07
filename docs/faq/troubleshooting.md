# Troubleshooting Guide

Common issues and solutions.

## Installation Issues

### "ModuleNotFoundError: No module named 'ga_trees'"

**Solution:**
```bash
pip install -e .
```

### "ImportError: graphviz not found"

**Solution:**
```bash
# Install system package
sudo apt-get install graphviz  # Ubuntu
brew install graphviz          # macOS

# Install Python package
pip install graphviz
```

## Training Issues

### Fitness not improving

**Symptoms:** Best fitness plateaus after 5-10 generations

**Solutions:**
1. Increase population size:
   ```yaml
   ga:
     population_size: 150  # Instead of 50
   ```

2. Increase generations:
   ```yaml
   ga:
     n_generations: 60  # Instead of 30
   ```

3. Adjust mutation rate:
   ```yaml
   ga:
     mutation_prob: 0.25  # Increase exploration
   ```

### Trees too large

**Symptoms:** Trees have 50+ nodes, difficult to interpret

**Solutions:**
1. Increase interpretability weight:
   ```yaml
   fitness:
     weights:
       accuracy: 0.50
       interpretability: 0.50
   ```

2. Increase node complexity penalty:
   ```yaml
   fitness:
     interpretability_weights:
       node_complexity: 0.70
   ```

3. More aggressive pruning:
   ```yaml
   ga:
     mutation_types:
       prune_subtree: 0.40
   ```

### Poor accuracy

**Solutions:**
1. Increase accuracy weight
2. Allow deeper trees
3. Increase population/generations
4. Run hyperparameter optimization
