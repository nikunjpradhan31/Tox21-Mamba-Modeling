# Mamba-Modeling

Mamba-Modeling is a machine learning research project exploring the application of State Space Models (specifically Mamba) combined with Graph Isomorphism Networks (GIN) for molecular toxicity prediction on the Tox21 dataset.

## Context & Architecture

This repository implements a **Hybrid GIN-Mamba Model** (`GINMambaHybrid`) to leverage both graph structural information and sequence modeling capabilities. Because Mamba models operate on 1D sequences, a key challenge in applying them to graph-structured molecular data is determining the optimal sequence traversal or node ordering.

To investigate this, the project evaluates various **Node Ordering Strategies**:
- `random`: Random permutation of nodes
- `atomic_number`: Sorting nodes based on their atomic number
- `electronegativity`: Sorting nodes based on elemental electronegativity
- `degree`: Sorting nodes by their degree within the molecular graph
- `learned`: A parameterized, learnable ordering function

The codebase also supports a baseline standalone GIN ablation for comparative analysis.

## Key Technologies
- **PyTorch & PyTorch Geometric (PyG)**: For graph neural networks and overall deep learning framework.
- **Mamba-SSM**: For the State Space Model layers.
- **RDKit**: For cheminformatics and molecular feature extraction.
- **Scikit-learn / Numpy / Pandas**: For metrics, data manipulation, and training utility.

## Project Structure
- `src/models/`: Contains the `GINMambaHybrid` architecture.
- `src/ordering/`: Implements the various node sequence ordering strategies.
- `src/data/`: Handles `Tox21Dataset` loading, feature processing, and scaffold splitting.
- `src/training/`: Training and evaluation loops with metrics (ROC-AUC, PRC-AUC, F1-Score).
- `configs/`: YAML configuration files for hyperparameter management.
- `main.py` / `run_experiments.py`: Main entry points for training and experimentation.

## Usage

You can run the training pipeline via `main.py` using different orderings or model types:

```bash
# Run hybrid model with atomic number ordering
python main.py --model_type hybrid --ordering atomic_number

# Run standalone GIN baseline
python main.py --model_type gin

# Run with learnable node ordering
python main.py --model_type hybrid --ordering learned
```