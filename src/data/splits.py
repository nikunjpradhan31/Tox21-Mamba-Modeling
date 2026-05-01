import torch
from torch.utils.data import Subset, random_split
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict


def generate_scaffold(smiles, include_chirality=False):
    """
    Compute the Bemis-Murcko scaffold for a given SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
        return scaffold
    except Exception:
        return ""


def scaffold_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    """
    Splits the dataset based on Bemis-Murcko scaffolds to ensure structural separation.

    Args:
        dataset: A PyTorch Geometric dataset (must have 'smiles' attribute for each data point).
        frac_train: Fraction of data for training.
        frac_valid: Fraction of data for validation.
        frac_test: Fraction of data for testing.

    Returns:
        tuple: (train_subset, valid_subset, test_subset)
    """
    assert len(dataset) > 0, "Dataset is empty"
    assert abs(frac_train + frac_valid + frac_test - 1.0) < 1e-5, (
        "Fractions must sum to 1.0"
    )

    scaffolds = defaultdict(list)

    for i in range(len(dataset)):
        data = dataset[i]

        smiles = None
        if hasattr(data, "smiles"):
            smiles = data.smiles
            if isinstance(smiles, list):
                smiles = smiles[0]

        if smiles:
            scaffold = generate_scaffold(smiles)
            scaffolds[scaffold].append(i)
        else:
            scaffolds[""].append(i)

    # Sort scaffolds by size (descending order), breaking ties by scaffold string
    # This places the largest scaffolds in the training set first
    scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[0]), reverse=True
        )
    ]

    train_idx = []
    valid_idx = []
    test_idx = []

    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train + frac_valid) * len(dataset)

    for scaffold_set in scaffold_sets:
        if len(train_idx) < train_cutoff:
            train_idx.extend(scaffold_set)
        elif len(valid_idx) < (valid_cutoff - train_cutoff):
            valid_idx.extend(scaffold_set)
        else:
            test_idx.extend(scaffold_set)

    return (
        Subset(dataset, train_idx),
        Subset(dataset, valid_idx),
        Subset(dataset, test_idx),
    )



def random_split_dataset(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42):
    assert abs(frac_train + frac_valid + frac_test - 1.0) < 1e-5

    n = len(dataset)
    n_train = int(frac_train * n)
    n_valid = int(frac_valid * n)
    n_test = n - n_train - n_valid

    generator = torch.Generator().manual_seed(seed)

    return random_split(dataset, [n_train, n_valid, n_test], generator=generator)