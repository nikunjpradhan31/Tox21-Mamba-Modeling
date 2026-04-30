import torch
from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolToSmiles, MolFromSmiles
from typing import Tuple, Optional


def get_order_single_mol(mol_data):
    """Logic for a single molecule only"""
    smiles = mol_data.smiles
    if isinstance(smiles, list): smiles = smiles[0]
    
    original_mol = MolFromSmiles(smiles)
    if original_mol is None:
        return torch.arange(mol_data.num_nodes, dtype=torch.long)
        
    # Get canonical atom ranking
    # canonical_rank[i] = the rank of the i-th atom in the original SMILES
    canonical_rank = list(Chem.CanonicalRankAtoms(original_mol))
    
    # To 'sort' atoms canonically, we return the indices that would 
    # put the atoms in their ranked order
    return torch.argsort(torch.tensor(canonical_rank, dtype=torch.long))

def get_order(data, descending: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    # data is a PyG Batch object
    device = data.x.device
    all_perms = []
    
    # We must iterate through the individual molecules in the batch
    mol_list = data.to_data_list()
    
    current_node_offset = 0
    for mol in mol_list:
        # Get the permutation for this specific molecule (0 to N_atoms-1)
        perm = get_order_single_mol(mol) 
        
        # Shift the indices to point to the correct position in the global batch tensor
        all_perms.append(perm + current_node_offset)
        
        current_node_offset += mol.num_nodes
        
    full_ordering = torch.cat(all_perms).to(device)
    scores = torch.ones(full_ordering.size(0), device=device)
    
    return full_ordering, scores