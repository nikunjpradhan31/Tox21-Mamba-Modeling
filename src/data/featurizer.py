import math
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen


def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]


def get_node_features(atom, gasteiger=0.0, logp=0.0, mr=0.0):
    # --- Atomic number (bucketed instead of raw) ---
    atomic_num = one_hot_encoding(
        atom.GetAtomicNum(),
        [1, 6, 7, 8, 9, 15, 16, 17, 35, 53, 0]  # common + unknown
    )

    # --- Degree ---
    degree = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])

    hybridization = one_hot_encoding(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            "other"
        ]
    )

    formal_charge = one_hot_encoding(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    num_h = one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    aromatic = [int(atom.GetIsAromatic())]
    in_ring = [int(atom.IsInRing())]
    chirality = one_hot_encoding(
        str(atom.GetChiralTag()),
        [
            "CHI_UNSPECIFIED",
            "CHI_TETRAHEDRAL_CW",
            "CHI_TETRAHEDRAL_CCW",
            "other"
        ]
    )

    radical = [atom.GetNumRadicalElectrons()]
    
    # --- Electronic properties ---
    electronic = [gasteiger, logp, mr]
    
    features = (
        atomic_num +
        degree +
        hybridization +
        formal_charge +
        num_h +
        aromatic +
        in_ring +
        chirality +
        radical +
        electronic
    )
    return list(map(float, features))


def get_edge_features(bond):
    """
    Tox21-optimized edge features for molecular graphs.
    """

    # Bond type encoding (more stable than raw float alone)
    bond_type = bond.GetBondType()
    bond_type_onehot = [
        int(bond_type == Chem.rdchem.BondType.SINGLE),
        int(bond_type == Chem.rdchem.BondType.DOUBLE),
        int(bond_type == Chem.rdchem.BondType.TRIPLE),
        int(bond_type == Chem.rdchem.BondType.AROMATIC),
    ]

    is_in_ring = int(bond.IsInRing())
    is_conjugated = int(bond.GetIsConjugated())

    # Directionality (important for stereochemistry-aware tasks)
    is_stereo = int(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE)

    # Aromatic flag (redundant but useful explicitly at edge level)
    is_aromatic = int(bond.GetIsAromatic())

    # Conformational flexibility proxy
    is_rotatable = int(bond.GetBondType() == Chem.rdchem.BondType.SINGLE and not bond.IsInRing())

    return (
        bond_type_onehot +
        [
            float(is_in_ring),
            float(is_conjugated),
            float(is_stereo),
            float(is_aromatic),
            float(is_rotatable),
        ]
    )


class MolFeaturizer:
    """
    Callable class to convert an RDKit molecule or a PyG Data object with SMILES
    into a PyG Data object with custom featurization.
    """

    def __call__(self, data_or_mol):
        # Handle PyG Data object with SMILES
        if isinstance(data_or_mol, Data) and hasattr(data_or_mol, "smiles"):
            smiles = data_or_mol.smiles
            if isinstance(smiles, list):
                smiles = smiles[0]
            mol = Chem.MolFromSmiles(smiles)
            y = data_or_mol.y
        elif isinstance(data_or_mol, Chem.Mol):
            mol = data_or_mol
            y = None
        elif isinstance(data_or_mol, str):
            mol = Chem.MolFromSmiles(data_or_mol)
            y = None
        else:
            raise ValueError(
                "Input must be an RDKit Mol, a SMILES string, or a PyG Data object with a 'smiles' attribute."
            )

        if mol is None:
            return data_or_mol

        num_nodes = mol.GetNumAtoms()

        # Compute electronic and structural descriptors globally for the molecule
        try:
            Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        except:
            pass
            
        try:
            contribs = Crippen._GetAtomContribs(mol)
        except:
            contribs = [(0.0, 0.0)] * num_nodes

        node_features = []
        z = []
        for i, atom in enumerate(mol.GetAtoms()):
            try:
                gasteiger = float(atom.GetProp('_GasteigerCharge'))
                if math.isnan(gasteiger) or math.isinf(gasteiger):
                    gasteiger = 0.0
            except:
                gasteiger = 0.0
                
            logp, mr = contribs[i] if i < len(contribs) else (0.0, 0.0)
            
            node_features.append(get_node_features(atom, gasteiger, logp, mr))
            z.append(atom.GetAtomicNum())

        edge_indices = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_features_ij = get_edge_features(bond)

            # Bidirectional edges for undirected graph
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            edge_features.append(edge_features_ij)
            edge_features.append(edge_features_ij)

        x = torch.tensor(node_features, dtype=torch.float)
        z_tensor = torch.tensor(z, dtype=torch.long)

        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 9), dtype=torch.float)

        # Global Morgan Fingerprint
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fp_tensor = torch.tensor(list(fp), dtype=torch.float).unsqueeze(0)
        except:
            fp_tensor = torch.zeros((1, 1024), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, z=z_tensor, fp=fp_tensor)

        if y is not None:
            data.y = y

        # Preserve SMILES if available
        if isinstance(data_or_mol, Data) and hasattr(data_or_mol, "smiles"):
            data.smiles = data_or_mol.smiles
        elif isinstance(data_or_mol, str):
            data.smiles = data_or_mol

        return data
