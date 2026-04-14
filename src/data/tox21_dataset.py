import os
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from .featurizer import MolFeaturizer


class Tox21Dataset(InMemoryDataset):
    """
    Custom Tox21 Dataset that reads directly from tox21.csv
    and applies MolFeaturizer.
    """

    def __init__(
        self, root="tox21.csv", transform=None, pre_transform=None, pre_filter=None
    ):
        self.csv_path = root 

        if pre_transform is None:
            pre_transform = MolFeaturizer()

        super().__init__(
            root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["tox21.csv"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        import shutil
        if os.path.exists(self.csv_path):
            os.makedirs(self.raw_dir, exist_ok=True)
            shutil.copy(self.csv_path, os.path.join(self.raw_dir, "tox21.csv"))
        else:
            raise FileNotFoundError(
                f"{self.csv_path} not found. Please provide tox21.csv in the root."
            )

    @property
    def num_tasks(self):
        return 12

    def process(self):
        df = pd.read_csv(self.raw_paths[0])

        target_cols = [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ]

        data_list = []
        for i, row in df.iterrows():
            smiles = row["smiles"]

            labels = row[target_cols].astype(float).tolist()
            y = torch.tensor([labels], dtype=torch.float32)

            data = Data(smiles=smiles, y=y)
            data.mol_id = row["mol_id"]

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if not hasattr(data, "x") or data.x is None:
                continue

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_tox21_dataset(root="tox21.csv"):

    dataset = Tox21Dataset(root=root)
    return dataset
