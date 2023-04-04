# Adapted from Source: https://github.com/activatedgeek/simplex-gp/

import os
import random
import torch
import itertools
import pickle
import numpy as np


from typing import List, Union
from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat

UCI_PATH = Path(os.path.expanduser("~/sgkigp/data/uci/"))
#UCI_PATH = Path(os.path.expanduser("~/kron/data/uci/"))

small_datasets = [
    "challenger",
    "fertility",
    "concreteslump",
    "autos",
    "servo",
    "breastcancer",
    "machine",
    "yacht",
    "autompg",
    "housing",
    "forest",
    "stock",
    "pendulum",
    "energy",
    "concrete",
    "solar",
    "airfoil",
    "wine",
    "gas",
    "skillcraft",
    "sml",
    "parkinsons",
    "pumadyn32nm",
]

medium_datasets = [
    "pol",
    "elevators",
    "bike",
    "kin40k",
    "protein",
    "keggdirected",
    "slice",
    "keggundirected",
]

large_datasets = ["3droad", "song", "buzz"]
huge_datasets = ["houseelectric"]

all_datasets = small_datasets + medium_datasets + large_datasets + huge_datasets


def prepare_dataset(dataset, uci_data_dir, device=None, train_val_split=0.8):

    # if dataset == 'airline':
    #     airline_path = os.path.expanduser("~/sgkigp/data/airline/ss_10000_idx_0.pkl")
    #     X, Y, X_test, Y_test, X_val, Y_val = pickle.load(open(airline_path, 'rb'))
    #
    #     for m in ["train", "val", "test"]:
    #
    #         x = (splits.get(m).x - x_mean) / (x_std + 1e-6)
    #         y = (splits.get(m).y - y_mean) / (y_std + 1e-6)
    #         yield m, splits.get(m).x.contiguous(), splits.get(m).y.contiguous()

    if uci_data_dir is None and os.environ.get('DATADIR') is not None:
        uci_data_dir = Path(os.path.join(os.environ.get('DATADIR'), 'uci'))

    assert dataset is not None, f'Select a dataset from "{uci_data_dir}"'

    splits = {
        m: UCIDataset.create(dataset, uci_data_dir=uci_data_dir, mode=m,
                             device=device, train_val_split=train_val_split)
        for m in ["train", "val", "test"]
    }

    x_mean = splits.get('train').x.mean(0, keepdim=True)
    x_std = splits.get('train').x.std(0, keepdim=True) + 1e-6

    y_mean = splits.get('train').y.mean(0, keepdim=True)
    y_std = splits.get('train').y.std(0, keepdim=True) + 1e-6

    # for m in ["train", "val", "test"]:
    #     x = (splits.get(m).x - x_mean) / (x_std + 1e-6)
    #     y = (splits.get(m).y - y_mean) / (y_std + 1e-6)
    #     yield m, x.contiguous(), y.contiguous()

    for m in ["train", "val", "test"]:
        x = (splits.get(m).x - x_mean) / (x_std + 1e-6)
        y = (splits.get(m).y - y_mean) / (y_std + 1e-6)
        yield m, splits.get(m).x.contiguous(), splits.get(m).y.contiguous()


class UCIDataset(Dataset):

    def __init__(
            self,
            dataset_path: Union[Path, str],
            mode: str = "train",
            dtype=torch.float32,
            device="cpu",
            train_val_split=0.8,
    ):
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_path.stem
        data = loadmat(str(dataset_path))["data"]
        data = torch.as_tensor(data, dtype=dtype, device=device)

        # make train/val/test split
        N = data.size(0)
        n_train_val = int(train_val_split * N)
        n_train = int(train_val_split * n_train_val)

        train_x, train_y = data[:n_train, :-1], data[:n_train, -1]
        val_x, val_y = data[n_train:n_train_val, :-1], data[n_train:n_train_val, -1]
        test_x, test_y = data[n_train_val:, :-1], data[n_train_val:, -1]

        if mode == "train":
            self.x, self.y = train_x, train_y
        elif mode == "val":
            self.x, self.y, = val_x, val_y
        elif mode == "test":
            self.x, self.y = test_x, test_y
        else:
            raise ValueError("mode must be one of 'train', 'val', or 'test'")

    def __getitem__(self, index):
        return (self.x.__getitem__(index), self.y.__getitem__(index))

    def __len__(self):
        return len(self.y)

    @staticmethod
    def all_dataset_paths(uci_data_dir: Path = UCI_PATH):
        return list(uci_data_dir.glob("*.mat"))

    @staticmethod
    def all_dataset_names(uci_data_dir: Path = UCI_PATH):
        return [p.stem for p in UCIDataset.all_dataset_paths(uci_data_dir)]

    @staticmethod
    def create(*names: Union[str, list], uci_data_dir: Path = None, mode="train", dtype=torch.float32, device="cpu",
               train_val_split=0.8) -> List:
        """Create one or more `UCIDataset`s from their names
        `names` can be the name of a group of datasets as listed below
        Example:
            ```
            UCIDataset.create("challenger")
            UCIDataset.create("challenger", "fertility")
            UCIDataset.create("small")
            ```
        """
        uci_data_dir = UCI_PATH

        def get(dataset_names: List[str]):
            return [(uci_data_dir / d / d).with_suffix(".mat") for d in dataset_names]

        groups = {
            **{
                "all": UCIDataset.all_dataset_paths(uci_data_dir),
                "small": get(small_datasets),
                "medium": get(medium_datasets),
                "large": get(large_datasets),
                "huge": get(huge_datasets),
            },
            # Allow individual dataset names
            **{p.stem: [p] for p in UCIDataset.all_dataset_paths(uci_data_dir)},
        }
        try:
            datasets = itertools.chain.from_iterable([groups[g_or_n] for g_or_n in names])
        except KeyError:
            for name in names:
                assert name in all_datasets
            datasets = get(list(names))

        datasets = [UCIDataset(dataset_path, mode=mode, device=device, dtype=dtype, train_val_split=train_val_split) for
                    dataset_path in datasets]

        if len(datasets) == 1:
            return datasets[0]
        else:
            return datasets
