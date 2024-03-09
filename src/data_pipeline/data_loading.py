import os, glob

from typing import Any, Callable, Optional

from torch_geometric.data.data import BaseData
from config_pckg.config_file import Config

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as pyg_t
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from data_pipeline.augmentation import SampleBoundaryPoints, SampleDomainPoints


class CfdDataset(InMemoryDataset):
    def __init__(self, split_idxs, root: str | None = None, transform: Callable[..., Any] | None = None, pre_transform: Callable[..., Any] | None = None, pre_filter: Callable[..., Any] | None = None, log: bool = True):
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.split_idxs = split_idxs

    def len(self):
        return self.split_idxs.shape[0]

    def get(self, idx: int) -> BaseData:
        data_filenames = np.array(sorted(glob.glob(pathname="*.pt", root_dir=self.root)))[self.split_idxs]
        print(f"idx - {idx} - {data_filenames[idx]}")
        return torch.load(os.path.join(self.root, data_filenames[idx]))


def get_data_loaders(conf: Config, n_workers: Optional[int] = 4):
    # TODO: implement transformation to add labels during loader creation
    # to avoid saving n different graphs for n different snapshots of the same simulation
    full_conf = conf.get_logging_info()
    transform_list = []
    if conf.flag_BC_PINN:
        transform_list.append(SampleBoundaryPoints(full_conf))
    if conf.PINN_mode != "supervised_only":
        transform_list.append(SampleDomainPoints(full_conf))

    transforms = pyg_t.Compose(transform_list)

    dataset_train = CfdDataset(conf.split_idxs["train"], root=conf.standard_dataset_dir, transform=transforms)
    dataset_val = CfdDataset(conf.split_idxs["val"], root=conf.standard_dataset_dir)
    dataset_test = CfdDataset(conf.split_idxs["test"], root=conf.standard_dataset_dir)

    train_dataloader = DataLoader(dataset_train, 
                            batch_size=conf.hyper_params["training"]["batch_size"],
                            shuffle=True, 
                            num_workers=n_workers, 
                            # persistent_workers=True if n_workers>0 else False,
                            )
    val_dataloader  = DataLoader(dataset_val,
                            batch_size=1)
    test_dataloader = DataLoader(dataset_test, 
                            batch_size=1)

    return train_dataloader, val_dataloader, test_dataloader
