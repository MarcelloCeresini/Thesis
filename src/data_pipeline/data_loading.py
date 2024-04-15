import os, glob

from typing import Any, Callable, Optional

from torch_geometric.data.data import BaseData
import wandb
from config_pckg.config_file import Config

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as pyg_t
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from data_pipeline.augmentation import SampleBoundaryPoints, SampleDomainPoints, \
        RemoveRadialAttributes, RemoveTurbulentLabels, NormalizeLabels


class CfdDataset(InMemoryDataset):
    def __init__(self, split_idxs, root: str | None = None, transform: Callable[..., Any] | None = None, pre_transform: Callable[..., Any] | None = None, pre_filter: Callable[..., Any] | None = None, log: bool = True):
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.split_idxs = split_idxs

    def len(self):
        return self.split_idxs.shape[0]

    def get(self, idx: int) -> BaseData:
        data_filenames = np.array(sorted(glob.glob(pathname="*.pt", root_dir=self.root)))[self.split_idxs]
        return torch.load(os.path.join(self.root, data_filenames[idx]))


def get_data_loaders(conf):

    transform_list_train = []
    transform_list_test = []
    general_transforms = []
    
    if conf["flag_BC_PINN"]:
        transform_list_train.append(SampleBoundaryPoints(conf))
        transform_list_test.append(SampleBoundaryPoints(conf, test=True))
    if conf["PINN_mode"] != "supervised_only":
        transform_list_train.append(SampleDomainPoints(conf))
        transform_list_test.append(SampleDomainPoints(conf, test=True))

    if not conf["bool_radial_attributes"]:
        general_transforms.append(RemoveRadialAttributes(conf))
    if not conf["output_turbulence"]:
        general_transforms.append(RemoveTurbulentLabels())

    general_transforms.append(NormalizeLabels(conf))

    transforms_train = pyg_t.Compose(transform_list_train+general_transforms)
    transforms_test = pyg_t.Compose(transform_list_test+general_transforms)

    dataset_train = CfdDataset(np.array(conf["split_idxs"]["train"]), root=conf["standard_dataset_dir"], transform=transforms_train)
    dataset_val = CfdDataset(np.array(conf["split_idxs"]["val"]), root=conf["standard_dataset_dir"], transform=transforms_test)
    dataset_test = CfdDataset(np.array(conf["split_idxs"]["test"]), root=conf["standard_dataset_dir"], transform=transforms_test)
    dataset_train_for_metrics = CfdDataset(np.array(conf["split_idxs"]["train"]), root=conf["standard_dataset_dir"], transform=transforms_test)

    n_workers_train=conf["hyper_params"]["training"]["n_workers_dataloaders"]
    train_dataloader = DataLoader(dataset_train, 
                            batch_size=conf["hyper_params"]["training"]["batch_size"],
                            shuffle=True, 
                            num_workers=n_workers_train, 
                            persistent_workers=n_workers_train>0,
                            pin_memory=True
                            )
    val_dataloader  = DataLoader(dataset_val, batch_size=1, pin_memory=False)
    test_dataloader = DataLoader(dataset_test, batch_size=1)
    train_dataloader_for_metrics = DataLoader(dataset_train_for_metrics, batch_size=1)

    return train_dataloader, val_dataloader, test_dataloader, train_dataloader_for_metrics
