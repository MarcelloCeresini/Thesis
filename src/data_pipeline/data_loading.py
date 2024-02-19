import os, glob

from typing import Any, Callable, Optional
from config_pckg.config_file import Config

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch


def get_data_loaders(conf: Config, load_from_disk = False, save_to_disk = False) -> None:
    # TODO: implement transformation to add labels during loader creation
    # to avoid saving n different graphs for n different snapshots of the same simulation

    data_found = False
    if load_from_disk:
        if os.path.isfile(conf.standard_datalist_path):
            data_list_train, data_list_val, data_list_test = torch.load(conf.standard_datalist_path)
            data_found = True
        else:
            print("No datalist at standard path, creating them from scratch")
    
    if (not load_from_disk) or (not data_found):
        data_filenames = sorted(glob.glob(pathname="*.pt", root_dir=conf.EXTERNAL_FOLDER_GRAPHS))

        data_list_train = [torch.load(os.path.join(conf.EXTERNAL_FOLDER_GRAPHS, data_filenames[idx])) for idx in conf.split_idxs["train"]]
        data_list_val   = [torch.load(os.path.join(conf.EXTERNAL_FOLDER_GRAPHS, data_filenames[idx])) for idx in conf.split_idxs["val"]]
        data_list_test  = [torch.load(os.path.join(conf.EXTERNAL_FOLDER_GRAPHS, data_filenames[idx])) for idx in conf.split_idxs["test"]]

    if save_to_disk:
        torch.save(
            (data_list_train, data_list_val, data_list_test),
            conf.standard_datalist_path
            )
    
    train_dataloader = DataLoader(data_list_train, 
                            batch_size=conf.hyper_params["training"]["batch_size"],
                            shuffle=True)
    val_dataloader  = DataLoader(data_list_val,  
                            batch_size=conf.hyper_params["val"]["batch_size"])
    test_dataloader = DataLoader(data_list_test, 
                            batch_size=1)

    return train_dataloader, val_dataloader, test_dataloader