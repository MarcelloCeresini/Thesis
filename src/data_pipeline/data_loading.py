import os, glob

from typing import Any, Callable, Optional
from config_pckg.config_file import Config

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

# class CFDDataSet(InMemoryDataset):
#     # FIXME: how to make the dataset work?
#     '''
#     Wants in INPUT the pyg_Data files (inside raw there should be data .pt files, while in processed only one file of the entire dataset)!!!!
#     '''
#     @property
#     def raw_file_names(self):
#         tmp = glob.glob(pathname="*.pt", root_dir=os.path.join(self.root, "raw"))
#         return tmp
    
#     @property
#     def processed_file_names(self):
#         return ["complete_dataset.pt"]
    
#     def __init__(self, 
#                  root: str | None = None, 
#                  transform: Callable[..., Any] | None = None, 
#                  pre_transform: Callable[..., Any] | None = None, 
#                  pre_filter: Callable[..., Any] | None = None, 
#                  log: bool = True):
#         super().__init__(root, transform, pre_transform, pre_filter, log)

#     # def download(self):
#     #     raise NotImplementedError(f"Put the files directly in {os.path.join(self.root, 'raw')}")
    
#     def process(self):
#         # for mesh_filename in self.raw_file_names:
#         #     graph_filename = os.path.join(mesh_filename.split("")[0]+"pt")
#         #     convert_msh_to_graph(mesh_filename, graph_filename)

#         data_list = []
#         for data_filename in self.raw_file_names:
#             data_path = os.path.join(self.raw_dir, data_filename)
#             data_list.append(torch.load(data_path))

#         self.save(data_list, self.processed_paths[0])
    

def get_data_loaders(conf: Config, load_from_disk = False, save_to_disk = False) -> None:
    # TODO: implement transformation to add labels during loader creation
    # to avoid saving n different graphs for n different snapshots of the same simulation

    loaders_found = False
    if load_from_disk:
        if os.path.isfile(conf.standard_dataloader_path):
            train_dataloader, val_dataloader, test_dataloader = torch.load(conf.standard_dataloader_path)
            loaders_found = True
        else:
            print("No dataloaders at standard path, creating them from scratch")
    
    if (not load_from_disk) or (not loaders_found):
        data_filenames = sorted(glob.glob(pathname="*.pt", root_dir=conf.EXTERNAL_FOLDER_GRAPHS))

        data_list_train = [torch.load(os.path.join(conf.EXTERNAL_FOLDER_GRAPHS, data_filenames[idx])) for idx in conf.split_idxs["train"]]
        data_list_val   = [torch.load(os.path.join(conf.EXTERNAL_FOLDER_GRAPHS, data_filenames[idx])) for idx in conf.split_idxs["val"]]
        data_list_test  = [torch.load(os.path.join(conf.EXTERNAL_FOLDER_GRAPHS, data_filenames[idx])) for idx in conf.split_idxs["test"]]

        train_dataloader = DataLoader(data_list_train, 
                                batch_size=conf.hyper_params["training"]["batch_size"],
                                shuffle=True)
        val_dataloader  = DataLoader(data_list_val,  
                                batch_size=conf.hyper_params["val"]["batch_size"])
        test_dataloader = DataLoader(data_list_test, 
                                batch_size=1)
    
    if save_to_disk:
        torch.save(
            (train_dataloader, val_dataloader, test_dataloader),
            conf.standard_dataloader_path
            )

    return train_dataloader, val_dataloader, test_dataloader