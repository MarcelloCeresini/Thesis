import os, glob

from typing import Any, Callable, Optional
from config_pckg.config_file import Config
from utils import convert_msh_to_graph

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

class CFDDataSet(InMemoryDataset):
    # FIXME: how to make the dataset work?
    '''
    Wants in INPUT the pyg_Data files (inside raw there should be data .pt files, while in processed only one file of the entire dataset)!!!!
    '''
    @property
    def raw_file_names(self):
        tmp = glob.glob(pathname="*.pt", root_dir=os.path.join(self.root, "raw"))
        return tmp
    
    @property
    def processed_file_names(self):
        return ["complete_dataset.pt"]
    
    def __init__(self, 
                 root: str | None = None, 
                 transform: Callable[..., Any] | None = None, 
                 pre_transform: Callable[..., Any] | None = None, 
                 pre_filter: Callable[..., Any] | None = None, 
                 log: bool = True):
        super().__init__(root, transform, pre_transform, pre_filter, log)

    # def download(self):
    #     raise NotImplementedError(f"Put the files directly in {os.path.join(self.root, 'raw')}")
    
    def process(self):
        # for mesh_filename in self.raw_file_names:
        #     graph_filename = os.path.join(mesh_filename.split("")[0]+"pt")
        #     convert_msh_to_graph(mesh_filename, graph_filename)

        data_list = []
        for data_filename in self.raw_file_names:
            data_path = os.path.join(self.raw_dir, data_filename)
            data_list.append(torch.load(data_path))

        self.save(data_list, self.processed_paths[0])
    

def get_data_loaders(conf: Config) -> None:
    # TODO: implement transformation to add labels during loader creation
    # to avoid saving n different graphs for n different snapshots of the same simulation
    data_filenames = glob.glob(pathname="*.pt", root_dir=os.path.join(conf.DATA_DIR, "raw"))
    data_filenames = data_filenames*4

    # TODO: check that ALL ATTRIBUTES INSIDE DATA are tensors! (torch.float32)
    data_list = [torch.load(os.path.join(conf.DATA_DIR, "raw", filename)) for filename in data_filenames]
    
    # TODO: add train/val/test splits
    # batch_size = N° of GRAPHS to feed

    dataloader = DataLoader(data_list, 
                            batch_size=conf.hyper_params["training"]["batch_size"],
                            shuffle=True)

    return dataloader