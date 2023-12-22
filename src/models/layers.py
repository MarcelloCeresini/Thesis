from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
from torch_geometric.nn.aggr import Aggregation
import torch_geometric.utils as pyg_utils
import torch_geometric.transforms as T


class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels,
                 aggr: str | List[str] | Aggregation | None = "add", 
                 *, 
                 aggr_kwargs: Dict[str, Any] | None = None, 
                 flow: str = "source_to_target", 
                 node_dim: int = -2, 
                 decomposed_layers: int = 1, 
                 **kwargs):
        super().__init__(aggr, aggr_kwargs=aggr_kwargs, flow=flow, node_dim=node_dim, decomposed_layers=decomposed_layers, **kwargs)

        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x: [N, in_channels]
        # edge_index: [2, E]

        # Add self loops
        # edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=x.size(0))

        # Transform node_feature matrix
        x = self.lin(x)

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
    
    def message(self, x_j, edge_index, size): # given all nodes on which we want to compute the message + the edge_index
        # x_j: [E, out_channels]
        return x_j
    
    def update(self, aggr_out): # after message passing before output
        # aggr_out: [N, out_channels]
        return aggr_out



