from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
from torch_geometric.nn.aggr import Aggregation
import torch_geometric.utils as pyg_utils
import torch_geometric.transforms as T


class MLPConv(pyg_nn.MessagePassing):
    def __init__(
            self, 
            in_channels,
            out_channels,
            mid_channels,
            edge_channels,
            mlp: nn.Module,
            aggr: str | List[str] | Aggregation | None = "add", 
            *, 
            aggr_kwargs: Dict[str, Any] | None = None, 
            flow: str = "source_to_target", 
            node_dim: int = -2, 
            decomposed_layers: int = 1, 
            **kwargs):
        
        super().__init__(
            aggr, 
            aggr_kwargs=aggr_kwargs, 
            flow=flow, 
            node_dim=node_dim, 
            decomposed_layers=decomposed_layers, 
            **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.edge_channels = edge_channels
        self.mlp = mlp
        
        self.m_node_1 = nn.Linear(
            in_features=self.in_channels,
            out_features=self.mid_channels,
        )

        self.m_node_2 = nn.Linear(
            in_features=self.in_channels,
            out_features=self.mid_channels,
        )

        self.m_edge = nn.Linear(
            in_features=self.edge_channels,
            out_features=self.mid_channels,
        )

        self.act_1 = nn.ReLU()

        self.self_node_contrib = nn.Linear(
            in_features=self.in_channels,
            out_features=self.out_channels,
        )

        self.message_contrib = nn.Linear(
            in_features=self.mid_channels,
            out_features=self.out_channels,
        )

        self.act_2 = nn.ReLU()


    def forward(self, x, edge_index, edge_attr, *args, **kwargs) -> Any:
        '''
        x : Tensor
            node features (n_nodes * n_node_features)
        edge_index: Tensor
            edge indexes (2 * n_edges)
        edge_attrs:
            edge features (n_edges * n_edge_features)
        '''

        msg_i = self.m_node_1(x)
        msg_j = self.m_node_2(x)
        msg_e = self.m_edge(edge_attr)

        new_node_features = self.propagate(
            edge_index, 
            size=(x.shape[0], x.shape[0]),
            x=x,
            msg_i=msg_i, 
            msg_j=msg_j, 
            msg_e=msg_e, 
        )

        # new_edge_features = self.edge_updater(edge_index, size=(x.shape[0], x.shape[0]) )
        return new_node_features

    def message(self, msg_i, msg_j, msg_e):
        msg = msg_i + msg_j + msg_e
        msg = self.act_1(msg)
        msg = self.mlp(msg)
        return msg


    def update(self, msg, x):
        out_1 = self.self_node_contrib(x)
        out_2 = self.message_contrib(msg)
        out = out_1 + out_2
        return self.act_2(out)
    
    def edge_update(edge_index, size, kwargs):
        pass


