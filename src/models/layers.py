from typing import Any, Dict, List, Optional, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module

import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
from torch_geometric.nn.aggr import Aggregation
import torch_geometric.utils as pyg_utils
import torch_geometric.transforms as T

from torch_scatter.composite import scatter_softmax

class MLPConv(pyg_nn.MessagePassing):
    def __init__(self, 
        in_channels, 
        out_channels, 
        mid_channels, 
        edge_channels, 
        mlp: Module, 
        add_global_info: bool,
        add_BC_info: bool,
        skip: bool = True,
        aggr: str | List[str] | Aggregation | None = "mean", 
            *, aggr_kwargs: Dict[str, Any] | None = None, 
            flow: str = "source_to_target", node_dim: int = -2, decomposed_layers: int = 1, **kwargs):
        
        super().__init__(aggr, aggr_kwargs=aggr_kwargs, flow=flow, node_dim=node_dim, 
                        decomposed_layers=decomposed_layers, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.edge_channels = edge_channels
        self.mlp = mlp # mid_channels --> mid_channels
        self.skip = skip
        self.add_global_info = add_global_info
        self.add_BC_info = add_BC_info
        
        self.m_node_1 = nn.Linear(  in_features=self.in_channels,
                                    out_features=self.mid_channels,)
        self.m_node_2 = nn.Linear(  in_features=self.in_channels,
                                    out_features=self.mid_channels,)
        self.m_edge = nn.Linear(    in_features=self.edge_channels,
                                    out_features=self.mid_channels,)
        self.act_1 = nn.LeakyReLU()
        self.self_node_contrib = nn.Linear( in_features=self.in_channels,
                                            out_features=self.out_channels,)
        self.message_contrib = nn.Linear(   in_features=self.mid_channels,
                                            out_features=self.out_channels,)
        self.act_2 = nn.LeakyReLU()


    def forward(self, 
            x: torch.Tensor, 
            edge_index: torch.Tensor, 
            edge_attr: torch.Tensor,
            x_graph: Optional[torch.Tensor],
            x_BC: Optional[torch.Tensor],
            batch: Optional[torch.Tensor],
            *args, **kwargs) -> Any:

        if self.add_global_info:
            x += x_graph[batch]
        if self.add_BC_info:
            x += x_BC[batch]
        
        msg_i = self.m_node_1(x)
        msg_j = self.m_node_2(x)
        msg_e = self.m_edge(edge_attr)
        new_node_features = self.propagate( edge_index, 
                                            size=(x.shape[0], x.shape[0]),
                                            x=x,
                                            msg_i=msg_i, 
                                            msg_j=msg_j, 
                                            msg_e=msg_e,)
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
        if self.skip:
            out += x
        return self.act_2(out)
    # def edge_update(edge_index, size, kwargs):
    #     pass


class Simple_MLPConv(pyg_nn.MessagePassing):
    def __init__(self,     
        hidden_channels,
        mlp: nn.Module,
        mlp_update: nn.Module,
        attention: bool,
        add_global_info: bool,
        add_BC_info: bool,
        update_edges: Optional[bool] = None,
        mlp_edges: Optional[nn.Module] = None,
        skip: bool = True,
        k_heads: Optional[int] = 1,
        channels_per_head: Optional[int] = 1,
        edge_in_channels: Optional[int] = None,
        aggr: str | List[str] | Aggregation | None = "mean", 
            *, aggr_kwargs: Dict[str, Any] | None = None, 
            flow: str = "source_to_target", node_dim: int = -2, decomposed_layers: int = 1, **kwargs):
        super().__init__(aggr, aggr_kwargs=aggr_kwargs, flow=flow, node_dim=node_dim, 
                        decomposed_layers=decomposed_layers, **kwargs)

        self.in_channels = hidden_channels
        self.mlp = mlp
        self.mlp_update = mlp_update
        self.attention = attention
        self.add_global_info = add_global_info
        self.add_BC_info = add_BC_info
        self.update_edges = update_edges
        self.mlp_edges = mlp_edges
        self.skip = skip
        self.k_heads = k_heads
        self.edge_in_channels = edge_in_channels

        self.act_msg = nn.LeakyReLU()
        self.act_update = nn.LeakyReLU()
        self.act_edge_update = nn.LeakyReLU()

        if self.attention:
            self.k_heads = k_heads
            self.channels_per_head = channels_per_head
            self.att_channels = self.k_heads * self.channels_per_head
            self.Q_layer = nn.Linear(in_features=self.in_channels, out_features=self.att_channels, bias=False)
            self.K_layer = nn.Linear(in_features=self.in_channels, out_features=self.att_channels, bias=False)
            self.edge_layer = nn.Linear(in_features=self.edge_in_channels, out_features=self.k_heads, bias=False)
            # self.mlp_attention = nn.Sequential(
            #     nn.Linear(in_features=2*self.in_channels, out_features=self.att_channels, bias=False),
            #     nn.LeakyReLU(),
            #     nn.Linear(in_features=self.att_channels, out_features=self.k_heads, bias=False),)


    def forward(self, 
            x: torch.Tensor, 
            edge_index: torch.Tensor, 
            edge_attr: torch.Tensor,
            x_graph: Optional[torch.Tensor],
            x_BC: Optional[torch.Tensor],
            batch: Optional[torch.Tensor],
            *args, **kwargs) -> Any:
        
        x_graph_x = x_graph[batch]  if self.add_global_info     else torch.zeros((x.shape[0], 1)).to(device=x.device)
        x_BC_x = x_BC[batch]        if self.add_BC_info         else torch.zeros((x.shape[0], 1)).to(device=x.device)

        x_idx_i = torch.unsqueeze(torch.arange(x.shape[0], device=x.device), 1)

        new_node_features = self.propagate(
            edge_index, 
            size=(x.shape[0], x.shape[0]),
            x_i=x,
            x_j=x,
            x=x,
            edge_attr=edge_attr,
            x_graph_x=x_graph_x,
            x_BC_x=x_BC_x,
            x_graph_i=x_graph_x,
            x_BC_i=x_BC_x,
            x_idx_i=x_idx_i,
        )

        if self.update_edges:
            # TODO: add graph and BC also here?
            new_edge_features = self.edge_updater(
                edge_index,
                size=(x.shape[0], x.shape[0]),
                x_i=x,
                x_j=x,
                edge_attr=edge_attr, 
                ) 
        else:
            new_edge_features = edge_attr
        return {"x": new_node_features, "edge_attr": new_edge_features}


    def message(self, x_i, x_j, edge_attr, x_graph_i, x_BC_i, x_idx_i):

        tmp = torch.concat((x_i, x_j, edge_attr), dim=1)
        if self.add_global_info:
            tmp = torch.concat((tmp, x_graph_i), dim=1)
        if self.add_BC_info:
            tmp = torch.concat((tmp, x_BC_i), dim=1)
        
        msg = self.mlp(tmp)
        msg = self.act_msg(msg)

        # split msg by head --> in the building of the layer choose mlp out channels as divisible by k_heads
        # ex. each head has 4 channels --> 16 heads --> 64 total channels
        # then multiply the first 4 channels of the message by the first channel of the attention coeff.

        if self.attention:
            # TODO: check if dimensions are right and apply it not to message but to input features????
            shape = (x_i.shape[0], self.k_heads, self.channels_per_head)
            Q = self.Q_layer(x_i).view(*shape)
            K = self.K_layer(x_j).view(*shape)
            A = torch.mul(Q, K).sum(dim=2)/self.channels_per_head**(1/2)
            A += self.edge_layer(edge_attr)

            attention_coefficients = pyg_utils.softmax(A, x_idx_i[:,0], dim=0) # n_nodes * k_heads coefficients
            msg *= attention_coefficients.repeat_interleave(self.channels_per_head, dim=1)

            # attention_scores = self.mlp_attention(torch.concat((x_i, x_j), dim=1))
            # attention_coefficients = pyg_utils.softmax(attention_scores, x_idx_i[:,0], dim=0)
            # msg *= attention_coefficients.repeat_interleave(self.channels_per_head, dim=1)

        return msg

    def update(self, msg, x, x_graph_x, x_BC_x)-> Tensor:
        tmp = torch.concat((x, msg), dim=1)
        if self.add_global_info:
            tmp = torch.concat((tmp, x_graph_x), dim=1)
        if self.add_BC_info:
            tmp = torch.concat((tmp, x_BC_x), dim=1)
        
        update = self.mlp_update(tmp)
        if self.skip:
            update += x
        update = self.act_update(update)
        return update


    def edge_update(self, x_i, x_j, edge_attr) -> Tensor:
        tmp = torch.concat((edge_attr, x_i, x_j), dim=1)

        update = self.mlp_edges(tmp)
        if self.skip:
            update += edge_attr
        update = self.act_edge_update(update)
        return update
        
