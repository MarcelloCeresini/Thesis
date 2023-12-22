import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils
import torch_geometric.transforms as T

from .layers import CustomConv

class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.convs = nn.ModuleList() # to enable model.parameters() to access the params inside list
        self.convs.append(CustomConv(input_dim, hidden_dim))

        for _ in range(2):
            self.convs.append(CustomConv(hidden_dim, hidden_dim))

        self.num_layers = len(self.convs)

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # self.dropout = 0.25

    def forward(self, data: pyg_data.Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # if data.num_node_features == 0:
        #     x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x # for visualization
            x = F.relu(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.post_mp(x)

        return emb, x # NO softmax because it's regression
    
    def loss(self, pred:torch.Tensor, label:torch.Tensor, mask:torch.Tensor =None):
        if mask is not None:
            loss = F.mse_loss(pred, label, reduction="none")
            return (loss * mask).mean()
        else:
            return F.mse_loss(pred, label, reduction="mean")


        
