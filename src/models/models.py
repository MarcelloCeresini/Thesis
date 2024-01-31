import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils
import torch_geometric.transforms as T

from .layers import CustomConv
from config_pckg.config_file import Config 
from loss_pckg.losses import MSE_loss


class BaselineModel(nn.Module):
    '''
    GCNConv (GraphConv uses skip connection for central node)
    No use of edge attributes ()
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, conf: Config) -> None:
        super().__init__()

        self.conf = conf

        self.convs = nn.ModuleList() # to enable model.parameters() to access the params inside list
        encode = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.convs.append(encode)

        for _ in range(self.conf.hyper_params["model"]["n_message_passing_layers"]):
            self.convs.append(pyg_nn.GCNConv(hidden_dim, hidden_dim))

        self.num_layers = len(self.convs)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # self.dropout = 0.25

    def forward(self, data: pyg_data.Data):
        x, x_mask, edge_index, edge_attr, batch  = data.x, data.x_mask, data.edge_index, data.edge_attr, data.batch

        # remove "component_id"
        x = x[:, :len(self.conf.graph_node_feature_dict)-1]
        x_mask = x_mask[:, :len(self.conf.graph_node_feature_dict)-1]
        
        x = torch.concat([x, x_mask], dim=1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x # for visualization
            x = F.relu(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.regressor(x)

        return emb, x # NO softmax because it's regression
    
    def loss(self, pred:torch.Tensor, label:torch.Tensor):
        return MSE_loss(pred, label)
        