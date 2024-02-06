from inspect import getfullargspec

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils
import torch_geometric.transforms as T

from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv

from config_pckg.config_file import Config 
from loss_pckg.losses import MSE_loss


from .model_utils import get_obj_from_structure, forward_for_general_layer


def get_model_instance(full_conf):
    return eval(full_conf["model"]["name"])(
        input_dim = full_conf["hyperparams"]["input_dim"],
        output_dim = full_conf["hyperparams"]["output_dim"],
        model_structure = full_conf["model"],
        conf = full_conf
    )

class EncodeProcessDecode_Baseline(nn.Module):
    '''
    GCNConv (GraphConv uses skip connection for central node)
    No use of edge attributes ()
    '''
    def __init__(self, input_dim, output_dim, model_structure: dict, conf, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO: improve definition from yaml: https://github.com/kaniblu/pytorch-models

        self.conf = conf
        self.model_structure = model_structure

        current_dim, self.encoder = get_obj_from_structure(
            in_channels=input_dim, 
            str_d=model_structure["encoder"],
            conf=conf,
            out_channels=model_structure["message_passer"]["out_channels"]
        )
        
        current_dim, self.message_passer = get_obj_from_structure(
            in_channels=current_dim, 
            str_d=model_structure["message_passer"],
            conf=conf,
        )
        
        _, self.decoder = get_obj_from_structure(
            in_channels=current_dim, 
            str_d=model_structure["decoder"], 
            conf=conf,
            out_channels=output_dim,
        )


    # def forward(self, data: pyg_data.Data):
    def forward(self, 
            x: torch.Tensor, 
            x_mask: torch.Tensor, 
            edge_index: torch.Tensor, 
            edge_attr: torch.Tensor, 
            batch: torch.Tensor
        ):
        # x, x_mask, edge_index, edge_attr, batch  = data.x, data.x_mask, data.edge_index, data.edge_attr, data.batch

        # remove "component_id"
        x      =      x[:, :-1]
        x_mask = x_mask[:, :-1]
        x = torch.concat([x, x_mask], dim=1)

        X = {
            "x":x,
            "edge_index":edge_index,
            "edge_attr":edge_attr
        }
        # edge_attr is 3d relative distance between nodes + the norm (4 columns)

        tmp = forward_for_general_layer(self.encoder, X)
        X.update(tmp)

        for _ in range(self.model_structure["message_passer"]["repeats_training"]):
            tmp = forward_for_general_layer(self.message_passer, X)
            X.update(tmp)
        
        tmp = forward_for_general_layer(self.decoder, X)
        X.update(tmp)
        
        return X["x"] # NO softmax because it's regression


    def loss(self, pred:torch.Tensor, label:torch.Tensor):
        return eval(self.conf["hyperparams"]["loss"])(pred, label)
        