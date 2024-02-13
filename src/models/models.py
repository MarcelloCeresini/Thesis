from inspect import getfullargspec
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils
import torch_geometric.transforms as T

from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean


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
            pos: torch.Tensor,
            batch: torch.Tensor,
            **kwargs
        ):
        # x, x_mask, edge_index, edge_attr, batch  = data.x, data.x_mask, data.edge_index, data.edge_attr, data.batch

        BC_idxs = x_mask[:, -1]
        x = torch.concat([x, x_mask], dim=1)

        X = {
            "x":            x,
            "edge_index":   edge_index,
            "edge_attr":    edge_attr,
            "pos":          pos,
            "batch":        batch,
        }
        # edge_attr is 3d relative distance between nodes + the norm (4 columns)

        # Encode features
        tmp = forward_for_general_layer(self.encoder, X)
        X.update(tmp)

        # Create graph-level and 'boundary' features
        # TODO: improve "boundary" by inserting relative distance (through pos) in computation
        # use a message passing with np_ones as adjacency matrix and give "pos_i" and "pos_j" inside propagate
        # then inside message give inside the MLP that encodes the message "pos_i - pos_j" as input
        # ALSO PASS BATCH otherwise you connect different graphs
        x_BC = X["x"][BC_idxs, :]
        batch_BC = X["batch"][BC_idxs]

        # FIXME: check if mean is done sample wise otherwise change it
        # BC is created at the beginning and never updated because it should encode the geometry (fixed during message passing)
        X.update({"x_BC": scatter_mean(x_BC, batch_BC, dim=0)}) # batch_size x num_features
        X.update({"x_graph": scatter_mean(X["x"], batch, dim=0)})
        
        # Process
        for _ in range(self.model_structure["message_passer"]["repeats_training"]):
            tmp = forward_for_general_layer(self.message_passer, X)
            X.update(tmp)
            X.update({"x_graph": scatter_mean(X["x"], batch, dim=0)}) # Graph encoding is updated even when not used
        
        # Decode
        tmp = forward_for_general_layer(self.decoder, X)
        X.update(tmp)
        
        return X["x"]


    def loss(self, pred:torch.Tensor, label:torch.Tensor):
        return eval(self.conf["hyperparams"]["loss"])(pred, label)
