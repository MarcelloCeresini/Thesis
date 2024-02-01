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

from .layers import CustomConv
from config_pckg.config_file import Config 
from loss_pckg.losses import MSE_loss


def update_channels(spec: dict, new_input_channels, new_out_channels):
    spec.update({
        "in_channels": new_input_channels,
        "out_channels": new_out_channels,
        "in_features": new_input_channels,
        "out_features": new_out_channels
    })


def get_reduced_spec(obj, spec: dict):
    required_args = getfullargspec(obj).args
    known_args = set(spec.keys()).intersection(set(required_args))
    return {key: spec[key] for key in known_args}


def get_module_list(current_channels, structure: dict, output_channels = None):
    if output_channels is None:
        n_repeats = structure["repeats"]
    else:
        n_repeats = structure["repeats"]-1
    
    if "spec" in structure.keys():
        spec = structure["spec"]
    else:
        spec = {}

    module_list = nn.ModuleList()
    layer_class = eval(structure["name"])
    out_channels = structure["hidden_dim"]
    update_channels(spec, current_channels, out_channels)
    layer_instance = layer_class(**get_reduced_spec(layer_class, spec))
    current_channels = out_channels

    for _ in range(n_repeats):
        module_list.append(layer_instance)
        module_list.append(nn.ReLU())

    if output_channels is not None:
        update_channels(spec, current_channels, output_channels)
        module_list.append(
            layer_class(**get_reduced_spec(layer_class, spec)
        ))
        current_channels = output_channels
    
    return module_list, current_channels


def forward_for_general_layer(layer, X_dict):
    match layer:
        case GCNConv():
            x = layer(
                X_dict["x"], 
                X_dict["edge_index"], 
                X_dict["edge_attr"][:,3]**(-2) # weight = 1/distance**2
            )
            return {"x": x}
        case Linear():
            return {"x": layer(X_dict["x"])}
        case ReLU():
            return {"x": layer(X_dict["x"])}
        case _:
            raise NotImplementedError()
        


class BaselineModel(nn.Module):
    '''
    GCNConv (GraphConv uses skip connection for central node)
    No use of edge attributes ()
    '''
    def __init__(self, input_dim, output_dim, model_structure: dict, conf, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO: improve definition from yaml: https://github.com/kaniblu/pytorch-models

        self.conf = conf

        self.encoder_layers, current_dim = get_module_list(
            input_dim, model_structure["encoder"], output_channels=model_structure["message_passer"]["hidden_dim"])
        
        self.message_passer_layers, current_dim = get_module_list(
            current_dim, model_structure["message_passer"])
        
        self.decoder_layers, current_dim = get_module_list(
            current_dim, model_structure["decoder"], output_channels=output_dim
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
        x      =      x[:, :len(self.conf.graph_node_feature_dict)-1]
        x_mask = x_mask[:, :len(self.conf.graph_node_feature_dict)-1]
        x = torch.concat([x, x_mask], dim=1)

        X = {
            "x":x,
            "edge_index":edge_index,
            "edge_attr":edge_attr
        }
        # edge_attr is 3d relative distance between nodes + the norm (4 columns)

        for layer in self.encoder_layers:
            tmp = forward_for_general_layer(layer, X)
            X.update(tmp)
        for layer in self.message_passer_layers:
            tmp = forward_for_general_layer(layer, X)
            X.update(tmp)
        for layer in self.decoder_layers:
            tmp = forward_for_general_layer(layer, X)
            X.update(tmp)
        
        return X["x"] # NO softmax because it's regression
    

    def loss(self, pred:torch.Tensor, label:torch.Tensor):
        return MSE_loss(pred, label)
        