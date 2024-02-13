from inspect import getfullargspec
from typing import Optional

import torch.nn as nn
import torch_geometric.nn as pyg_nn

from torch.nn import Linear, ReLU, Sequential, LeakyReLU
from torch_geometric.nn import GCNConv, NNConv

from config_pckg.config_file import Config
from .layers import MLPConv, Simple_MLPConv

def get_only_required_kwargs(obj, complete_kwargs: dict):
    required_args = getfullargspec(obj).args
    known_args = set(complete_kwargs.keys()).intersection(set(required_args))
    return {key: complete_kwargs[key] for key in known_args}


def get_obj_from_structure(
        in_channels: int, 
        str_d: dict, 
        conf: Config, 
        out_channels: Optional[int] = None
    ):

    match str_d["type"]:
        case "layer":
            match str_d["name"]:
                case "Linear":
                    out_channels = str_d["out_features"]
                    obj = Linear(in_features=in_channels, out_features=out_channels)
                    return out_channels, obj
                case "ReLU":
                    obj = ReLU()
                    return in_channels, obj
                case "LeakyReLU":
                    obj = LeakyReLU()
                    return in_channels, obj
                case _:
                    raise NotImplementedError()
        
        case "sequential_model":
            m_list = []
            curr_channels = in_channels
            for layer_structure in str_d["layers"]:
                curr_channels, layer = get_obj_from_structure(
                    in_channels=curr_channels,
                    str_d=layer_structure,
                    conf=conf
                )
                m_list.append(layer)
            
            if out_channels is not None:
                m_list.append(nn.Linear(in_features=curr_channels, out_features=out_channels))
                curr_channels = out_channels
            
            return curr_channels, nn.Sequential(*m_list)

        case "repeated_shared_layer":
            match str_d["name"]:
                case "GCNConv":
                    out_channels = str_d["out_channels"]
                    obj = pyg_nn.GCNConv(
                            in_channels=in_channels, 
                            out_channels=out_channels
                        )
                    return out_channels, obj
                        
                case "NNConv":
                    out_channels_conv = str_d["out_channels"]
                    _, mlp = get_obj_from_structure(
                                in_channels=conf["hyperparams"]["feature_dim"],
                                str_d=str_d["nn"],
                                conf=conf,
                                out_channels=in_channels*str_d["out_channels"]
                            )
                    obj = pyg_nn.NNConv(
                            in_channels=in_channels,
                            out_channels=out_channels_conv,
                            nn=mlp
                        )
                    return out_channels_conv, obj
                        
                case "MLPConv":
                    out_channels = str_d["out_channels"]
                    _, mlp = get_obj_from_structure(
                                in_channels=str_d["mid_channels"],
                                str_d=str_d["nn"],
                                conf=conf,
                                out_channels=str_d["mid_channels"],
                            )
                    obj = MLPConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        mid_channels=str_d["mid_channels"],
                        edge_channels=conf["hyperparams"]["feature_dim"],
                        skip=str_d["skip"],
                        add_global_info=str_d["add_global_info"],
                        add_BC_info=str_d["add_BC_info"],
                        mlp=mlp,
                        aggr=str_d["aggr"],
                    )
                    return out_channels, obj
                case "Simple_MLPConv": # Simple_MLPConv_edges is deprecated
                    out_channels = str_d["out_channels"]
                    in_channels_mlp = 2*out_channels + conf["hyperparams"]["feature_dim"]
                    in_channels_mlp_update = 2*out_channels
                    if str_d["add_global_info"]: # this is because we use concatenation
                        in_channels_mlp += out_channels
                        in_channels_mlp_update += out_channels
                    if str_d["add_BC_info"]:
                        in_channels_mlp += out_channels
                        in_channels_mlp_update += out_channels
                    _, mlp = get_obj_from_structure(
                                in_channels=in_channels_mlp,
                                str_d=str_d["nn"],
                                conf=conf,
                                out_channels=out_channels,
                            )
                    _, mlp_update = get_obj_from_structure(
                                in_channels=in_channels_mlp_update,
                                str_d=str_d["nn_update"],
                                conf=conf,
                                out_channels=out_channels,
                            )
                    obj = Simple_MLPConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        mlp=mlp,
                        mlp_update=mlp_update,
                        attention=str_d["attention"],
                        add_global_info=str_d["add_global_info"],
                        add_BC_info=str_d["add_BC_info"],
                        skip=str_d["skip"],
                        k_heads=str_d["k_heads"],
                        aggr=str_d["aggr"],
                    )
                    return out_channels, obj
        case "repeated_not_shared_layer":
            raise NotImplementedError()
        case _:
            raise NotImplementedError()


def forward_for_general_layer(layer, X_dict):
    match layer:
        case Simple_MLPConv():
            x = layer(
                x=          X_dict["x"], 
                edge_index= X_dict["edge_index"], 
                edge_attr=  X_dict["edge_attr"],
                x_graph=    X_dict["x_graph"],
                x_BC=       X_dict["x_BC"],
                batch=      X_dict["batch"],
            )
            return {"x": x}
        case MLPConv():
            x = layer(
                x=          X_dict["x"], 
                edge_index= X_dict["edge_index"], 
                edge_attr=  X_dict["edge_attr"],
                x_graph=    X_dict["x_graph"],
                x_BC=       X_dict["x_BC"],
                batch=      X_dict["batch"],
            )
            return {"x": x}
        case GCNConv():
            x = layer(
                x=              X_dict["x"], 
                edge_index=     X_dict["edge_index"], 
                edge_weight=    X_dict["edge_attr"][:,3]**(-2) # weight = 1/distance**2
            )
            return {"x": x}
        case NNConv():
            x = layer(
                x=          X_dict["x"], 
                edge_index= X_dict["edge_index"], 
                edge_attr=  X_dict["edge_attr"],
            )
            return {"x": x}
        case Sequential():
            return {"x": layer(X_dict["x"])}
        case Linear():
            return {"x": layer(X_dict["x"])}
        case ReLU():
            return {"x": layer(X_dict["x"])}
        case LeakyReLU():
            return {"x": layer(X_dict["x"])}
        case _:
            raise NotImplementedError()
        