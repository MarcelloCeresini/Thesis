from inspect import getfullargspec
from typing import Optional

import torch.nn as nn
import torch_geometric.nn as pyg_nn

from torch.nn import Linear, ReLU, Sequential, LeakyReLU, Softplus
from torch_geometric.nn import GCNConv, NNConv
from math import ceil

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
                case "Softplus":
                    obj = Softplus()
                    return in_channels, obj
                case _:
                    raise NotImplementedError()
        
        case "sequential_model":
            m_list = []
            curr_channels = in_channels

            if str_d["layers"] is not None:
                for layer_structure in str_d["layers"]:
                    if layer_structure is None:
                        continue
                    
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
                        standard_activation=str_d["standard_activation"],
                        aggr=str_d["aggr"],
                    )
                    return out_channels, obj
                case "Simple_MLPConv": # Simple_MLPConv_edges is deprecated
                    out_channels = str_d["out_channels"]
                    update_edges = str_d["update_edges"]
                    # edge_channels = str_d["edges_channels"] if update_edges else conf["hyperparams"]["feature_dim"]
                    #### If it doesn't work, remove line below and add line above
                    edge_channels = str_d["edges_channels"] if update_edges else conf["hyperparams"]["edge_feature_dim"]

                    in_channels_mlp = 2*out_channels + edge_channels
                    in_channels_mlp_update = out_channels
                    in_channels_mlp_edges = 2*out_channels + edge_channels

                    if str_d["add_global_info"]: # this is because we use concatenation
                        in_channels_mlp += out_channels
                        in_channels_mlp_update += out_channels
                    if str_d["add_BC_info"]:
                        in_channels_mlp += out_channels
                        in_channels_mlp_update += out_channels

                    if str_d["attention"]:
                        out_channels_mlp = str_d["channels_per_head"] * str_d["k_heads"]
                    else:
                        out_channels_mlp = out_channels

                    _, mlp = get_obj_from_structure(
                                in_channels=in_channels_mlp,
                                str_d=str_d["nn"],
                                conf=conf,
                                out_channels=out_channels_mlp,)

                    in_channels_mlp_update += out_channels_mlp

                    _, mlp_update = get_obj_from_structure(
                                in_channels=in_channels_mlp_update,
                                str_d=str_d["nn_update"],
                                conf=conf,
                                out_channels=out_channels,) # always out channels
                    
                    if update_edges:
                        _, mlp_edges = get_obj_from_structure(
                                    in_channels=in_channels_mlp_edges,
                                    str_d=str_d["nn_edges"],
                                    conf=conf,
                                    out_channels=edge_channels,)
                    
                    obj = Simple_MLPConv(
                        hidden_channels=in_channels,
                        mlp=mlp,
                        mlp_update=mlp_update,
                        mlp_edges=mlp_edges if update_edges else None,
                        attention=str_d["attention"],
                        add_global_info=str_d["add_global_info"],
                        add_BC_info=str_d["add_BC_info"],
                        update_edges=str_d["update_edges"],
                        skip=str_d["skip"],
                        k_heads=str_d.get("k_heads", None),
                        channels_per_head=str_d.get("channels_per_head", None),
                        edge_in_channels=edge_channels,
                        standard_activation=str_d["standard_activation"],
                        aggr=str_d["aggr"],
                    )
                    return out_channels, obj
        case "repeated_not_shared_layer":
            raise NotImplementedError()
        case _:
            raise NotImplementedError()


def forward_for_general_layer(layer, X_dict: dict):
    match layer:
        case Simple_MLPConv():
            return layer(
                x=          X_dict["x"], 
                edge_index= X_dict["edge_index"], 
                edge_attr=  X_dict["edge_attr"],
                x_graph=    X_dict.get("x_graph", None),
                x_BC=       X_dict.get("x_BC", None),
                batch=      X_dict.get("batch", None),
            )
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
        case Sequential() | Linear() | ReLU() | LeakyReLU():
            if "x" in X_dict.keys():
                return {"x": layer(X_dict["x"])}
            elif "edge_attr" in X_dict.keys():
                return {"edge_attr": layer(X_dict["edge_attr"])}
            else:
                raise NotImplementedError("Only 'x' and 'edge_attr' are implemented")
            