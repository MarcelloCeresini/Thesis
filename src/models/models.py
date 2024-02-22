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

from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

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
    
    def __init__(self, input_dim, output_dim, model_structure: dict, conf, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO: improve definition from yaml: https://github.com/kaniblu/pytorch-models

        self.conf = conf
        self.model_structure = model_structure
        self.add_self_loops = self.model_structure["add_self_loops"]
        self.update_edges = self.model_structure["update_edges"] if "update_edges" in self.model_structure.keys() else False
        if self.update_edges:
            self.edges_channels = self.model_structure["edges_channels"]

        current_dim, self.encoder = get_obj_from_structure(
            in_channels=input_dim, 
            str_d=model_structure["encoder"],
            conf=conf,
            out_channels=model_structure["message_passer"]["out_channels"],
        )

        message_passer_structure = model_structure["message_passer"]
        if self.update_edges:
            _, self.edge_encoder = get_obj_from_structure(
                in_channels=conf["hyperparams"]["edge_feature_dim"], 
                str_d=model_structure["edge_encoder"],
                conf=conf,
                out_channels=self.edges_channels
            )
            message_passer_structure.update({
                "update_edges": True,
                "edges_channels": self.edges_channels,
            })
        else:
            message_passer_structure.update({
                "update_edges": False,})
        
        current_dim, self.message_passer = get_obj_from_structure(
            in_channels=current_dim, 
            str_d=message_passer_structure,
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

        if self.add_self_loops:
            edge_index, edge_attr = pyg_utils.add_self_loops(edge_index, edge_attr, 
                                                                fill_value=0.)

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

        if self.update_edges:
            tmp = forward_for_general_layer(self.edge_encoder, {"edge_attr": X["edge_attr"]})
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
        X.update({"x_BC": pyg_utils.scatter(x_BC, batch_BC, dim=0, reduce="mean")}) # batch_size x num_features
        X.update({"x_graph": pyg_nn.pool.global_mean_pool(X["x"], batch)})
        
        # Process
        for _ in range(self.model_structure["message_passer"]["repeats_training"]):
            tmp = forward_for_general_layer(self.message_passer, X)
            X.update(tmp)
            X.update({"x_graph": pyg_nn.pool.global_mean_pool(X["x"], batch)}) # Graph encoding is updated even when not used
        
        # Decode
        tmp = forward_for_general_layer(self.decoder, X)
        X.update(tmp)
        
        return X["x"]


    def loss(self, pred:torch.Tensor, label:torch.Tensor):
        return eval(self.conf["hyperparams"]["loss"])(pred, label)



class PINN(nn.Module):
    
    def __init__(self, net, conf, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO: improve definition from yaml: https://github.com/kaniblu/pytorch-models

        self.net = net
        self.conf = conf
        self.input_sampling = "all"
        self.output_sampling = "all"

    def get_grads(self, output, input, create_graph=True):
        return torch.autograd.grad(output, input, 
                                    create_graph=create_graph,
                                    grad_outputs=torch.ones_like(output),
                                    # allow_unused=True, # Now we don't use position for output so add this if it throws error
                                    )[0]

    # def forward(self, data: pyg_data.Data):
    def forward(self,
            x: torch.Tensor, 
            x_mask: torch.Tensor, 
            edge_index: torch.Tensor, 
            edge_attr: torch.Tensor,
            pos: torch.Tensor,
            batch: torch.Tensor,
            # triangulation: Delaunay, # TODO: if sampling = "all" precompute triangulation for each sample to ease computation 
            **kwargs
        ):

        pos = pos[:,:2]

        if self.input_sampling == "all":
            pass
        if self.output_sampling == "all":
            sampling_points = pos

        output = self.net(x, x_mask, edge_index, edge_attr, pos, batch)

        # TODO: improve this --> maybe better to sample cell and then point inside cell
        triangulation = Delaunay(pos)
        interpolator = LinearNDInterpolator(triangulation, output)
        U = torch.tensor(interpolator(sampling_points))

        # TODO: divide by batch
        U_pos = self.get_grads(U, pos)
        U_x, U_y = U_pos[:,0], U_pos[:,1]
        Uvel_xx = self.get_grads(U_x[:,:2], pos[:,0], create_graph=False)
        Uvel_yy = self.get_grads(U_y[:,:2], pos[:,1], create_graph=False)

        u, v = U[:,0], U[:,1]
        u_x, v_x, p_x = U_x[:,0], U_x[:,1], U_x[:,2]
        u_y, v_y, p_y = U_y[:,0], U_y[:,1], U_y[:,2]
        u_xx, v_xx = Uvel_xx[:,0], Uvel_xx[:,1]
        u_yy, v_yy = Uvel_yy[:,0], Uvel_yy[:,1]

        eps_1 = u*u_x + v*u_y + p_x - self.C * (u_xx + u_yy)
        eps_2 = u*v_x + v*v_y + p_y - self.C * (v_xx + v_yy)
        eps_3 = u_x + v_y

        return u, (eps_1, eps_2, eps_3)

    def loss(self, pred:torch.Tensor, label:torch.Tensor, residuals: tuple = (0)):
        # TODO: better to divide them and log them singularly, then sum and optimize
        loss = self.net.loss(pred, label)
        
        for residual in residuals:
            loss += residual.abs().sum()
        
        return loss
