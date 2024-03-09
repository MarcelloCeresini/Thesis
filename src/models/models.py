from inspect import getfullargspec
from typing import Literal, Optional
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
from torch.func import functional_call, vmap, jacrev, hessian

from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import pyvista

from config_pckg.config_file import Config 
from loss_pckg.losses import MSE_loss
from .model_utils import get_obj_from_structure, forward_for_general_layer


def get_model_instance(model_conf):
    net = eval(model_conf["model"]["name"])(
        input_dim = model_conf["hyperparams"]["input_dim"],
        output_dim = model_conf["hyperparams"]["output_dim"],
        model_structure = model_conf["model"],
        conf = model_conf
    )

    if model_conf["model"]["PINN"] == True:
        return PINN(net=net, conf=model_conf)
    else:
        return net


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



class EPD_with_sampling(nn.Module):
    
    def __init__(self, input_dim, output_dim, model_structure: dict, conf, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO: improve definition from yaml: https://github.com/kaniblu/pytorch-models

        self.conf = conf
        self.model_structure = model_structure
        self.add_self_loops = self.model_structure["add_self_loops"]
        self.update_edges = self.model_structure["update_edges"] if "update_edges" in self.model_structure.keys() else False
        

        if self.update_edges:
            self.edges_channels = self.model_structure["edges_channels"]

        _, self.positional_encoder = get_obj_from_structure(
            in_channels=2, 
            str_d=model_structure["positional_encoder"],
            conf=conf,
            out_channels=model_structure["message_passer"]["out_channels"],
        )

        match self.conf["hyperparams"]["inference_mode_latent_sampled_points"]:
            case "squared_distance":
                self.graph_sampling_p_for_interpolation = self.conf["hyperparams"]["graph_sampling_p_for_interpolation"]
            case "fourier_features":
                self.use_fourier_features = True
                self.fourier_pos_enc_matrix = torch.nn.Parameter(
                    torch.nn.init.normal_(
                        tensor=torch.zeros(2, self.conf["hyperparams"]["fourier_feat_dim"]), 
                            mean=0, std=1
                    ))
                # self.fourier_pos_enc_matrix = torch.nn.Parameter(
                #         torch.zeros(2, self.conf["hyperparams"]["fourier_feat_dim"]), 
                #     )

                # input_dim += 2*self.conf["hyperparams"]["fourier_feat_dim"]
        
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


    def get_fourier_pos_enc(self, x, learnable_matrix) -> torch.Tensor:
        '''https://arxiv.org/pdf/2106.02795.pdf'''
        return torch.concatenate(
                (torch.cos(x@learnable_matrix), torch.sin(x@learnable_matrix)), dim=-1
            ) / (2*learnable_matrix.shape[1])**(0.5)


    # def forward(self, data: pyg_data.Data):
    def forward(self,
            x: torch.Tensor, 
            x_mask: torch.Tensor, 
            edge_index: torch.Tensor, 
            edge_attr: torch.Tensor,
            pos: torch.Tensor,
            batch: torch.Tensor,
            sampling_points: Optional[torch.Tensor] = None,
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

        if self.use_fourier_features:
            input_fourier_features = \
                self.get_fourier_pos_enc(pos, self.fourier_pos_enc_matrix)
            # X["x"] = torch.concat((X["x"], input_fourier_features), dim=1)
        
        # Encode features
        tmp = forward_for_general_layer(self.encoder, X)
        X.update(tmp)

        if self.update_edges:
            tmp = forward_for_general_layer(self.edge_encoder, {"edge_attr": X["edge_attr"]})
            X.update(tmp)
        
        x_BC = X["x"][BC_idxs, :]
        batch_BC = X["batch"][BC_idxs]

        # FIXME: check if mean is done sample wise otherwise change it
        # BC is created at the beginning and never updated because it should encode the geometry (fixed during message passing)
        X.update({"x_BC": pyg_utils.scatter(x_BC, batch_BC, dim=0, reduce="mean")}) # batch_size x num_features
        X.update({"x_graph": pyg_nn.pool.global_mean_pool(X["x"], batch)})
        
        # Process
        for _ in range(self.model_structure["message_passer"]["repeats_training"]):
            processed = forward_for_general_layer(self.message_passer, X)
            # processed = {k: v / torch.linalg.vector_norm(v, dim=-1, keepdim=True) for k,v in processed.items()}
            X.update(processed)
            X.update({"x_graph": pyg_nn.pool.global_mean_pool(X["x"], batch)}) # Graph encoding is updated even when not used
        
        decoded = forward_for_general_layer(self.decoder, X)
        
        if sampling_points is not None:

            match self.conf["hyperparams"]["inference_mode_latent_sampled_points"]:
                case "squared_distance":
                    # FIXME: we NEED knn to be scalable, but with functional_call+vmap it doesn't work
                    # positional_encoding_graph_points = self.positional_encoder(pos)
                    # positional_encoding_sampling_points = self.positional_encoder(sampling_points)
                    
                    n_nodes = pos.shape[0]
                    num_samples = int(n_nodes*self.graph_sampling_p_for_interpolation)
                    sampled_graph_pos = torch.multinomial(
                        torch.ones(n_nodes, dtype=torch.float), num_samples, replacement=False) # as np.random.choice
                    
                    squared_spatial_distance = torch.pow(sampling_points - pos[sampled_graph_pos,:], 2).sum(dim=-1)
                    # spatial_diff = sampling_points - pos[sampled_graph_pos,:]
                    # latent_diff = positional_encoding_sampling_points - positional_encoding_graph_points
                    # squared_spatial_distance = (spatial_diff*spatial_diff).sum(dim=-1)
                    # squared_latent_distance = (latent_diff*latent_diff).sum(dim=-1, keepdim=True)

                    weights = 1.0 / torch.clamp(squared_spatial_distance, min=1e-16)
                    weights /= weights.sum()
                    # weights = 1.0 / torch.clamp(squared_spatial_distance + squared_latent_distance, min=1e-16)
                    sampled_points_encoding = torch.einsum("ij,i->j", 
                        processed["x"][sampled_graph_pos,:], weights.to(torch.float32))

                case "fourier_features":
                    # https://arxiv.org/pdf/2007.14902.pdf --> linear approximated attention
                    # the approximation is in the softmax, that is transformed in its first-order Taylor series
                    # similarity_func(q,k) = e**(q@k) ~~ 1+q@k. 
                    # To assure its positivity (because it is a similarity measure) we normalize, so that q@k >= -1.
                    # Then we simplify and compute keys/values AGGREGATED quantities to re-use for all queries
                    # This is even better in our case because it's cross attention, so we can have N_queries >> N_keys if we wish
                    # And still keep memory requirements linear with N_keys

                    # OSS: move all except results to double, to avoid NaNs
                    sampling_points_fourier_features = \
                        self.get_fourier_pos_enc(sampling_points, self.fourier_pos_enc_matrix)

                    query = sampling_points_fourier_features.to(torch.float64)        # [1,K]
                    keys = input_fourier_features.to(torch.float64)                   # [N,K]
                    values = processed["x"].to(torch.float64)                         # [N,V]
                    N = keys.shape[0]

                    norm_query = query / torch.linalg.vector_norm(query, keepdim=True)
                    norm_keys = keys / torch.linalg.vector_norm(keys, dim=1, keepdim=True)

                    vals_sum = torch.sum(values, dim=0)             # [1,V]
                    key_weighted_vals_sum = norm_keys.T @ values    # [K,V]
                    keys_sum = torch.sum(norm_keys, dim=0)          # [1,K]

                    sampled_points_encoding = (vals_sum + norm_query@key_weighted_vals_sum) \
                        / (N + norm_query@keys_sum)
                    
                case "baseline_positional_encoder":
                    raise NotImplementedError("need to debug")
                    sampled_points_encoding = self.positional_encoder(sampling_points)

            U_sampled = forward_for_general_layer(self.decoder, {"x": sampled_points_encoding.to(torch.float32)})
            return U_sampled["x"], decoded["x"]
        else:
            return decoded["x"]

            
    def loss(self, pred:torch.Tensor, label:torch.Tensor):
        return eval(self.conf["hyperparams"]["loss"])(pred, label)


class PINN(nn.Module):
    
    def __init__(self, net:nn.Module, conf, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO: improve definition from yaml: https://github.com/kaniblu/pytorch-models

        self.net = net
        self.conf = conf

        self.domain_sampling = self.conf["hyperparams"]["domain_sampling"]
        self.boundary_sampling = self.conf["hyperparams"]["boundary_sampling"]

        self.Re = 1/(1.45e-5) # V_char=1, L_char=1
        self.loss_weights = {"continuity":1, "momentum_x":1, "momentum_y":1}


    def get_BC_residuals_single_sample(self, x, x_mask, u, v, p, u_x, v_y, p_x, p_y):
        feat_dict = self.conf['hyperparams']["feat_dict"]
        mask_dict = self.conf['hyperparams']["mask_dict"]

        mask_dict_copy = mask_dict.copy()
        mask_dict_copy.pop("is_BC")

        t_x, t_y = x[feat_dict["tangent_versor_x"]], x[feat_dict["tangent_versor_y"]]
        n_x, n_y = -t_y, +t_x

        residual = {k: torch.zeros_like(p_x, device=p_x.device) for k in mask_dict_copy}
        c = torch.zeros_like(p_x, device=p_x.device)
        # TODO: dynamic weights for these components?
        for k in mask_dict_copy:
            match k:
                case "v_t":
                    v_t_pred = (t_x*u + t_y*v)
                    residual["v_t"] += (v_t_pred - x[feat_dict["v_t"]]).abs()       * x_mask[mask_dict[k]]
                    c += x_mask[mask_dict[k]]
                case "v_n": # "inward" is not a problem with walls because v_n should always be 0 --> no matter the direction 
                    v_n_pred = (n_x*u + n_y*v)
                    residual["v_n"] += (v_n_pred - x[feat_dict["v_n"]]).abs()       * x_mask[mask_dict[k]]
                    c += x_mask[mask_dict[k]]
                case "p":
                    residual["p"] += (p - x[feat_dict["p"]]).abs()                  * x_mask[mask_dict[k]]
                    c += x_mask[mask_dict[k]]
                case "dv_dn": # pressure-outlet = n_x*U_x + n_y*U_y = n_x*u_x + n_y*v_y
                    dv_dn_pred = (n_x*u_x + n_y*v_y)
                    residual["dv_dn"] += (dv_dn_pred - x[feat_dict["dv_dn"]]).abs() * x_mask[mask_dict[k]]
                    c += x_mask[mask_dict[k]]
                    # should be torch.dot(normal, (u_x, v_y)), but pressure-outlet is vertical
                    # residual += u_x.square()
                case "dp_dn": # 
                    dp_dn_pred = (n_x*p_x + n_y*p_y)
                    residual["dp_dn"] += (dp_dn_pred - x[feat_dict["dp_dn"]]).abs() * x_mask[mask_dict[k]]
                    c += x_mask[mask_dict[k]]
        
        residual.update({"total":sum(residual.values())/c})
        return residual


    def forward(self,
            x: torch.Tensor,
            x_mask: torch.Tensor, 
            edge_index: torch.Tensor, 
            edge_attr: torch.Tensor,
            pos: torch.Tensor,
            batch: torch.Tensor,
            triangulated_cells: Optional[torch.Tensor] = None,
            # triangulation: Delaunay, # TODO: if sampling = "all" precompute triangulation for each sample to ease computation 
            **kwargs
        ):

        pos = pos[:,:2]
        edge_attr = edge_attr[:, torch.tensor([0,1,3])]
        
        if kwargs.get("domain_sampling_points", None) is not None:
            domain_sampling_points = torch.autograd.Variable(kwargs["domain_sampling_points"], requires_grad=True)
            n_domain_points = domain_sampling_points.shape[0]
        else:
            domain_sampling_points = torch.autograd.Variable(torch.FloatTensor())
            n_domain_points = 0

        if kwargs.get("boundary_sampling_points", None) is not None:
            idxs_boundary_sampled = kwargs["idxs_boundary_sampled"]
            x_BC, x_mask_BC = x[idxs_boundary_sampled], x_mask[idxs_boundary_sampled]

            boundary_sampling_points = torch.autograd.Variable(kwargs["boundary_sampling_points"], requires_grad=True)
            n_boundary_points = boundary_sampling_points.shape[0]
        else:
            boundary_sampling_points = torch.autograd.Variable(torch.FloatTensor())
            n_boundary_points = 0

        sampling_points = torch.concatenate((domain_sampling_points, boundary_sampling_points))
        domain_slice = torch.arange(n_domain_points)
        boundary_slice = torch.arange(start=n_domain_points, end=n_domain_points+n_boundary_points)
        
        model_params = ({k: v for k, v in self.net.named_parameters()}, 
                        {k: v for k, v in self.net.named_buffers()},)

        # import matplotlib.pyplot as plt
        # a = domain_sampling_points.detach().numpy()
        # b = boundary_sampling_points.detach().numpy()
        # plt.scatter(a[:,0], a[:,1], color="b")
        # plt.scatter(b[:,0], b[:,1], color="r")

        def compute_output(sampling_points):
            out_samp, out_sup = functional_call(
                self.net,
                model_params,
                (x, x_mask, edge_index, edge_attr, pos, batch, sampling_points),
            )
            return out_samp, (out_samp, out_sup)


        def compute_first_derivative(sampling_points):
            grads_samp, (out_samp, out_sup) = \
                jacrev(compute_output, has_aux=True)(sampling_points)
            return grads_samp, (grads_samp, out_samp, out_sup)


        def compute_second_derivative(sampling_points):
            hess_samp, (grads_samp, out_samp, out_sup) = \
                jacrev(compute_first_derivative, has_aux=True)(sampling_points)
            return hess_samp, grads_samp, out_samp, out_sup
        

        def get_correct_slice(slice_idxs, hess_samp, grads_samp, out_samp):
            u, v, p = out_samp[slice_idxs,0], out_samp[slice_idxs,1], out_samp[slice_idxs,2]
            u_x, u_y = grads_samp[slice_idxs,0,0], grads_samp[slice_idxs,0,1]
            v_x, v_y = grads_samp[slice_idxs,1,0], grads_samp[slice_idxs,1,1]
            p_x, p_y = grads_samp[slice_idxs,2,0], grads_samp[slice_idxs,2,1]
            u_xx, u_yy = hess_samp[slice_idxs,0,0,0], hess_samp[slice_idxs,0,1,1]
            v_xx, v_yy = hess_samp[slice_idxs,1,0,0], hess_samp[slice_idxs,1,1,1]
            return u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy


        def get_domain_residuals(slice_idxs, hess_samp, grads_samp, out_samp):
            u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy = get_correct_slice(
                slice_idxs, hess_samp, grads_samp, out_samp)
            residuals = {}
            match self.conf['hyperparams']["PINN_mode"]:
                case "supervised_only":
                    pass
                case "continuity_only":
                    residuals.update({"continuity": (u_x + v_y)})
                case "full_laminar":
                    residuals.update({"continuity": (u_x + v_y),
                                        "momentum_x": u*u_x + v*u_y + p_x - (u_xx + u_yy)/self.Re,
                                        "momentum_y": u*v_x + v*v_y + p_y - (v_xx + v_yy)/self.Re,})
                case _:
                    raise NotImplementedError(f"{self.conf['hyperparams']['PINN_mode']} is not implemented yet")
            
            return residuals

        def get_boundary_residuals(slice_idxs, hess_samp, grads_samp, out_samp):
            u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy = get_correct_slice(
                slice_idxs, hess_samp, grads_samp, out_samp)
            if self.conf['hyperparams']["flag_BC_PINN"]:
                residuals = vmap(self.get_BC_residuals_single_sample)(
                    x_BC, x_mask_BC, u, v, p, u_x, v_y, p_x, p_y
                )
                return {"BC_"+k: v for k,v in residuals.items()}
            else:
                return {}

        residuals = {}
        if sampling_points.shape[0] != 0:
            hess_samp, grads_samp, out_samp, out_sup = vmap(
                compute_second_derivative, out_dims=(0,0,0,None), randomness="different")(
                    sampling_points
            )
            
            if torch.isnan(hess_samp).sum() > 0:
                print(hess_samp)
            if torch.isnan(grads_samp).sum() > 0:
                print(grads_samp)
            if torch.isnan(out_samp).sum() > 0:
                print(out_samp)
            if torch.isnan(out_sup).sum() > 0:
                print(out_sup)

            residuals.update(get_domain_residuals(domain_slice, hess_samp, grads_samp, out_samp))
            
            boundary_residuals = get_boundary_residuals(boundary_slice, hess_samp, grads_samp, out_samp)
            
            if boundary_residuals.get("BC_total", None) is not None:
                residuals.update({"boundary": boundary_residuals["BC_total"]})

        else:
            out_sup = functional_call(
                self.net,
                model_params,
                (x, x_mask, edge_index, edge_attr, pos, batch),
            )

        if self.conf["hyperparams"]["domain_sampling"]["add_edges"] and (domain_slice.shape[0] != 0):
            residuals.update({"output_sampled_domain": out_samp[domain_slice]})

        return out_sup, residuals


    def loss(self, pred:torch.Tensor, label:torch.Tensor, data: Optional[pyg_data.Data]=None):

        out_supervised, residuals = pred
        assert isinstance(out_supervised, torch.Tensor)
        assert isinstance(residuals, dict)

        loss_dict = {}

        if (data is not None) and self.conf["hyperparams"]["domain_sampling"]["add_edges"]:
            
            output_sampled_domain = residuals.pop("output_sampled_domain")
            device = output_sampled_domain.device

            idxs =  data.new_edges_not_shifted[0,:]
            faces = data.new_edges_not_shifted[1,:]
            n = output_sampled_domain.shape[0]
            assert idxs.max()+1 == n, "Something wrong"

            x_vel = torch.zeros(n, device=device).scatter_reduce_(0, idxs, label[faces, 0], "mean")
            y_vel = torch.zeros(n, device=device).scatter_reduce_(0, idxs, label[faces, 1], "mean")
            press = torch.zeros(n, device=device).scatter_reduce_(0, idxs, label[faces, 2], "mean")

            loss_dict.update({"supervised_on_sampled": self.net.loss(output_sampled_domain, 
                                                                        torch.stack((x_vel, y_vel, press), dim=1))})
        else:
            loss_dict.update({"supervised": self.net.loss(out_supervised, label)})

        
        loss_dict.update({k: v.abs().mean() for k,v in residuals.items()})
        
        return sum(loss_dict.values()), loss_dict



def plot_continuity(pos: torch.Tensor, res: torch.Tensor):
    points = np.stack([
        pos[:,0].detach().cpu().numpy(), 
        pos[:,1].detach().cpu().numpy(), 
        res["continuity"].detach().cpu().numpy()]).T
    
    range_x = points[:,0].max() - points[:,0].min()
    range_y = points[:,1].max() - points[:,1].min()
    range_z = points[:,2].max() - points[:,2].min()

    points[:,2] = points[:,2]/range_z*max(range_x, range_y)*0.1

    pointcloud = pyvista.PolyData(points)
    pointcloud.plot()