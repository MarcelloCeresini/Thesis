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
        self.use_positional_features = self.model_structure["update_edges"]
        self.graph_sampling_p_for_interpolation = self.conf["hyperparams"]["graph_sampling_p_for_interpolation"]

        if self.update_edges:
            self.edges_channels = self.model_structure["edges_channels"]

        _, self.positional_encoder = get_obj_from_structure(
            in_channels=2, 
            str_d=model_structure["positional_encoder"],
            conf=conf,
            out_channels=model_structure["message_passer"]["out_channels"],
        )

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
            processed = forward_for_general_layer(self.message_passer, X)
            X.update(processed)
            X.update({"x_graph": pyg_nn.pool.global_mean_pool(X["x"], batch)}) # Graph encoding is updated even when not used
        
        decoded = forward_for_general_layer(self.decoder, X)
        
        if sampling_points is not None:

            if self.use_positional_features:
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
                # weights = 1.0 / torch.clamp(squared_spatial_distance + squared_latent_distance, min=1e-16)
                sampled_points_encoding = torch.einsum("ij,i->j", 
                    processed["x"][sampled_graph_pos,:], weights.to(torch.float32)) / weights.sum()

            else:
                sampled_points_encoding = self.positional_encoder(sampling_points)

            U_sampled = forward_for_general_layer(self.decoder, {"x": sampled_points_encoding})
            return U_sampled["x"], decoded["x"]
            # Decode
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

        t_x, t_y = x[feat_dict["tangent_versor_x"]], x[feat_dict["tangent_versor_y"]]
        n_x, n_y = -t_y, +t_x

        residual = torch.zeros_like(p_x, device=p_x.device)
        c = torch.zeros_like(p_x, device=p_x.device)
        for k in mask_dict:
            match k:
                case "v_t":
                    v_t_pred = (t_x*u + t_y*v)
                    residual += (v_t_pred - x[feat_dict["v_t"]]).abs()       * x_mask[mask_dict[k]]
                    c += x_mask[mask_dict[k]]
                case "v_n": # "inward" is not a problem with walls because v_n should always be 0 --> no matter the direction 
                    v_n_pred = (n_x*u + n_y*v)
                    residual += (v_n_pred - x[feat_dict["v_n"]]).abs()       * x_mask[mask_dict[k]]
                    c += x_mask[mask_dict[k]]
                case "p":
                    residual += (p - x[feat_dict["p"]]).abs()                * x_mask[mask_dict[k]]
                    c += x_mask[mask_dict[k]]
                case "dv_dn": # pressure-outlet = n_x*U_x + n_y*U_y = n_x*u_x + n_y*v_y
                    dv_dn_pred = (n_x*u_x + n_y*v_y)
                    residual += (dv_dn_pred - x[feat_dict["dv_dn"]]).abs()   * x_mask[mask_dict[k]]
                    c += x_mask[mask_dict[k]]
                    # should be torch.dot(normal, (u_x, v_y)), but pressure-outlet is vertical
                    # residual += u_x.square()
                case "dp_dn": # 
                    dp_dn_pred = (n_x*p_x + n_y*p_y)
                    residual += (dp_dn_pred - x[feat_dict["dp_dn"]]).abs()   * x_mask[mask_dict[k]]
                    c += x_mask[mask_dict[k]]
        
        return residual / c


    def get_random_position_in_cell(self, cell_idx, CcFc_edges, pos):
        
        # FIXME: parallelize this and don't pull down on cpu --> 
        # create the full triangulation inside the data obj directly
        # matrix [n_triangles, 3(vertices), 2(x,y)]
        # sample triangle --> sample point inside triangle

        spatial_positions = pos[CcFc_edges[(CcFc_edges[:,0]==cell_idx),1], :]

        if spatial_positions.shape[0] > 3:
            tri = Delaunay(spatial_positions, qhull_options="QJ")
            sampled_simplex = torch.multinomial(
                torch.ones(tri.simplices.shape[0], dtype=torch.float), 1) # as np.random.choice
            spatial_positions = tri.points[tri.simplices[sampled_simplex]]

        barycentric_coords = np.random.exponential(scale=1.0, size=3)
        barycentric_coords /= sum(barycentric_coords)

        cartesian_coords = [np.dot(spatial_positions[:,0], barycentric_coords),
                                np.dot(spatial_positions[:,1], barycentric_coords)]
        
        return cartesian_coords


    def get_random_position_on_face(self, x, pos):
        tangent_versor_x = x[self.conf["hyperparams"]["feat_dict"]["tangent_versor_x"]]
        tangent_versor_y = x[self.conf["hyperparams"]["feat_dict"]["tangent_versor_y"]]
        face_area = x[self.conf["hyperparams"]["feat_dict"]["face_area"]]

        lb_x = pos[0] - (tangent_versor_x*face_area)/2
        ub_x = pos[0] + (tangent_versor_x*face_area)/2

        lb_y = pos[1] - (tangent_versor_y*face_area)/2
        ub_y = pos[1] + (tangent_versor_y*face_area)/2

        lam = torch.rand(1, device=x.device)

        return lam*lb_x + (1-lam)*ub_x, lam*lb_y + (1-lam)*ub_y


    def forward(self,
            x: torch.Tensor,
            x_mask: torch.Tensor, 
            edge_index: torch.Tensor, 
            edge_attr: torch.Tensor,
            pos: torch.Tensor,
            batch: torch.Tensor,
            CcFc_edges: Optional[torch.Tensor] = None,
            # triangulation: Delaunay, # TODO: if sampling = "all" precompute triangulation for each sample to ease computation 
            **kwargs
        ):
        
        pos = pos[:,:2]
        edge_attr = edge_attr[:, torch.tensor([0,1,3])]

        idxs_isnt_BC = (x_mask[:,-1] == 0)
        match self.domain_sampling["mode"]:
            case "all_domain":
                idxs_domain_sampled = idxs_isnt_BC
            case "percentage_of_domain":
                num_samples = int(self.domain_sampling["percentage"] * idxs_isnt_BC.count_nonzero())
                idxs_domain_sampled = \
                    torch.multinomial(idxs_isnt_BC.to(torch.float), num_samples, replacement=False) # as np.random.choice
                domain_sampling_points = pos[idxs_domain_sampled]
            case "uniformly_cells":
                assert CcFc_edges is not None, "Cannot sample from cells if CcFc_edges is not provided"
                num_samples = int(self.domain_sampling["percentage"] * CcFc_edges.shape[0])
                idxs_domain_sampled_cells = \
                    torch.multinomial(torch.ones(num_samples, dtype=torch.float), num_samples, replacement=False) # as np.random.choice

                cpu_pos = pos.cpu()
                domain_sampling_points = np.stack([self.get_random_position_in_cell(cell_idx, CcFc_edges, cpu_pos) 
                        for cell_idx in idxs_domain_sampled_cells])
                
                domain_sampling_points = torch.tensor(domain_sampling_points, device=x.device)

            case _:
                raise NotImplementedError()
        
        domain_sampling_points = torch.autograd.Variable(domain_sampling_points, requires_grad=True)
        
        idxs_is_BC = (x_mask[:,-1] == 1)
        match self.boundary_sampling["mode"]:
            case "all_boundary":
                idxs_boundary_sampled = idxs_is_BC
            case "percentage_of_boundary":
                num_samples = int(self.boundary_sampling["percentage"] * idxs_is_BC.count_nonzero())
                idxs_boundary_sampled = torch.multinomial(idxs_is_BC.to(torch.float), num_samples, replacement=False) # as np.random.choice
            case _:
                raise NotImplementedError("Only 'all' is implemented for now")
        
        x_BC, x_mask_BC = x[idxs_boundary_sampled], x_mask[idxs_boundary_sampled]

        if self.boundary_sampling["shift_on_face"]:
            boundary_sampling_points = vmap(
                self.get_random_position_on_face, randomness="different")(
                x_BC, pos[idxs_boundary_sampled]
            )
            boundary_sampling_points = torch.concatenate(boundary_sampling_points, dim=1)
        else:
            boundary_sampling_points = pos[idxs_boundary_sampled]

        boundary_sampling_points = torch.autograd.Variable(boundary_sampling_points, requires_grad=True)
        # TODO: do i need to add as variable also x and x_mask? because they are used to compute residuals
        
        model_params = ({k: v for k, v in self.net.named_parameters()}, 
                        {k: v for k, v in self.net.named_buffers()},)

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
        
        hess_samp, grads_samp, out_samp, out_sup = vmap(
            compute_second_derivative, out_dims=(0,0,0,None), randomness="different")(
                torch.concatenate((domain_sampling_points, boundary_sampling_points))
        )

        domain_slice = torch.arange(domain_sampling_points.shape[0])
        boundary_slice = torch.arange(start=domain_sampling_points.shape[0], 
                                        end=domain_sampling_points.shape[0]+boundary_sampling_points.shape[0])

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
                return {"boundary": vmap(self.get_BC_residuals_single_sample)(
                    x_BC, x_mask_BC, u, v, p, u_x, v_y, p_x, p_y
                )}
            else:
                return {}

        residuals = {}
        residuals.update(get_domain_residuals(domain_slice, hess_samp, grads_samp, out_samp))
        residuals.update(get_boundary_residuals(boundary_slice, hess_samp, grads_samp, out_samp))

        return out_sup, residuals


    def loss(self, pred:torch.Tensor, label:torch.Tensor):

        out_supervised, residuals = pred
        assert isinstance(out_supervised, torch.Tensor)
        assert isinstance(residuals, dict)

        loss_dict = {}
        loss_dict.update({"supervised": self.net.loss(out_supervised, label)})
        loss_dict.update({k: v.abs().mean() for k,v in residuals.items()})
        
        return sum(loss_dict.values()), loss_dict
