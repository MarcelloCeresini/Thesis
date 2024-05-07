import copy
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

from torch.nn import Linear, ReLU, Softplus, Tanhshrink
from torch_geometric.nn import GCNConv
# from torch_scatter import scatter_mean
from torch.func import functional_call, vmap, jacrev, hessian

from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import pyvista

from config_pckg.config_file import Config 
from loss_pckg.losses import MSE_loss
from .model_utils import get_obj_from_structure, forward_for_general_layer
from .layers import Simple_MLPConv
from utils import denormalize_label, get_coefficients, normalize_label

def plot_PYVISTA(pos: torch.Tensor, value: torch.Tensor, pos2: Optional[torch.Tensor]=None, value2: Optional[torch.Tensor]=None, rescale_z=False, point_size=3):
    points = np.stack([
        pos[:,0].detach().cpu().numpy(), 
        pos[:,1].detach().cpu().numpy(), 
        value.detach().cpu().numpy()]).T
    
    if rescale_z:
        range_x = points[:,0].max() - points[:,0].min()
        range_y = points[:,1].max() - points[:,1].min()
        range_z = points[:,2].max() - points[:,2].min()

        points[:,2] = points[:,2]/range_z*max(range_x, range_y)

    if pos2 is None:
        scalars = points[:,2]
    else:
        scalars = np.zeros(points.shape[0])
    pl = pyvista.Plotter()
    pl.add_points(
        points,
        scalars=scalars,
        style='points',
        point_size=point_size,)
    
    if pos2 is not None:
        points2 = np.stack([
            pos2[:,0].detach().cpu().numpy(), 
            pos2[:,1].detach().cpu().numpy(), 
            value2.detach().cpu().numpy()]).T
        # same normalization
        points2[:,2] = points2[:,2]
        pl.add_points(
            points2,
            scalars = np.ones(points2.shape[0]),
            style='points',
            point_size=point_size,)

    return pl


def get_model_instance(conf):
    
    net = eval(conf["model_structure"]["name"])(
        input_dim = conf["input_dim"],
        output_dim = conf["output_dim"],
        model_structure = conf["model_structure"],
        conf = conf
    )

    if conf["model_structure"]["PINN"] == True:
        return PINN(net=net, conf=conf)
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
                in_channels=conf["edge_feature_dim"], 
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
        return eval(self.conf["hyper_params"]["loss"])(pred, label)


class EPD_with_sampling(nn.Module):
    
    def __init__(self, input_dim, output_dim, model_structure: dict, conf, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO: improve definition from yaml: https://github.com/kaniblu/pytorch-models

        self.conf = conf
        self.model_structure = model_structure
        self.add_self_loops = self.model_structure["add_self_loops"]
        self.update_edges = self.model_structure.get("update_edges", False)
        
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
                in_channels=conf["edge_feature_dim"], 
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
        
        self.use_fourier_features = False
        match self.conf["inference_mode_latent_sampled_points"]:
            case "squared_distance":
                self.graph_sampling_p_for_interpolation = self.conf["graph_sampling_p_for_interpolation"]
            case "fourier_features":
                self.use_fourier_features = True
                self.fourier_pos_enc_matrix = torch.nn.Parameter(
                    torch.nn.init.normal_(
                        tensor=torch.zeros(2, self.conf["fourier_feat_dim"]), 
                            mean=0, std=1e-2
                    ))
                # self.fourier_pos_enc_matrix = torch.nn.Parameter(
                #         torch.zeros(2, self.conf["hyperparams"]["fourier_feat_dim"]), 
                #     )

                # input_dim += 2*self.conf["hyperparams"]["fourier_feat_dim"]
            case "baseline_positional_encoder":
                _, self.positional_encoder = get_obj_from_structure(
                    in_channels=2, 
                    str_d=model_structure["positional_encoder"],
                    conf=conf,
                    out_channels=model_structure["message_passer"]["out_channels"],
                )
            case "new_edges":
                current_dim, self.new_edges_msg_mlp = get_obj_from_structure(
                    in_channels=current_dim + self.conf["edge_feature_dim"], 
                    str_d=model_structure["new_edges_mlp"]["msg_mlp"], 
                    conf=conf,
                    out_channels=current_dim,
                )
                self.add_global_info_new_edges = model_structure["new_edges_mlp"].get("add_global_info", False)
                self.add_BC_info_new_edges = model_structure["new_edges_mlp"].get("add_BC_info", False)
                if self.add_global_info_new_edges or self.add_BC_info_new_edges:
                    self.new_edges_msg_activation = eval(model_structure["new_edges_mlp"]["standard_activation"])()
                    
                    current_dim, self.new_edges_update_mlp = get_obj_from_structure(
                        in_channels=current_dim*(1+self.add_global_info_new_edges+self.add_BC_info_new_edges), 
                        str_d=model_structure["new_edges_mlp"]["update_mlp"], 
                        conf=conf,
                        out_channels=current_dim,
                    )


        _, self.decoder = get_obj_from_structure(
            in_channels=current_dim, 
            str_d=model_structure["decoder"], 
            conf=conf,
            out_channels=output_dim,
        )

        if self.conf.get("bool_bootstrap_bias",False):
            init_bias = torch.zeros_like(self.decoder[-1].bias)
            for i, (k, value) in enumerate(self.conf["bootstrap_bias"].items()):
                init_bias[i] = normalize_label(value, k, 
                    self.conf.label_normalization_mode, self.conf.dict_labels_train, 
                    self.conf.air_speed, self.conf.Q)
            with torch.no_grad():
                self.decoder[-1].bias = nn.Parameter(init_bias.to(torch.float32))

        if self.conf.get("activation_for_max_normalized_features",False):
            if self.model_structure["decoder"].get("act_for_max_norm_feat", False) != False:
                self.turbolence_activation = eval(self.model_structure["decoder"]["act_for_max_norm_feat"])()

        self.loss_fn = eval(self.conf["hyper_params"]["loss"])


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
            new_edges: Optional[torch.Tensor] = None,
            new_edge_attributes: Optional[torch.Tensor] = None,
            batch_sampling_points: Optional[torch.Tensor] = None,
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
            match self.conf["inference_mode_latent_sampled_points"]:
                case "squared_distance":
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
                    
                case "new_edges":
                    # mask = new_edge_index[0,:] == sampling_points_idx
                    # distances = sampling_points-pos[new_edge_index[1,:]]
                    # dist = torch.where(mask.view(-1,1).repeat((1,2)), distances, torch.tensor((0,0)))
                    # final_dist = torch.cat((dist, torch.linalg.norm(dist, dim=1, keepdim=True)), dim=1)

                    # no update and no skip because there is no info in the sampled node
                    tmp_x = processed["x"][new_edges]
                    tmp_msg = self.new_edges_msg_mlp(
                        torch.concat((
                            tmp_x, 
                            new_edge_attributes), 
                        dim=1))
                    tmp_msg = tmp_x + tmp_msg
                    aggregated_msg = tmp_msg.mean(dim=0)
                    aggregated_msg = self.new_edges_msg_activation(aggregated_msg)

                    if self.add_global_info_new_edges:
                        correct_x_graph = X["x_graph"].index_select(0, batch_sampling_points.view(1)).view(-1)
                    else:
                        correct_x_graph = torch.FloatTensor(device=x.device)

                    if self.add_BC_info_new_edges:
                        correct_x_BC = X["x_BC"].index_select(0, batch_sampling_points.view(1)).view(-1)
                    else:
                        correct_x_BC = torch.FloatTensor(device=x.device)
                    
                    update = self.new_edges_update_mlp(
                        torch.concat((
                            aggregated_msg, 
                            correct_x_BC,
                            correct_x_graph,
                        )))
                    sampled_points_encoding = aggregated_msg + update

                case "baseline_positional_encoder":
                    raise NotImplementedError("need to debug")
                    sampled_points_encoding = self.positional_encoder(sampling_points)

                case _:
                    raise NotImplementedError("Not implemented")

            U_sampled = forward_for_general_layer(self.decoder, {"x": sampled_points_encoding.to(torch.float32)})

            if self.conf.get("activation_for_max_normalized_features",False):
                if self.model_structure["decoder"].get("act_for_max_norm_feat", False) != False:
                    
                    tmp = self.turbolence_activation(U_sampled["x"][self.conf["idx_to_apply_activation"]])
                    U_sampled["x"][self.conf["idx_to_apply_activation"]] = tmp

                    tmp = self.turbolence_activation(decoded["x"][:,self.conf["idx_to_apply_activation"]])
                    decoded["x"][:,self.conf["idx_to_apply_activation"]] = tmp
                
            return U_sampled["x"], decoded["x"]
        else:
            return decoded["x"]


    def loss(self, pred: torch.Tensor, label: torch.Tensor, ptr: Optional[torch.Tensor]=None):
        if ptr is None:
            return self.loss_fn(pred, label)
        else:
            return sum([self.loss_fn(pred[ptr[i]:ptr[i+1]], label[ptr[i]:ptr[i+1]]) 
                                for i in range(ptr.shape[0]-1)])


class PINN(nn.Module):
    
    def __init__(self, net:nn.Module, conf, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO: improve definition from yaml: https://github.com/kaniblu/pytorch-models

        self.net = net
        self.conf = conf

        self.domain_sampling = self.conf["domain_sampling"]
        self.boundary_sampling = self.conf["boundary_sampling"]

        # self.loss_weights = {"continuity":1, "momentum_x":1, "momentum_y":1}


    def get_BC_residuals_pointwise(self, x, x_mask, u, v, p, u_x, u_y, v_x, v_y, p_x, p_y):
        feat_dict = self.conf["graph_node_feature_dict"]
        mask_dict = self.conf["graph_node_feature_mask"]

        mask_dict_copy = mask_dict.copy()
        mask_dict_copy.pop("is_BC")

        t_x, t_y = x[feat_dict["tangent_versor_x"]], x[feat_dict["tangent_versor_y"]]
        n_x_tmp, n_y_tmp = -t_y, +t_x

        # in this way v-inlet (only one with normal BC where value != 0) is correct (right-facing)
        n_y = torch.where(n_x_tmp>0, n_y_tmp, -n_y_tmp)
        n_x = torch.where(n_x_tmp>0, n_x_tmp, -n_x_tmp) # could also do abs

        residual = {k: torch.zeros_like(p_x, device=p_x.device) for k in mask_dict_copy}
        c = torch.zeros_like(p_x, device=p_x.device)
        for k in mask_dict_copy:
            match k:
                case "v_t":
                    v_t_pred = (t_x*u + t_y*v)
                    residual["v_t"] += (v_t_pred - x[feat_dict["v_t"]]).abs()       * x_mask[mask_dict[k]]
                    c += x_mask[mask_dict[k]]
                case "v_n": # if it's a wall, no problem because v_n=0 (for inlet, see above)
                    v_n_pred = (n_x*u + n_y*v)
                    residual["v_n"] += (v_n_pred - x[feat_dict["v_n"]]).abs()       * x_mask[mask_dict[k]]
                    c += x_mask[mask_dict[k]]
                case "p": # only zero for now
                    residual["p"] += (p - x[feat_dict["p"]]).abs()                  * x_mask[mask_dict[k]]
                    c += x_mask[mask_dict[k]]
                case "dv_dn": # pressure-outlet = n_x*U_x + n_y*U_y = n_x*u_x + n_y*v_y
                    # FIXME: is this derivative right?
                    dv_dn_pred = (n_x*u_x + n_y*u_y) + (n_x*v_x + n_y*v_y)
                    residual["dv_dn"] += (dv_dn_pred).abs() * x_mask[mask_dict[k]]
                    c += 2*x_mask[mask_dict[k]]
                    # should be torch.dot(normal, (u_x, v_y)), but pressure-outlet is vertical
                    # residual += u_x
                case "dp_dn": # 
                    dp_dn_pred = (n_x*p_x + n_y*p_y)
                    residual["dp_dn"] += (dp_dn_pred - x[feat_dict["dp_dn"]]).abs() * x_mask[mask_dict[k]]
                    c += x_mask[mask_dict[k]]
        
        residual.update({"total":sum(residual.values())/c})
        return residual


    # def get_stress_tensors(self, k, w, u_x, u_y, v_x, v_y, k_x, k_y, w_x, w_y, \
    #                                         u_xx, u_yx, u_yy, v_xx, v_xy, v_yy, as_solver=True):
    #     '''tij = k/w * (dui/dxj + duj/dxi) - (2/3)*k*delta_cronecker(i,j)'''
    #     if as_solver:
    #         raise NotImplementedError("Look if this clamp is ok")
    #         w_clamped = torch.clamp(w, min=self.conf.w_min_for_clamp)
    #         nu_t = k/w_clamped
    #         txx_x = 2*(nu_t*u_xx - k_x/3)
    #         txy_y = nu_t*(u_yy+v_xy)
    #         tyx_x = nu_t*(v_xx+u_yx)
    #         tyy_y = 2*(nu_t*v_yy - k_y/3)
    #     else:
    #         w_clamped = torch.clamp(w, min=self.conf.w_min_for_clamp)
    #         txx_x = 2*((k_x*u_x+k*u_xx)/w_clamped - k*u_x*w_x/w_clamped**2 - k_x/3)
    #         txy_y = (k_y*(u_y+v_x) + k*(u_yy+v_xy))/w_clamped - (u_y+v_x)*k*w_y/w_clamped**2
    #         tyx_x = (k_x*(v_x+u_y) + k*(v_xx+u_yx))/w_clamped - (v_x+u_y)*k*w_x/w_clamped**2
    #         tyy_y = 2*((k_y*v_y+k*v_yy)/w_clamped - k*v_y*w_y/w_clamped**2 - k_y/3)
    #     return txx_x, txy_y, tyx_x, tyy_y


    def forward(self,
            *,
            x: torch.Tensor,
            x_mask: torch.Tensor,
            x_additional: torch.Tensor, 
            edge_index: torch.Tensor, 
            edge_attr: torch.Tensor,
            pos: torch.Tensor,
            batch: torch.Tensor,
            num_domain_sampling_points: torch.Tensor,
            num_boundary_sampling_points: torch.Tensor,
            n_cells: torch.Tensor,
            faces_in_cell: torch.Tensor,
            triangulated_cells: Optional[torch.Tensor] = None,
            # triangulation: Delaunay, # TODO: if sampling = "all" precompute triangulation for each sample to ease computation 
            **kwargs
        ):

        pos = pos[:,:2]
        if edge_attr.shape[1] == 4:
            # edge_attr = edge_attr[:, torch.tensor([0,1,3])]
            edge_attr = edge_attr[:, :2] # no norm
        
        if kwargs.get("domain_sampling_points", None) is not None:
            domain_sampling_points = torch.autograd.Variable(kwargs["domain_sampling_points"], requires_grad=True)
            n_domain_points = domain_sampling_points.shape[0]
            
            new_edges = kwargs.get("new_edges_index", None).T
            # needed for graph average --> each point is summed with its graph avg latent vector inside the model
            batch_domain_nodes = torch.cat([torch.full((1, num_domain_sampling_points[i]), i).view(-1) 
                for i in torch.arange(num_domain_sampling_points.shape[0])])
        else:
            domain_sampling_points = torch.autograd.Variable(torch.FloatTensor().to(x.device)) # legacy constructor expects device type: cpu
            n_domain_points = 0
            batch_domain_nodes = torch.IntTensor()
            new_edges = torch.IntTensor().view(-1,1).repeat(1,3).to(x.device)

        if kwargs.get("boundary_sampling_points", None) is not None:
            idxs_boundary_sampled = kwargs["index_boundary_sampled"].view(-1)
            x_BC, x_mask_BC = x[idxs_boundary_sampled], x_mask[idxs_boundary_sampled]

            boundary_sampling_points = torch.autograd.Variable(kwargs["boundary_sampling_points"], requires_grad=True)
            n_boundary_points = boundary_sampling_points.shape[0]
            # needed for graph average --> each point is summed with its graph avg latent vector inside the model
            batch_boundary_nodes = torch.cat([torch.full((1, num_boundary_sampling_points[i]), i).view(-1) 
                for i in torch.arange(num_boundary_sampling_points.shape[0])])
            # additional_boundary = kwargs["x_additional_boundary"]
        else:
            boundary_sampling_points = torch.autograd.Variable(torch.FloatTensor().to(x.device)) # legacy constructor expects device type: cpu
            n_boundary_points = 0
            batch_boundary_nodes = torch.IntTensor()
            idxs_boundary_sampled = torch.IntTensor().to(x.device)

        sampling_points = torch.concatenate((domain_sampling_points, boundary_sampling_points))

        # import matplotlib.pyplot as plt
        # a = domain_sampling_points.detach().numpy()
        # b = boundary_sampling_points.detach().numpy()
        # plt.scatter(a[:,0], a[:,1], color="b")
        # plt.scatter(b[:,0], b[:,1], color="r")

        def compute_output(sampling_points, new_edges, batch_sampling_points):
            # if the distance goes to zero, the grad on the norm goes to NaN (because of the derivative of the sqrt)
            new_edge_attributes = (pos[new_edges, :2]-sampling_points)

            out_samp, out_sup = functional_call(
                self.net,
                model_params,
                (x, x_mask, edge_index, edge_attr, pos, batch, 
                    sampling_points, new_edges, new_edge_attributes, batch_sampling_points),
            )
            return out_samp, (out_samp, out_sup)


        def compute_first_derivative(sampling_points, new_edges, batch_sampling_points):
            '''
            https://pytorch.org/tutorials/intermediate/jacobians_hessians.html
            
            jacfwd and jacrev can be substituted for each other but they have different performance characteristics.
            As a general rule of thumb, if you’re computing the jacobian of an R^N -> R^M function, and there are 
            many more outputs than inputs (for example, M>N) then jacfwd is preferred, otherwise use jacrev.
            '''
            grads_samp, (out_samp, out_sup) = \
                jacrev(compute_output, has_aux=True, argnums=0)(sampling_points, new_edges, batch_sampling_points)
            return grads_samp, (grads_samp, out_samp, out_sup)


        def compute_second_derivative(sampling_points, new_edges=None, batch_sampling_points=None):
            hess_samp, (grads_samp, out_samp, out_sup) = \
                jacrev(compute_first_derivative, has_aux=True, argnums=0)(sampling_points, new_edges, batch_sampling_points)
            return hess_samp, grads_samp, out_samp, out_sup
        

        def get_correct_slice(slice_idxs, hess_samp, grads_samp, out_samp, turbolence=False):
            u, v, p = out_samp[slice_idxs,0], out_samp[slice_idxs,1], out_samp[slice_idxs,2]
            u_x, u_y = grads_samp[slice_idxs,0,0], grads_samp[slice_idxs,0,1]
            v_x, v_y = grads_samp[slice_idxs,1,0], grads_samp[slice_idxs,1,1]
            p_x, p_y = grads_samp[slice_idxs,2,0], grads_samp[slice_idxs,2,1]
            u_xx, u_yy = hess_samp[slice_idxs,0,0,0], hess_samp[slice_idxs,0,1,1]
            v_xx, v_yy = hess_samp[slice_idxs,1,0,0], hess_samp[slice_idxs,1,1,1]
            if not turbolence:
                return u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy
            else:
                k, w = out_samp[slice_idxs,3], out_samp[slice_idxs,4]
                k_x, k_y = grads_samp[slice_idxs,3,0], grads_samp[slice_idxs,3,1]
                w_x, w_y = grads_samp[slice_idxs,4,0], grads_samp[slice_idxs,4,1]
                u_xy, u_yx = hess_samp[slice_idxs,0,1,0], hess_samp[slice_idxs,0,0,1]
                v_xy, v_yx = hess_samp[slice_idxs,1,1,0], hess_samp[slice_idxs,1,0,1]
                return u, v, p, k, w, \
                        u_x, u_y, v_x, v_y, p_x, p_y, k_x, k_y, w_x, w_y, \
                        u_xx, u_xy, u_yx, u_yy, v_xx, v_xy, v_yx, v_yy


        def get_domain_residuals(slice_idxs, hess_samp, grads_samp, out_samp):
            if not "turbulent_kw" in self.conf["PINN_mode"]:
                u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy = get_correct_slice(
                    slice_idxs, hess_samp, grads_samp, out_samp, turbolence=False)
            else:
                u, v, p, k, w, \
                u_x, u_y, v_x, v_y, p_x, p_y, k_x, k_y, w_x, w_y, \
                u_xx, u_xy, u_yx, u_yy, v_xx, v_xy, v_yx, v_yy = get_correct_slice(
                    slice_idxs, hess_samp, grads_samp, out_samp, turbolence=True)
                
            residuals = {}
            match self.conf["PINN_mode"]:
                case "supervised_only" | "supervised_with_sampling":
                    physical_quantities = (u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy)
                case "continuity_only":
                    u_x = denormalize_label(u_x, "x-velocity", self.conf)
                    v_y = denormalize_label(v_y, "y-velocity", self.conf)
                    residuals.update({"continuity": (u_x + v_y)*self.conf.air_density})
                    physical_quantities = (u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy)
                case "full_laminar":
                    u, u_x, u_y, u_xx, u_yy = vmap(denormalize_label, in_dims=(0,None,None))(torch.stack((u, u_x, u_y, u_xx, u_yy)), "x-velocity", self.conf)
                    v, v_x, v_y, v_xx, v_yy = vmap(denormalize_label, in_dims=(0,None,None))(torch.stack((v, v_x, v_y, v_xx, v_yy)), "y-velocity", self.conf)
                    p_x, p_y = vmap(denormalize_label, in_dims=(0,None,None))(torch.stack((p_x, p_y)), "pressure", self.conf)
                    residuals.update({"continuity": (u_x + v_y)*self.conf.air_density,
                                        "momentum_x": u*u_x + v*u_y + p_x/self.conf.air_density - (u_xx + u_yy)*self.conf.air_kinematic_viscosity,
                                        "momentum_y": u*v_x + v*v_y + p_y/self.conf.air_density - (v_xx + v_yy)*self.conf.air_kinematic_viscosity,})
                    physical_quantities = (u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy)
                case "turbulent_kw":
                    assert self.conf["output_turbulence"], "Cannot use turbolent equations if you don't output turbulence"
                    
                    # OSS: before denormalization, otherwise the values are crazy
                    if self.conf.get("physical_constraint_loss", False) and self.conf.PINN_mode == "turbulent_kw":
                        residuals["negative_k"] = torch.clamp_max(k, 0.)
                        residuals["negative_w"] = torch.clamp_max(w, 0.)
                    
                    u, u_x, u_y, u_xx, u_yx, u_yy = vmap(denormalize_label, in_dims=(0,None,None))(torch.stack((u, u_x, u_y, u_xx, u_yx, u_yy)), "x-velocity", self.conf)
                    v, v_x, v_y, v_xx, v_xy, v_yy = vmap(denormalize_label, in_dims=(0,None,None))(torch.stack((v, v_x, v_y, v_xx, v_xy, v_yy)), "y-velocity", self.conf)
                    p_x, p_y = vmap(denormalize_label, in_dims=(0,None,None))(torch.stack((p_x, p_y)), "pressure", self.conf)
                    k, k_x, k_y = vmap(denormalize_label, in_dims=(0,None,None))(torch.stack((k, k_x, k_y)), "turb-kinetic-energy", self.conf)
                    w, w_x, w_y = vmap(denormalize_label, in_dims=(0,None,None))(torch.stack((w, w_x, w_y)), "turb-diss-rate", self.conf)

                    k = torch.clamp_min(k, 0.)
                    # reference for the clipping https://doi.org/10.2514/1.36541
                    tmp = self.conf.C_lim_w*torch.sqrt((2*u_x**2 + (u_y+v_x)**2 + 2*v_y**2)/self.conf.beta_star_w)
                    w = torch.maximum(w, tmp)
                    
                    residuals.update({
                        "continuity": (u_x + v_y)*self.conf.air_density,
                        "momentum_x": u*u_x + v*u_y + p_x/self.conf.air_density 
                                        - (self.conf.air_kinematic_viscosity + k/w)*(u_xx + u_yy) 
                                        - ((k_x*w - k*w_x) * u_x + (k_y*w - k*w_y) * u_y) / w**2,
                        "momentum_y": u*v_x + v*v_y + p_y/self.conf.air_density
                                        - (self.conf.air_kinematic_viscosity + k/w)*(v_xx + v_yy)
                                        - ((k_x*w - k*w_x) * v_x + (k_y*w - k*w_y) * v_y) / w**2,
                                    })
                    physical_quantities = (u, v, p, k, w, \
                                            u_x, u_y, v_x, v_y, p_x, p_y, k_x, k_y, w_x, w_y, \
                                            u_xx, u_xy, u_yx, u_yy, v_xx, v_xy, v_yx, v_yy)
                case _:
                    raise NotImplementedError(f"{self.conf['PINN_mode']} is not implemented yet")
            
            return residuals, physical_quantities


        def get_boundary_residuals(slice_idxs, hess_samp, grads_samp, out_samp):
            u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy = get_correct_slice(
                slice_idxs, hess_samp, grads_samp, out_samp)
            # TODO: denormalize here instead than inside single_sample
            if self.conf["flag_BC_PINN"]:
                residuals = vmap(self.get_BC_residuals_pointwise)(
                    x_BC, x_mask_BC, u, v, p, u_x, u_y, v_x, v_y, p_x, p_y
                )
                return {"BC_"+k: v for k,v in residuals.items()}, (u_x, u_y, v_x, v_y)
            else:
                return {}, ()

        residuals = {}
        model_params = ({k: v for k, v in self.net.named_parameters()}, 
                        {k: v for k, v in self.net.named_buffers()},)

        if sampling_points.shape[0] != 0:
            
            batch_sampling_points = torch.concatenate((batch_domain_nodes, batch_boundary_nodes)).to(x.device)

            new_edges = torch.cat((
                new_edges,
                idxs_boundary_sampled.view(-1,1).repeat(1,3)))

            domain_slice = torch.arange(n_domain_points)
            boundary_slice = torch.arange(start=n_domain_points, end=n_domain_points+n_boundary_points)
            
            ####################################################################################
            hess_samp, grads_samp, out_samp, out_sup = vmap(
                compute_second_derivative, out_dims=(0,0,0,None), randomness="different")(
                    sampling_points, new_edges, batch_sampling_points
            )
            #####################################################################################
            
            # if (tmp := torch.isnan(hess_samp).sum()) > 0:
            #     print(f"hess_samp - {tmp} NaNs")
            if (tmp := torch.isnan(grads_samp).sum()) > 0:
                print(f"grads_samp - {tmp} NaNs")
            if (tmp := torch.isnan(out_samp).sum()) > 0:
                print(f"out_samp - {tmp} NaNs")
            if (tmp := torch.isnan(out_sup).sum()) > 0:
                print(f"out_sup - {tmp} NaNs")

            # for both domain and boundary residuals, batch is not important because it's a node-wise calculation
            domain_res, domain_quantities = get_domain_residuals(domain_slice, hess_samp, grads_samp, out_samp)
            residuals.update(domain_res)
            
            boundary_residuals, velocity_derivatives_at_B = get_boundary_residuals(boundary_slice, hess_samp, grads_samp, out_samp)
            
            if boundary_residuals.get("BC_total", None) is not None:
                residuals.update({"boundary": boundary_residuals.pop("BC_total")})
                residuals.update({"debug_only_"+k: v for k, v in boundary_residuals.items()})
            
            if self.conf["general_sampling"]["add_edges"] and (domain_slice.shape[0] != 0):
                residuals.update({"output_sampled_domain": out_samp[domain_slice]})

        else:
            out_sup = functional_call(
                self.net,
                model_params,
                (x, x_mask, edge_index, edge_attr, pos, batch),
            )
            velocity_derivatives_at_B = ()
            domain_quantities = ()

        if self.conf.get("bool_algebraic_continuity", False):
            ptr_cells = torch.tensor([n_cells[:i].sum() for i in range(n_cells.shape[0]+1)])
            faces_in_cell = torch.clone(faces_in_cell)
            for i in range(n_cells.shape[0]):
                faces_in_cell_slice = faces_in_cell[ptr_cells[i]:ptr_cells[i+1]]
                mask = faces_in_cell_slice != -1
                faces_in_cell[ptr_cells[i]:ptr_cells[i+1],:][mask] += ptr_cells[i]

            algebraic_continuity = vmap(self.compute_algebraic_continuity_for_one_cell, in_dims=(0,None,None,None,None,None))(
                faces_in_cell,                           # faces_in_cell_idxs
                pos[:,:2],                               # all_face_center_positions
                x[:,:2],                                 # all_tangents
                x[:,2],                                  # all_areas
                out_sup[:,0],    # all_u
                out_sup[:,1])    # all_v

            residuals.update({"algebraic_continuity":algebraic_continuity}) # one value (positive/negative) per cell
        
        return out_sup, residuals, velocity_derivatives_at_B, domain_quantities


    def compute_algebraic_continuity_for_one_cell(self,
            faces_in_cell_idxs, 
            all_face_center_positions, all_tangents, all_areas, all_u, all_v):

        faces_in_cell_idxs_repeated = faces_in_cell_idxs.repeat(2,1).T
        face_center_positions = torch.where(faces_in_cell_idxs_repeated == -1, torch.zeros_like(faces_in_cell_idxs_repeated),
            all_face_center_positions[faces_in_cell_idxs])

        n_faces_in_cell = (faces_in_cell_idxs != -1).sum()
        cell_center_position = torch.sum(face_center_positions, dim=0) / n_faces_in_cell

        CcFc_vectors = torch.where(faces_in_cell_idxs_repeated == -1, torch.ones_like(faces_in_cell_idxs_repeated),
            all_face_center_positions[faces_in_cell_idxs] - cell_center_position)
        
        tangents = torch.where(faces_in_cell_idxs_repeated == -1, torch.ones_like(faces_in_cell_idxs_repeated),
            all_tangents[faces_in_cell_idxs])
        
        normals = torch.stack(
            (-tangents[:,1], tangents[:,0])
        ).T

        areas = torch.where(faces_in_cell_idxs == -1, torch.zeros_like(faces_in_cell_idxs),
            all_areas[faces_in_cell_idxs])

        u = torch.where(faces_in_cell_idxs == -1, torch.zeros_like(faces_in_cell_idxs),
            all_u[faces_in_cell_idxs])
        
        v = torch.where(faces_in_cell_idxs == -1, torch.zeros_like(faces_in_cell_idxs),
            all_v[faces_in_cell_idxs])
        
        outward_normals = torch.where((vmap(torch.dot, in_dims=(0,0))(CcFc_vectors, normals) > 0).repeat(2,1).T,
            normals, -normals)
        
        algebraic_continuity_x = torch.sum(areas*outward_normals[:,0]*u)
        algebraic_continuity_y = torch.sum(areas*outward_normals[:,1]*v)

        return self.conf.air_density*(algebraic_continuity_x+algebraic_continuity_y)


    def loss(self, pred:torch.Tensor, label:torch.Tensor, batch: pyg_data.Data|pyg_data.Batch):

        out_supervised, residuals = pred[0], pred[1]
        domain_quantities = pred[3]
        batch_size = batch.batch_size
        assert isinstance(out_supervised, torch.Tensor)
        assert isinstance(residuals, dict)
        assert out_supervised.shape[0] == label.shape[0], f"Dimensions do not match: out_supervised.shape[0] = {out_supervised.shape[0]} \
            != label.shape[0] = {label.shape[0]}"

        loss_dict = {}

        output_sampled_domain = residuals.pop("output_sampled_domain", None)
        if output_sampled_domain is not None:
            device = output_sampled_domain.device

            idxs =  batch.new_edges_not_shifted.T[0,:] # idxs of sampling points (for each batch between 0 and data.num_domain_sampling_points)
            new_idxs = torch.clone(idxs)
            num_sampled = batch.num_domain_sampling_points
            ptr_num_sampled = torch.tensor([num_sampled[:i].sum() for i in range(num_sampled.shape[0]+1)])
            
            faces = batch.new_edges_not_shifted.T[1,:] # idxs of faces to which sampled points are connected to (for each batch between 0 and data.ptr[i])
            new_faces = torch.clone(faces)
            num_new_edges_not_shifted = batch.num_new_edges_not_shifted
            ptr_new_edges_not_shifted = torch.tensor([num_new_edges_not_shifted[:i].sum() for i in range(num_new_edges_not_shifted.shape[0]+1)])
            
            for i in range(batch_size):
                new_idxs[ptr_new_edges_not_shifted[i]:ptr_new_edges_not_shifted[i+1]] = \
                    idxs[ptr_new_edges_not_shifted[i]:ptr_new_edges_not_shifted[i+1]] + ptr_num_sampled[i]
                new_faces[ptr_new_edges_not_shifted[i]:ptr_new_edges_not_shifted[i+1]] = \
                    faces[ptr_new_edges_not_shifted[i]:ptr_new_edges_not_shifted[i+1]] +batch.ptr[i]

            n = output_sampled_domain.shape[0]
            assert new_idxs.max()+1 == n, "Something wrong"

            # new_edges_weights is 1/norm_of(face_position - sampled_point_position)
            distance_weighted_label = label[new_faces]*batch.new_edges_weights.view(-1,1)

            if (tmp := torch.isnan(distance_weighted_label).sum()) > 0:
                print(f"distance_weighted_label (in loss) - {tmp} NaNs")

            x_vel = torch.ones(n, device=device, 
                dtype=distance_weighted_label.dtype).scatter_reduce_(0, new_idxs, distance_weighted_label[:,0], "sum", include_self=False)
            y_vel = torch.ones(n, device=device, 
                dtype=distance_weighted_label.dtype).scatter_reduce_(0, new_idxs, distance_weighted_label[:,1], "sum", include_self=False)
            press = torch.ones(n, device=device, 
                dtype=distance_weighted_label.dtype).scatter_reduce_(0, new_idxs, distance_weighted_label[:,2], "sum", include_self=False)

            if output_sampled_domain.shape[1] > 3:
                k_turb = torch.ones(n, device=device,
                    dtype=distance_weighted_label.dtype).scatter_reduce_(0, new_idxs, distance_weighted_label[:,3], "sum", include_self=False)
                w_turb = torch.ones(n, device=device, 
                    dtype=distance_weighted_label.dtype).scatter_reduce_(0, new_idxs, distance_weighted_label[:,4], "sum", include_self=False)

            normalization_const = torch.zeros(n, device=device).scatter_reduce_(0, new_idxs, batch.new_edges_weights, "sum", include_self=False)

            if (tmp := torch.isnan(normalization_const).sum()) > 0:
                print(f"normalization_const (in loss) - {tmp} NaNs")

            if output_sampled_domain.shape[1] <= 3:
                gt_in_sampled = torch.stack((x_vel, y_vel, press), dim=1)/normalization_const.view(-1,1)
            else:
                gt_in_sampled = torch.stack((x_vel, y_vel, press, k_turb, w_turb), dim=1)/normalization_const.view(-1,1)

            # i=0   # 0,1,2,3,4
            # pl = plot_PYVISTA(data.domain_sampling_points, gt_in_sampled[:,i], data.pos[:,:2], data.y[:,i])
            # pl = plot_PYVISTA(data.pos[:,:2], data.y[:,0])
            # pl.set_scale(zscale=10)
            # pl.show()

            ### WITHOUT WEIGHTS (squared distance)
            # x_vel = torch.ones(n, device=device).scatter_reduce_(0, idxs, label[faces][:,0], "mean", include_self=False)
            # y_vel = torch.ones(n, device=device).scatter_reduce_(0, idxs, label[faces][:,1], "mean", include_self=False)
            # press = torch.ones(n, device=device).scatter_reduce_(0, idxs, label[faces][:,2], "mean", include_self=False)
            # gt_in_sampled = torch.stack((x_vel, y_vel, press), dim=1)
            loss_dict.update({"supervised_on_sampled": self.net.loss(output_sampled_domain, gt_in_sampled, ptr_num_sampled)})
        
        ### CAN CHOOSE TO ONLY SUPERVISE IN SAMPLED POINTS (but then you have no supervision on mesh points,
        ###     so it doesn't work very well)
        loss_dict.update({"supervised": self.net.loss(out_supervised, label, ptr=batch.ptr)})
        # else:
        #     loss_dict.update({"supervised": self.net.loss(out_supervised, label)})

        def get_values_per_sample(sample_residuals, additional_boundary=None):
            loss_fn = lambda x: x.abs().mean() if self.conf.residual_loss == "MAE" else x.square().mean()
            loss_fn_LOGGING = lambda x: x.abs().mean()
            sample_loss_dict = {}
            sample_optional_values = {}
            components = set(self.conf.graph_node_features_not_for_training).difference([
                                "component_id", "is_car", "is_flap", "is_tyre"])
            for k in sample_residuals:
                if "debug_only_" in k:
                    with torch.no_grad():
                        k_to_log = k.removeprefix("debug_only_")
                        tmp = loss_fn_LOGGING(sample_residuals[k][sample_residuals[k].nonzero()])
                        sample_optional_values[k_to_log] = tmp if not tmp.isnan() else torch.tensor(0.)
                        for comp in components:
                            correct_idxs = additional_boundary[:,self.conf.graph_node_features_not_for_training[comp]] == 1
                            tmp = loss_fn_LOGGING(sample_residuals[k][correct_idxs])
                            
                            sample_optional_values[comp+"_"+k_to_log] = tmp if not tmp.isnan() \
                                else torch.tensor(0., device=additional_boundary.device)

                            sample_optional_values[comp+"_total"] = sample_optional_values.get(comp+"_total",0.) + \
                                                                        sample_optional_values[comp+"_"+k_to_log]
                else:
                    sample_loss_dict[k] = loss_fn(residuals[k])
            return sample_loss_dict, sample_optional_values

        flag_boundary = residuals.get("boundary", None)
        if flag_boundary is not None:
            num_sampled_BC = batch.num_boundary_sampling_points
            ptr_num_sampled_BC = torch.tensor([num_sampled_BC[:i].sum() for i in range(num_sampled_BC.shape[0]+1)])

        if self.conf.get("bool_algebraic_continuity", False):
            ptr_cells = torch.tensor([batch.n_cells[:i].sum() for i in range(batch.n_cells.shape[0]+1)])

        optional_values = {}
        sample_residuals = {}
        x_additional_boundary = getattr(batch, "x_additional_boundary", None)
        for i in range(batch_size):
            for k in residuals:
                # [data.ptr[i]:data.ptr[i+1]]
                if k in ["continuity", "momentum_x", "momentum_y"]: # domain sampling
                    sample_residuals[k] = residuals[k][ptr_num_sampled[i]:ptr_num_sampled[i+1]]
                elif k == "algebraic_continuity":
                    sample_residuals[k] = residuals[k][ptr_cells[i]:ptr_cells[i+1]]
                else: # boundary sampling
                    sample_residuals[k] = residuals[k][ptr_num_sampled_BC[i]:ptr_num_sampled_BC[i+1]]
            
            if x_additional_boundary is not None:
                x_additional_boundary_sliced = x_additional_boundary[ptr_num_sampled_BC[i]:ptr_num_sampled_BC[i+1]]
            else:
                x_additional_boundary_sliced = None
            
            sample_loss_dict, sample_optional_values = get_values_per_sample(
                sample_residuals, x_additional_boundary_sliced)
            
            for k in sample_loss_dict:
                loss_dict[k] = loss_dict.get(k,0.) + sample_loss_dict[k]
            for k in sample_optional_values:
                optional_values[k] = optional_values.get(k,0.) + sample_optional_values[k]

        loss_dict = {k:v/batch_size for k,v in loss_dict.items()}
        optional_values = {k:v/batch_size for k,v in optional_values.items()}

        momentum_percentage = (loss_dict.get("momentum_x", 0.)+loss_dict.get("momentum_y", 0.)) / loss_dict["supervised"]
        if momentum_percentage > 1:
            with torch.no_grad():
                u, v, p, k, w, \
                    u_x, u_y, v_x, v_y, p_x, p_y, k_x, k_y, w_x, w_y, \
                    u_xx, u_xy, u_yx, u_yy, v_xx, v_xy, v_yx, v_yy = domain_quantities
                # print((u*u_x).abs().max())
                # print((v*u_y).abs().max())
                # print((p_x/self.conf.air_density).abs().max())
                # print(((self.conf.air_kinematic_viscosity + k/w)*(u_xx + u_yy)).abs().max())
                # print((((k_x*w - k*w_x) * u_x + (k_y*w - k*w_y) * u_y) / w**2).abs().max())

        if self.conf.get("normalize_denormalized_loss_components", False):
            supervised_value_lb = loss_dict["supervised"].item() * self.conf.get("minimum_continuity_relative_weight", 0)
            
            tmp = loss_dict.get("continuity", None)
            if tmp is not None and tmp != 0 and tmp < supervised_value_lb:
                    loss_dict[k] = (supervised_value_lb) * tmp / tmp.item()

            supervised_value_lb = loss_dict["supervised"].item() * self.conf.get("minimum_momentum_relative_weight", 0)
            supervised_value_ub = loss_dict["supervised"].item() * self.conf.get("maximum_momentum_relative_weight", 1e20)

            for k in ["momentum_x", "momentum_y"]:
                tmp = loss_dict.get(k, None)
                if tmp is not None and tmp != 0:
                    if tmp < supervised_value_lb:
                        loss_dict[k] = (supervised_value_lb) * tmp / tmp.item()
                    elif tmp > supervised_value_ub:
                        loss_dict[k] = (supervised_value_ub) * tmp / tmp.item()


        if self.conf.get("bool_aero_loss", False):
            aero_loss = {}
            for i in range(len(batch)):
                data = batch[i]
                pred_supervised_pts_pressure = pred[0][batch.ptr[i]:batch.ptr[i+1], 2]
                # assert batch.ptr.shape[0] == 2, "Check derivatives for batch size higher than 1"

                if self.conf.flag_BC_PINN and self.conf.output_turbulence:
                    ptr_num_sampled_boundary = torch.tensor([batch.num_boundary_sampling_points[:i].sum() 
                        for i in range(batch.num_boundary_sampling_points.shape[0]+1)])
                    pred_vel_derivatives = torch.stack(
                        [p[ptr_num_sampled_boundary[i]:ptr_num_sampled_boundary[i+1]] for p in pred[2]])
                    pred_turb_values = pred[0][batch.ptr[i]:batch.ptr[i+1], 3:]
                    pred_coefficients = get_coefficients(self.conf, data, pred_supervised_pts_pressure, 
                        velocity_derivatives=pred_vel_derivatives, turbulent_values=pred_turb_values, 
                        denormalize=True, from_boundary_sampling=True)
                    
                    aero_loss["main_shear"] = aero_loss.get("main_shear", 0) \
                        + self.net.loss(pred_coefficients["main_flap"][1], data.components_coefficients["main_flap"][1])
                else:
                    pred_coefficients = get_coefficients(self.conf, data, pred_supervised_pts_pressure, denormalize=True)
            
                aero_loss["main_pressure"] = aero_loss.get("main_pressure", 0) \
                    + self.net.loss(pred_coefficients["main_flap"][0], data.components_coefficients["main_flap"][0])
                
            loss_dict.update({f"aero_loss_{k}": aero_loss[k]/batch_size for k in aero_loss})
        

        # for k in residuals:
        #     if "debug_only_" in k:
        #         with torch.no_grad():
        #             k_to_log = k.removeprefix("debug_only_")
        #             tmp = residuals[k][residuals[k].nonzero()].abs().mean()
        #             optional_values[k_to_log] = tmp if not tmp.isnan() else torch.tensor(0.)
        #             tmp_dict = {}
        #             for comp in set(self.conf.graph_node_features_not_for_training).difference([
        #                     "component_id", "is_car", "is_flap", "is_tyre"]):
        #                 correct_idxs = data.x_additional_boundary[:,self.conf.graph_node_features_not_for_training[comp]] == 1
        #                 tmp = residuals[k][correct_idxs].abs().mean()
                        
        #                 optional_values[comp+"_"+k_to_log] = tmp if not tmp.isnan() else torch.tensor(0.)
        #                 tmp_dict[k_to_log] = optional_values[comp+"_"+k_to_log]
        #             optional_values[comp+"_total"] = sum(tmp_dict.values())
        #     else:
        #         loss_dict[k] = residuals[k].abs().mean()

        return sum(loss_dict.values()), loss_dict, optional_values


def plot_optional_values(optional_values):
    tmp = {k:v.detach().cpu().numpy() for k,v in optional_values.items()}
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame([tmp], index=["A"])
    df = df.sort_values(by="A", axis=1).astype(float)
    
    data = df.to_dict("records")[0]
    names = list(data.keys())
    values = list(data.values())

    #tick_label does the some work as plt.xticks()
    plt.bar(range(len(data)),values,tick_label=names)
    plt.xticks(rotation=90)
    plt.show()
