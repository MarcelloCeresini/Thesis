import torch
from torch_geometric.transforms import BaseTransform
import torch_geometric.data as pyg_data

from torch import vmap
import numpy as np
import wandb
from utils import normalize_label

class NormalizeLabels(BaseTransform):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf

    def forward(self, data: torch.Any) -> torch.Any:
        for i, (values, label) in enumerate(zip(data.y.T, self.conf.labels_to_keep_for_training)):
            data.y[:,i] = normalize_label(values, label, self.conf)
        return data


class RemoveRadialAttributes(BaseTransform):
    def __init__(self, conf) -> None:
        super().__init__()
        self.n_radial_attributes = conf["n_radial_attributes"]

    def forward(self, data: torch.Any) -> torch.Any:
        data.x = data.x[:,:-self.n_radial_attributes]
        return data

class RemoveTurbulentLabels(BaseTransform):
    def forward(self, data: torch.Any) -> torch.Any:
        data.y = data.y[:,:-2]
        return data

class SampleDomainPoints(BaseTransform):
    def __init__(self, conf, test=False) -> None:
        super().__init__()
        self.domain_sampling = conf["domain_sampling"]
        self.general_sampling = conf["general_sampling"]
        self.num_labels = conf["output_dim"]
        self.n_sampled_new_edges = conf["n_sampled_new_edges"]
        self.test=test

    def get_random_position_in_simplex(self, spatial_positions):
        '''Uniform sampling inside simplex --> each coordinate is the linear combination of
            vertex coordinates, with weights sampled from an exponential distribution and then normalized
            
            Since we cannot get different random inplace sampling with vmap, we exploit the inverse transform sampling
            https://en.wikipedia.org/wiki/Inverse_transform_sampling#Definition
            and sample x from a uniform to then transform it to an exponential by -ln(x)'''
        match self.test:
            case False:
                eps = torch.finfo(torch.float64).eps
                barycentric_coords = - torch.rand(3, 
                    device=spatial_positions.device, dtype=torch.float64).clamp(min=eps).log() 
                barycentric_coords /= sum(barycentric_coords)
                barycentric_coords = barycentric_coords.to(torch.float32)
            case True:
                barycentric_coords = torch.ones(3, dtype=torch.float32) / 3
        return vmap(torch.dot, in_dims=(1,None))(spatial_positions, barycentric_coords)


    def forward(self, data: pyg_data.Data) -> pyg_data.Data:
        idxs_isnt_BC = (data.x_mask[:,-1] == 0)
        match self.domain_sampling["mode"]:
            case "all_domain":
                idxs_domain_sampled = idxs_isnt_BC
            case "percentage_of_domain":
                p = self.domain_sampling["percentage"]
                num_samples = int(p * idxs_isnt_BC.count_nonzero())
                idxs_domain_sampled = \
                    torch.multinomial(idxs_isnt_BC.to(torch.float), 
                                        num_samples, 
                                        replacement = p>=1) # as np.random.choice
                domain_sampling_points = data.pos[idxs_domain_sampled]
            case "uniformly_cells":
                assert data.triangulated_cells is not None, "Cannot sample from cells if triangulated_cells is not provided"
                p = self.domain_sampling["percentage"]
                num_samples = int(p * data.triangulated_cells.shape[0])

                if not self.test:
                    if self.general_sampling.get("use_sampling_weights", False): 
                        # all of the sampling points then train on all labels, but the placement of the points is chosen
                        # with an equally distributed number of samples for each label 
                        num_samples_per_label = num_samples // self.num_labels
                        num_samples = num_samples_per_label * self.num_labels

                        # TODO: rescale the weights?

                        idxs_domain_sampled_triangs = vmap(
                            lambda x: torch.multinomial(x[data.idx_of_triangulated_cell], 
                                                            num_samples_per_label, 
                                                            replacement = (p/self.num_labels)>=1),
                            in_dims=(1), randomness="different")(data.sampling_weights).view(-1)
                    else:
                        idxs_domain_sampled_triangs = \
                            torch.multinomial(torch.ones(data.triangulated_cells.shape[0], dtype=torch.float), 
                                                num_samples, 
                                                replacement = p>=1) # as np.random.choice
                else:
                    idxs_domain_sampled_triangs = torch.arange(data.triangulated_cells.shape[0])

                counter_resampling = 0
                while counter_resampling < 100:
                    domain_sampling_points = vmap(self.get_random_position_in_simplex, randomness="different") \
                        (data.triangulated_cells[idxs_domain_sampled_triangs,...].to(torch.float32))
                
                    if domain_sampling_points.isnan().sum() == 0:
                        break
                    else:
                        counter_resampling += 1
                        print(f"NANS in domain sampling, resampling. Counter at {counter_resampling}")
                    
                if domain_sampling_points.isnan().sum() > 0:
                    raise ValueError("NANS in domain sampling") 
                
                if self.general_sampling["add_edges"]:
                    ### FOR TRAINING
                    idxs_domain_sampled_cells = data.idx_of_triangulated_cell[idxs_domain_sampled_triangs]

                    sampled_padded_faces = data.faces_in_cell[idxs_domain_sampled_cells, :]
                    mask = sampled_padded_faces != -1

                    k_sampled_faces_idxs = torch.multinomial(
                        mask.to(torch.float32), 
                        num_samples=self.n_sampled_new_edges, 
                        replacement=self.n_sampled_new_edges>3)

                    k_sampled_faces = torch.gather(sampled_padded_faces, dim=1, index=k_sampled_faces_idxs)

                    # import matplotlib.pyplot as plt
                    # for i in range(num_samples):
                    #     # a = data.pos[data.faces_in_cell[:, idxs_domain_sampled_cells[i]]]
                    #     a = data.pos[k_sampled_faces[i,:]]
                    #     b = domain_sampling_points[i]
                    #     plt.scatter(a[:,0], a[:,1], c="b")
                    #     plt.scatter(b[0], b[1], c="y")
                    #     plt.show()

                    data.new_edges_index = k_sampled_faces.T
                    # data.new_edge_attributes = vmap(lambda x, y: torch.cat((y-x, torch.linalg.norm(y-x, dim=1, keepdim=True)), dim=1))(
                    #     domain_sampling_points, data.pos[k_sampled_faces, :2]
                    # )

                    ### FOR LOSS
                    sampled_faces_len = data.len_faces[idxs_domain_sampled_cells]
                    idxs_tensor = torch.arange(
                        idxs_domain_sampled_triangs.shape[0]).repeat_interleave(sampled_faces_len, dim=0)

                    # only from existing faces to these new points, not vice-versa
                    # message passing only goes to new places for prediction
                    sampled_faces = sampled_padded_faces[mask]
                    data.new_edges_not_shifted = torch.stack((idxs_tensor, sampled_faces)).T
                    new_edges_weights_tmp = vmap(lambda x, y: 1/torch.linalg.norm((y-x).clamp(min=1e-7))**2)(
                        domain_sampling_points[idxs_tensor], data.pos[sampled_faces][:,:2]
                    )
                    data.num_new_edges_not_shifted = idxs_tensor.shape[0]

                    if (tmp := torch.isnan(new_edges_weights_tmp).sum()) > 0:
                        print(f"distance_weighted_label (in augmentation) - {tmp} NaNs")
                        print(f"max (in augmentation) - {new_edges_weights_tmp.max()}")

                    data.new_edges_weights = new_edges_weights_tmp

                    # import matplotlib.pyplot as plt
                    # for i in range(num_samples):
                    #     # a = data.pos[data.faces_in_cell[:, idxs_domain_sampled_cells[i]]]
                    #     mask = idxs_tensor == i
                    #     a = data.pos[sampled_faces[mask]]
                    #     b = domain_sampling_points[i]
                    #     plt.scatter(a[:,0], a[:,1], c="b")
                    #     plt.scatter(b[0], b[1], c="y")
                    #     plt.show()
            case _:
                raise NotImplementedError()
        data.domain_sampling_points = domain_sampling_points
        data.num_domain_sampling_points = domain_sampling_points.shape[0]
        return data


class SampleBoundaryPoints(BaseTransform):
    def __init__(self, conf, test=False) -> None:
        super().__init__()
        self.boundary_sampling = conf["boundary_sampling"]
        self.feat_dict = conf["graph_node_feature_dict"]
        self.test = test

    def get_random_position_on_face(self, x, pos):
        tangent_versor_x = x[self.feat_dict["tangent_versor_x"]]
        tangent_versor_y = x[self.feat_dict["tangent_versor_y"]]
        face_area = x[self.feat_dict["face_area"]]

        lb_x = pos[0] - (tangent_versor_x*face_area)/2
        ub_x = pos[0] + (tangent_versor_x*face_area)/2

        lb_y = pos[1] - (tangent_versor_y*face_area)/2
        ub_y = pos[1] + (tangent_versor_y*face_area)/2

        lam = torch.rand(1, device=x.device)

        return lam*lb_x + (1-lam)*ub_x, lam*lb_y + (1-lam)*ub_y


    def forward(self, data: pyg_data.Data) -> pyg_data.Data:
        idxs_is_BC = (data.x_mask[:,-1] == 1)
        if not self.test:
            match self.boundary_sampling["mode"]:
                case "all_boundary":
                    idxs_boundary_sampled = idxs_is_BC.nonzero().view(-1)
                case "percentage_of_boundary":
                    p = self.boundary_sampling["percentage"]
                    num_samples = int(p * idxs_is_BC.count_nonzero())
                    idxs_boundary_sampled = torch.multinomial(
                        idxs_is_BC.to(torch.float), num_samples, 
                        replacement = False if p<1 else True)
                case _:
                    raise NotImplementedError("Only 'all' is implemented for now")
        else:
            idxs_boundary_sampled = idxs_is_BC.nonzero().view(-1)

        # x_BC, x_mask_BC = data.x[idxs_boundary_sampled], data.x_mask[idxs_boundary_sampled]

        if self.boundary_sampling["shift_on_face"] and not self.test:
            x_BC = data.x[idxs_boundary_sampled]
            boundary_sampling_points = vmap(
                self.get_random_position_on_face, randomness="different")(
                x_BC, data.pos[idxs_boundary_sampled]
            )
            boundary_sampling_points = torch.concatenate(boundary_sampling_points, dim=1)
        else:
            boundary_sampling_points = data.pos[idxs_boundary_sampled,:2]
        
        data.boundary_sampling_points = boundary_sampling_points
        data.num_boundary_sampling_points = boundary_sampling_points.shape[0]
        data.index_boundary_sampled = idxs_boundary_sampled.view(1,-1)
        data.x_additional_boundary = data.x_additional[idxs_boundary_sampled]
        return data