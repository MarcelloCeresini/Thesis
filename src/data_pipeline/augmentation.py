import torch
from torch_geometric.transforms import BaseTransform
import torch_geometric.data as pyg_data

from torch import vmap

class SampleDomainPoints(BaseTransform):
    def __init__(self, full_conf) -> None:
        super().__init__()
        self.full_conf = full_conf
        self.domain_sampling = self.full_conf["hyperparams"]["domain_sampling"]
        self.general_sampling = self.full_conf["hyperparams"]["general_sampling"]

    def get_random_position_in_simplex(self, spatial_positions):
        '''Uniform sampling inside simplex --> each coordinate is the linear combination of
            vertex coordinates, with weights sampled from an exponential distribution and then normalized
            
            Since we cannot get different random inplace sampling with vmap, we exploit the inverse transform sampling
            https://en.wikipedia.org/wiki/Inverse_transform_sampling#Definition
            and sample x from a uniform to then transform it to an exponential by -ln(x)'''
        eps = torch.finfo(torch.float64).eps
        barycentric_coords = - torch.rand(3, 
            device=spatial_positions.device, dtype=torch.float64).clamp(min=eps).log() 
        barycentric_coords /= sum(barycentric_coords)
        barycentric_coords = barycentric_coords.to(torch.float32)
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
                idxs_domain_sampled_triangs = \
                    torch.multinomial(torch.ones(data.triangulated_cells.shape[0], dtype=torch.float), 
                                        num_samples, 
                                        replacement = p>=1) # as np.random.choice

                domain_sampling_points = vmap(self.get_random_position_in_simplex, randomness="different") \
                    (data.triangulated_cells[idxs_domain_sampled_triangs,...].to(torch.float32))
                
                if self.general_sampling["add_edges"]:
                    ### FOR TRAINING
                    idxs_domain_sampled_cells = data.idx_of_triangulated_cell[idxs_domain_sampled_triangs]

                    sampled_padded_faces = data.faces_in_cell[:, idxs_domain_sampled_cells]
                    mask = sampled_padded_faces != -1

                    k_sampled_faces_idxs = torch.multinomial(
                        mask.to(torch.float32).T, 
                        num_samples=self.full_conf["hyperparams"]["n_sampled_new_edges"], 
                        replacement=self.full_conf["hyperparams"]["n_sampled_new_edges"]>3)

                    k_sampled_faces = torch.gather(sampled_padded_faces.T, dim=1, index=k_sampled_faces_idxs)

                    # import matplotlib.pyplot as plt
                    # for i in range(num_samples):
                    #     # a = data.pos[data.faces_in_cell[:, idxs_domain_sampled_cells[i]]]
                    #     a = data.pos[k_sampled_faces[i,:]]
                    #     b = domain_sampling_points[i]
                    #     plt.scatter(a[:,0], a[:,1], c="b")
                    #     plt.scatter(b[0], b[1], c="y")
                    #     plt.show()

                    data.new_edges = k_sampled_faces
                    # data.new_edge_attributes = vmap(lambda x, y: torch.cat((y-x, torch.linalg.norm(y-x, dim=1, keepdim=True)), dim=1))(
                    #     domain_sampling_points, data.pos[k_sampled_faces, :2]
                    # )

                    ### FOR LOSS
                    sampled_faces_len = data.len_faces[idxs_domain_sampled_cells]
                    idxs_tensor = torch.arange(
                        idxs_domain_sampled_triangs.shape[0]).repeat_interleave(sampled_faces_len, dim=0)

                    # only from existing faces to these new points, not vice-versa
                    # message passing only goes to new places for prediction
                    sampled_faces = sampled_padded_faces.T[mask.T]
                    data.new_edges_not_shifted = torch.stack((idxs_tensor, sampled_faces))
                    data.new_edges_weights = vmap(lambda x, y: 1/torch.linalg.norm(y-x)**2)(
                        domain_sampling_points[idxs_tensor], data.pos[sampled_faces][:,:2]
                    )

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
        return data


class SampleBoundaryPoints(BaseTransform):
    def __init__(self, full_conf) -> None:
        super().__init__()
        self.full_conf = full_conf
        self.boundary_sampling = self.full_conf["hyperparams"]["boundary_sampling"]

    def get_random_position_on_face(self, x, pos):
        tangent_versor_x = x[self.full_conf["hyperparams"]["feat_dict"]["tangent_versor_x"]]
        tangent_versor_y = x[self.full_conf["hyperparams"]["feat_dict"]["tangent_versor_y"]]
        face_area = x[self.full_conf["hyperparams"]["feat_dict"]["face_area"]]

        lb_x = pos[0] - (tangent_versor_x*face_area)/2
        ub_x = pos[0] + (tangent_versor_x*face_area)/2

        lb_y = pos[1] - (tangent_versor_y*face_area)/2
        ub_y = pos[1] + (tangent_versor_y*face_area)/2

        lam = torch.rand(1, device=x.device)

        return lam*lb_x + (1-lam)*ub_x, lam*lb_y + (1-lam)*ub_y


    def forward(self, data: pyg_data.Data) -> pyg_data.Data:
        idxs_is_BC = (data.x_mask[:,-1] == 1)
        match self.boundary_sampling["mode"]:
            case "all_boundary":
                idxs_boundary_sampled = idxs_is_BC
            case "percentage_of_boundary":
                p = self.boundary_sampling["percentage"]
                num_samples = int(p * idxs_is_BC.count_nonzero())
                idxs_boundary_sampled = torch.multinomial(
                    idxs_is_BC.to(torch.float), num_samples, 
                    replacement = False if p<1 else True)
            case _:
                raise NotImplementedError("Only 'all' is implemented for now")

        # x_BC, x_mask_BC = data.x[idxs_boundary_sampled], data.x_mask[idxs_boundary_sampled]
        x_BC = data.x[idxs_boundary_sampled]

        if self.boundary_sampling["shift_on_face"]:
            boundary_sampling_points = vmap(
                self.get_random_position_on_face, randomness="different")(
                x_BC, data.pos[idxs_boundary_sampled]
            )
            boundary_sampling_points = torch.concatenate(boundary_sampling_points, dim=1)
        else:
            boundary_sampling_points = data.pos[idxs_boundary_sampled]
        
        data.boundary_sampling_points = boundary_sampling_points
        data.idxs_boundary_sampled = idxs_boundary_sampled
        return data