import os
import pickle
from typing import Literal, Optional, Union
import itertools

from tqdm import tqdm
import wandb
import meshio
import pyvista
import toughio
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from icecream import ic
import torch
from torch_geometric.data import Data, HeteroData
import pandas as pd
from circle_fit import hyperLSQ
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
from torch_geometric.utils import to_dense_adj
from rustworkx import PyGraph
from rustworkx import distance_matrix
from torch import vmap

from config_pckg.config_file import Config
import read_mesh_meshio_forked
from mesh_exploration import plot_2d_cfd


def init_wandb(conf: Config, overwrite_WANDB_MODE: Optional[Literal["online", "offline"]] = None):
    if overwrite_WANDB_MODE is not None:
        conf.WANDB_MODE = overwrite_WANDB_MODE
    os.environ["WANDB_MODE"] = conf.WANDB_MODE
    wandb.init(
        project="Thesis",
        config=conf
    )


def read_mesh(filename, mode: Literal["meshio", "pyvista", "toughio"], conf: Config):
    '''Reads mesh given mode'''
    # mesh = meshio.read(filename)
    match mode:
        case "meshio":
            mesh = read_mesh_meshio_forked.read(filename)
            mesh.points *= conf.mesh_to_features_scale_factor
            return mesh
        case "pyvista":
            # Useful to plot the mesh but not useful for mesh manipulation
            # See: https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.UnstructuredGrid.html#pyvista.UnstructuredGrid
            mesh = pyvista.read(filename)
            mesh.points *= conf.mesh_to_features_scale_factor
            return mesh
        case "toughio":
            raise NotImplementedError("How to implement scaling points?")
            # Parent classes to meshio Mesh type with (it seems) useful utilities
            # https://toughio.readthedocs.io/en/latest/mesh.html
            return toughio.read_mesh(filename)
        case _:
            raise NotImplementedError('Only "meshio", "pyvista" and "toughio" are accepted values')


def sort_matrix(matrix):
    '''Sorts a matrix by columns'''
    if matrix[0].shape[0] == 2:
        idxs = np.lexsort((matrix[:,1], matrix[:,0]))
        return matrix[idxs], idxs
    elif matrix[0].shape[0] == 3:
        idxs = np.lexsort((matrix[:,2], matrix[:,1], matrix[:,0]))
        return matrix[idxs], idxs
    else:
        raise NotImplementedError


def match_mesh_and_feature_pts(mesh, features, conf: Config, dim=2, check_biunivocity=True):
    eps = conf.epsilon_for_point_matching

    map_mesh_to_feature = np.zeros([len(mesh.points), 2])
    if dim == 2:
        pts = mesh.points[:,:2]
        mesh_m, mesh_old_idxs = sort_matrix(pts)
        features_m, features_old_idxs = sort_matrix(features[["    x-coordinate", "    y-coordinate"]].to_numpy())
        
        mesh_abs = np.abs(mesh_m)
        features_low_bound = np.abs(features_m*(1-eps))
        features_high_bound = np.abs(features_m*(1+eps))

        for i, (mesh_old_idx, ft_old_idx, m_a, ft_lb, ft_hb) in enumerate(zip(mesh_old_idxs,
                                                                              features_old_idxs,
                                                                              mesh_abs,
                                                                              features_low_bound,
                                                                              features_high_bound)):
            if (ft_lb<=m_a).all() and (m_a<=ft_hb).all():
                map_mesh_to_feature[i, 0], map_mesh_to_feature[i, 1] = mesh_old_idx, ft_old_idx
            else:
                for j, (ft_lb_2, ft_hb_2) in enumerate(zip(features_low_bound, features_high_bound)):
                    if (ft_lb_2<=m_a).all() and (m_a<=ft_hb_2).all():
                        map_mesh_to_feature[i, 0], map_mesh_to_feature[i, 1] = mesh_old_idx, features_old_idxs[j]
                        break
                if j >= len(features_low_bound)-1:
                    ic(i, (ft_lb+ft_hb)/(2*m_a))
        if check_biunivocity:
            assert len(np.unique(map_mesh_to_feature[:,0])) == len(map_mesh_to_feature), "Not biunivocal, for some mesh point there is no correspondance in features"
            assert len(np.unique(map_mesh_to_feature[:,1])) == len(map_mesh_to_feature), "Not biunivocal, for some feature points there is no correspondant mesh point"
        map_mesh_to_feature, _ =  sort_matrix(map_mesh_to_feature)
        return map_mesh_to_feature[:,1].astype(np.int64)
    else:
        raise NotImplementedError("Not implemented for 3d")


def get_edges_from_face(idxs):
    '''Returns bidirectional edges between vertices of lines, triangles and quads'''
    if (sh := idxs.shape[0]) == 2: # line
        return np.stack([
            idxs, np.flip(idxs)
        ])
    elif sh == 3 or sh == 4: # traingle or quad
        idxs = np.concatenate([idxs, [idxs[0]]])
        pairs = np.lib.stride_tricks.sliding_window_view(idxs, 2)
        return np.concatenate([pairs, np.flip(pairs)])
    else:
        raise NotImplementedError("Get simpler connections before through thoughio.get_connections()")


def get_edges_from_component(cellblock):
    return np.concatenate(list(map(get_edges_from_face, cellblock.data)))


def get_all_edges(mesh):
    if isinstance(mesh, meshio.Mesh) or isinstance(mesh, toughio.Mesh):
        return np.concatenate(list(map(get_edges_from_component, mesh.cells)))
    elif isinstance(mesh, pyvista.UnstructuredGrid):
        raise NotImplementedError("Didn't implement cell connectivity for pyvista yet")


# def convert_msh_csv_to_graph(filename_input_msh, filename_input_csv, filename_output_graph, conf: Config):
    # '''Given the ansys .msh file and the .csv feature file, saves in memory the complete graph from torch_geometric'''

    # mesh = read_mesh(filename_input_msh, mode="meshio")
    # features = pd.read_csv(filename_input_csv)

    # map_mesh_to_feature = match_mesh_and_feature_pts(mesh, features, conf)

    # reduced_features = features[features.columns.difference(conf.features_to_remove)]
    # mesh_features = reduced_features[reduced_features.columns.difference(conf.features_coordinates)].iloc[map_mesh_to_feature]
    # mesh_coords = reduced_features[conf.features_coordinates].iloc[map_mesh_to_feature]

    # mesh_features = features[features.columns.difference(conf.features_to_remove)].iloc[map_mesh_to_feature]

    # ##### To get full graph node-node connectivity
    # edges = get_all_edges(mesh)

    # data = Data(x=torch.tensor(mesh_features.to_numpy()),
    #             edge_index=torch.tensor(edges).t().contiguous(), 
    #             pos=torch.tensor(mesh.points))
    
    # torch.save(data, filename_output_graph)

    # return data


def make_cell(a):
    cell = list(a[0,:])
    cell_closing_value = cell[0]
    for i in range(1,len(a)):
        correspondance = np.argwhere(a[:,0]==cell[i])
        if correspondance.shape not in {(0,0), (0,1)}:

            if correspondance.shape[0] > 1:
                print("WARNING: this cell could be rotated in the other direction (msh is not coherent: adjacency list returns faces that do not recreate 'cyclically' a cell)")
                print(a)
                # more than one match (should be always this case)
                for tmp in correspondance:
                    tmp = tmp[0]
                    value = a[tmp, 1]
                    if (value not in cell) or (i==len(a)-1 and value == cell_closing_value):
                        idx = tmp
                        break
            else:
                idx = correspondance[0][0]

            cell += [a[idx,1]]

        else:
            correspondance = np.argwhere(a[:,1]==cell[i])

            if correspondance.shape[0] > 1:
                # more than one match (should be always this case)
                for tmp in correspondance:
                    tmp = tmp[0]
                    value = a[tmp, 0]
                    if (value not in cell) or (i==len(a)-1 and value == cell_closing_value):
                        idx = tmp
                        break
            else:
                idx = correspondance[0][0]

            cell += [a[idx,0]]
            print("WARNING: this cell could be rotated in the other direction (msh is not coherent: adjacency list returns faces that do not recreate 'cyclically' a cell)")
            print(a)# examd faulty cell: [[A, B], [B, C], [A, C]] --> connecting right side of each element to a corrisponding left side is impossible (C is never on the left)
            # this means that one of the faces is given incorrectly from the msh, and we don't know the correct ordering of the vertices indexes inside the cell
            
    assert cell[0] == cell[-1], "Something broken in reconstruction of cell"
    assert len(cell) == len(a)+1, "Something strange happened"
    return cell[:-1]


def get_adjacency_list(mesh, idx_component):
    return(mesh.info["elements"][mesh.info["zone_id_list_cellblocks"][idx_component]]["adj_data"])


def recreate_cells(mesh: meshio.Mesh, conf:Config):
    '''
    Given a meshio.Mesh returns:
    - cellblocks: dict [key, val]
        - key: str --> accepted toughio.Mesh format for cell type
        - value: list --> list of CELL vertices, all with the dimension stated in key
    - cell_vertices_list: list --> same as above but the cells are ORDERED as stated in the .msh file, 
        useful for final indexing of cells. Here, cells can have different dimensions
    - cell_connectivity: np.array [n_edges, 2] where on the same row you have 2 indices of adjacent cells
    '''

    edge_list = np.concatenate([c.data for c in mesh.cells])
    adjacency_list = np.concatenate([get_adjacency_list(mesh, i) for i in range(len(mesh.cells))])

    if not len(mesh.info["cell_type_cumulative"]) == 1:
        raise NotImplementedError("Not implemented disconnected meshes yet") # TODO: does "cell_type_cumulative" always appear only once in msh?
    
    cell_types_list = mesh.info["cell_type_cumulative"][0]

    unique_vals, unique_counts = np.unique(cell_types_list, return_counts=True)

    # TODO: make it more robust for 3D
    cellblocks = {
        "triangle": [],
        "quad":     [],
    }
    cell_vertices_list = []

    for i, cell_type in enumerate(cell_types_list):

        idxs_l = np.argwhere(adjacency_list[:,0]== i+1)[:,0]
        tmp = [edge_list[j] for j in idxs_l]

        idxs_r = np.argwhere(adjacency_list[:,1]== i+1)[:,0]
        tmp += [np.flip(edge_list[j]) for j in idxs_r]

        tmp = np.stack(tmp, axis=0)

        cell = make_cell(tmp)
        cellblocks[conf.cell_type_dict[cell_type]].append(cell) 
        cell_vertices_list.append(cell)

    for val, count in zip(unique_vals, unique_counts):
        assert len(cellblocks[conf.cell_type_dict[val]]) == count, "Definition of cells in .msh doesn't correspond to reconstructed cells"

    # We remove all the adjaceny pairs with a 0 in it because in .msh it means that a face doesn't have both sides adjacent to a cell
    # Since we only care cell-cell adjacency, we discard it 
    real_adjacency = adjacency_list[np.logical_and(adjacency_list[:,0] != 0, adjacency_list[:,1] != 0)]

    # Bidirectional edges, create a flipped copy
    cell_connectivity = np.concatenate([real_adjacency, np.flip(real_adjacency, axis=1)], axis=0)

    # Sanity check, remove duplicates (there shouldn't be any)
    cell_connectivity = np.unique(cell_connectivity, axis=0) 
    
    # Bring all connections in [0, n_cells-1] (instead now they were in [0, n_cells], where 0 meant "no_connection")
    cell_connectivity -= 1

    return cellblocks, cell_vertices_list, cell_connectivity


def map_vertex_pair_to_face_idx(vertex_pair, vertices_in_faces):
    tmp = np.nonzero(np.logical_and(vertices_in_faces[:,0] == vertex_pair[0], 
                                    vertices_in_faces[:,1] == vertex_pair[1]))[0]
    if len(tmp) == 1:
        return tmp
    else: # swap vertices
        return np.nonzero(np.logical_and(vertices_in_faces[:,0] == vertex_pair[1], 
                                         vertices_in_faces[:,1] == vertex_pair[0]))[0]


def face_id_inside_faceblock_to_mesh_face_id(face_id_inside_faceblock, faceblock_id, len_faceblocks):
    return face_id_inside_faceblock + np.sum(len_faceblocks[:faceblock_id+1])


def shoelace(coords):
    idx_x = np.arange(coords.shape[0]-1)
    idx_y = idx_x+1
    return np.abs(np.sum(coords[idx_x,0]*coords[idx_y,1]) - coords[-1,0]*coords[0,1]) / 2


def get_cell_data(mesh: meshio.Mesh, conf: Config):
    '''
    Given a mesh returns:
        - cell_center_positions: np.array with shape [n_cells, 3] (x,y,z)
        - cell_center_cell_center_edges: np.array with shape [n_cell_cell_edges, 2], where both values inside a row are a index of cell_center_positions
        - cell_node_components: np.array with shape [n_cells]
    '''
    dict_vertices_in_cells, vertices_in_cells, CcCc_edges_bidir = recreate_cells(mesh, conf)

    # Create a mesh from cells instead than from faces
    cell_mesh = toughio.Mesh(mesh.points, [(key, np.stack(val)) for key, val in dict_vertices_in_cells.items()])

    cell_volumes = np.fromiter(map(shoelace, [mesh.points[v+[v[0]]][:,:2] for v in vertices_in_cells]), dtype=np.float64)

    return cell_mesh.centers, CcCc_edges_bidir, vertices_in_cells, cell_volumes


def get_face_data(face_mesh: meshio.Mesh, vertices_in_cells):

    vertices_in_faces = np.concatenate([c.data for c in face_mesh.cells], axis=0) # face_list[face_idx] = [list_of_node_idxs_in_that_face]
    tmp = set([frozenset([vertex_pair[0], vertex_pair[1]]) for vertex_pair in vertices_in_faces])
    vertices_in_faces = np.stack([np.array(list(fset)) for fset in tmp])

    face_areas = np.linalg.norm(face_mesh.points[vertices_in_faces[:,1]] - 
                                    face_mesh.points[vertices_in_faces[:,0]], axis=1)

    face_mesh = toughio.Mesh(face_mesh.points, [("line",vertices_in_faces)])
    face_center_positions = face_mesh.centers
    
    CcFc_edges = []
    FcFc_edges = []
    for i, vert_in_cell in enumerate(vertices_in_cells):
        face_ids_in_cell = []
        for vertex_pair in np.lib.stride_tricks.sliding_window_view(vert_in_cell+[vert_in_cell[0]], 2):
            # vertex_pair = pair of nodes on an face
            face_id = map_vertex_pair_to_face_idx(vertex_pair, vertices_in_faces)[0]
            CcFc_edges.append([i, face_id])
            face_ids_in_cell.append(face_id)
        
        # add edges between faces (face centers) of the same cell
        FcFc_edges += list(itertools.combinations(face_ids_in_cell, 2))

    CcFc_edges = np.stack(CcFc_edges)
    FcFc_edges = np.stack(FcFc_edges)
    
    return face_center_positions, FcFc_edges, vertices_in_faces, CcFc_edges, face_areas


def get_labels(positions, csv_filename, conf, check_biunivocity):
    '''Returns a [len(positions), N_features] matrix'''
    # eps = conf.epsilon_for_point_matching

    features = pd.read_csv(csv_filename)
    features.columns = [f_name.strip() for f_name in features.columns]

    assert len(positions) == len(features), f"Number of points ({len(positions)}) and CSV rows ({len(features)}) do not match"

    features = features[features.columns.difference(conf.features_to_remove)]

    map_pos_to_feature = np.zeros([len(positions), 2])
    if conf.dim == 2:
        pts = positions[:,:2]
        ord_pos, pos_old_idxs = sort_matrix(pts)
        ord_features, features_old_idxs = sort_matrix(features[conf.features_coordinates].to_numpy())
        
        for i, (pos_old_idx, ft_old_idx, pos) in enumerate(zip(pos_old_idxs,
                                                            features_old_idxs,
                                                            ord_pos)):
            map_pos_to_feature[i, 0], map_pos_to_feature[i, 1] = pos_old_idx, features_old_idxs[np.argmin(cdist([pos], ord_features)[0])]
        # features_bounds = np.sort(np.stack([ord_features*(1-eps), ord_features*(1+eps)]), axis=0)
        # features_bound_1 = features_bounds[0,...]
        # features_bound_2 = features_bounds[1,...]

        # for i, (pos_old_idx, ft_old_idx, pos, ft_b1, ft_b2) in enumerate(zip(pos_old_idxs,
        #                                                                       features_old_idxs,
        #                                                                       ord_pos,
        #                                                                       features_bound_1,
        #                                                                       features_bound_2)):
        #     if (ft_b1<=pos).all() and (pos<=ft_b2).all():
        #         map_pos_to_feature[i, 0], map_pos_to_feature[i, 1] = pos_old_idx, ft_old_idx
        #     else: # due to precision errors, ordering of the nodes could be different
        #         found = False
        #         for j, (tmp_1, tmp_2) in enumerate(zip(features_bound_1, features_bound_2)): # check them one by one
                    
        #             if (tmp_1<=pos).all() and (pos<=tmp_2).all():
        #                 map_pos_to_feature[i, 0], map_pos_to_feature[i, 1] = pos_old_idx, features_old_idxs[j]
        #                 found = True
        #                 break
        #         if not found:
        #             raise ValueError("Points do not correspond")
                
        assert len(np.unique(map_pos_to_feature[:,0])) == len(map_pos_to_feature), "Not biunivocal, for some mesh point there is no correspondance in features"
        assert len(np.unique(map_pos_to_feature[:,1])) == len(map_pos_to_feature), "Not biunivocal, for some feature points there is no correspondant mesh point"
        
        map_pos_to_feature, _ = sort_matrix(map_pos_to_feature)
        map_pos_to_feature = map_pos_to_feature[:,1].astype(np.int64)

        # Remove the position features (already in the graph)
        features = features[features.columns.difference(conf.features_coordinates)]
        # order the Df columns
        features = features[conf.features_to_keep]
        # order the rows
        return features.iloc[map_pos_to_feature]
    else:
        raise NotImplementedError("Not implemented for 3d")
    

def convert_msh_to_mesh_complete_info_obj(
        conf: Config,
        filename_input_msh,
        filename_output_mesh_complete_obj: Optional[str] = None,
        add_distance_from_BC: Optional[bool] = False,
        ):

    '''Given an ASCII .msh file from ANSA, returns a graph and saves it to memory'''
    if filename_output_mesh_complete_obj is None:
        print("Warning: no output location specified, meshCompleteObj will NOT be saved to disk")
        
    conf = Config()

    mesh = read_mesh(filename_input_msh, mode="meshio", conf=conf)

    cell_center_positions, CcCc_edges_bidir, vertices_in_cells, cell_volumes = get_cell_data(mesh, conf)
    
    face_center_positions, FcFc_edges, vertices_in_faces, CcFc_edges, face_areas = get_face_data(mesh, vertices_in_cells)
    
    mesh_complete_instance = MeshCompleteInfo(
        conf,
        filename_input_msh,
        mesh,
        cell_center_positions,
        CcCc_edges_bidir,
        vertices_in_cells,
        cell_volumes,
        face_center_positions,
        FcFc_edges,
        vertices_in_faces,
        CcFc_edges,
        face_areas,
    )

    if add_distance_from_BC:
        mesh_complete_instance.add_distance_from_BC()

    if filename_output_mesh_complete_obj is not None:
        mesh_complete_instance.save_to_disk(filename_output_mesh_complete_obj)

    return mesh_complete_instance


def normalize_features(features, conf):
    match conf.feature_normalization_mode:
        case "None":
            return features
        case "Physical":
            features[:, conf.graph_node_feature_dict["v_t"]] /= conf.air_speed
            features[:, conf.graph_node_feature_dict["v_n"]] /= conf.air_speed
            features[:, conf.graph_node_feature_dict["p"]] /= (conf.air_speed**2)/2
            return features
        case _:
            raise NotImplementedError("Only implemented 'None' and 'Physical'")


def normalize_label(values, label, conf):
    label_normalization_mode = conf["label_normalization_mode"]
    dict_labels_train = conf["dict_labels_train"] 

    # TODO: maybe add a graph-wise norm?
    match label_normalization_mode[label]["main"]:
        case "physical":
            if label in ["x-velocity", "y-velocity"]:
                tmp = values / conf["air_speed"]
            elif label == "pressure":
                tmp = values / (conf["air_speed"]**2)/2
            else:
                raise ValueError(f"Physical normalization no available for {label}")
        case "max-normalization":
            if (label in ["x-velocity", "y-velocity"]) and \
                    label_normalization_mode[label].get("magnitude", False):
                tmp = values / dict_labels_train["max_magnitude"]["v_mag"]
            else:
                tmp = values / dict_labels_train["max_magnitude"][label]

        case "standardization":
            if (label in ["x-velocity", "y-velocity"]) and \
                    label_normalization_mode[label].get("magnitude", False):
                tmp = (values-dict_labels_train["mean"]["v_mag"])/ \
                    dict_labels_train["std"]["v_mag"]
            else:
                tmp = (values-dict_labels_train["mean"][label])/ \
                    dict_labels_train["std"][label]
        case _:
            raise NotImplementedError()
    
    return tmp


def denormalize_label(values, label, conf):
    '''
    Connected to NormalizeLabels augmentation
    '''
    label_normalization_mode = conf["label_normalization_mode"]
    dict_labels_train = conf["dict_labels_train"] 

    match label_normalization_mode[label]["main"]:
        case "physical":
            if label in ["x-velocity", "y-velocity"]:
                values *= conf["air_speed"]
            elif label == "pressure":
                values *= (conf["air_speed"]**2)/2
            else:
                raise ValueError(f"Physical normalization no available for {label}")
        case "max-normalization":
            if (label in ["x-velocity", "y-velocity"]) and \
                    label_normalization_mode[label].get("magnitude", False):
                values *= dict_labels_train["max_magnitude"]["v_mag"]
            else:
                values *= dict_labels_train["max_magnitude"][label]

        case "standardization":
            if (label in ["x-velocity", "y-velocity"]) and \
                    label_normalization_mode[label].get("magnitude", False):
                values = values*dict_labels_train["std"]["v_mag"]+ \
                            dict_labels_train["mean"]["v_mag"]
            else:
                values = values*dict_labels_train["std"][label]+\
                            dict_labels_train["mean"][label]
        case _:
            raise NotImplementedError()
        
    return values


def plot_gt_pred_label_comparison(data: Data, model_output, conf, run_name: Optional[str]= None):
    data.name = data.name.removesuffix("_ascii.msh")
    with open(os.path.join(conf["EXTERNAL_FOLDER_MESHCOMPLETE_W_LABELS"], data.name+".pkl"), "rb") as f:
        meshCI = pickle.load(f)
        
    print(f"Plotting {meshCI.name}")
    pred_labels = model_output[0].detach().numpy()
    pred = np.zeros_like(pred_labels)
    for i, (values, label) in enumerate(zip(pred_labels.T, conf.labels_to_keep_for_training)):
        pred[:,i] = denormalize_label(values, label, conf)
    
    meshCI.set_conf(conf)
    # meshCI.plot_mesh(labels=pred, run_name = run_name, conf=conf)
    meshCI.plot_all_results(data, (pred, model_output[1]), run_name = run_name, conf=conf)


def get_inward_normal_areas(
        faces_idxs,
        face_areas,
        faces_x_component,
        faces_y_component,
        CcFc_edges,
        cell_center_positions,
        face_center_positions,):
    '''Inward = going away from fluid bulk, towards inside of boundary'''

    faces_x_component = faces_x_component[faces_idxs]
    faces_y_component = faces_y_component[faces_idxs]

    CcFc_edges = torch.tensor(CcFc_edges)
    normal_to_surface = torch.stack((-faces_y_component, faces_x_component), axis=1)
    opposite_normal_to_surface = torch.stack((faces_y_component, -faces_x_component), axis=1)

    in_tensor = CcFc_edges[:,1].view((CcFc_edges[:,1].shape[0], 1))
    vectorized_func = torch.vmap(lambda x: (x == faces_idxs).sum() > 0)
    CcFc_edges_indexes_of_surface_faces =  vectorized_func(in_tensor)

    CcFc_edges_component = CcFc_edges[CcFc_edges_indexes_of_surface_faces]
    CcFc_vectors = torch.tensor(face_center_positions[CcFc_edges_component[:,1]][:,:2] - 
                                    cell_center_positions[CcFc_edges_component[:,0]][:,:2])

    inward_normals = torch.where(
        ((normal_to_surface*CcFc_vectors).sum(dim=1) > 0).view(-1,1).repeat(1,2), 
            normal_to_surface, opposite_normal_to_surface)

    return inward_normals*face_areas[faces_idxs].view(-1,1).repeat(1,2)


def get_forces(
        conf:Config, 
        data:Data, 
        pressure_values, 
        velocity_derivatives=None, 
        turbulent_values=None,
        denormalize=False,
        from_boundary_sampling=False):
    
    flap_faces = data.x_additional[:, conf.graph_node_features_not_for_training["is_flap"]].nonzero()[:,0]
    tyre_faces = data.x_additional[:, conf.graph_node_features_not_for_training["is_tyre"]].nonzero()[:,0]

    if denormalize:
        pressure_values = denormalize_label(pressure_values, "pressure", conf)
    pressure_forces_flap = torch.sum((data.inward_normal_areas[flap_faces]*
                                pressure_values[flap_faces].view(-1,1).repeat(1,2)), dim=0)
    pressure_forces_tyre = torch.sum((data.inward_normal_areas[tyre_faces]*
                                pressure_values[tyre_faces].view(-1,1).repeat(1,2)), dim=0)
    
    shear_stress_flap = torch.tensor((0.,0.))
    shear_stress_tyre = torch.tensor((0.,0.))
    if velocity_derivatives is not None:
        assert turbulent_values is not None, "Cannot compute forces without turbulent values"

        if from_boundary_sampling:
            idxs_is_BC = (data.x_mask[:,-1] == 1)
            flap_faces_vel = data.x_additional[idxs_is_BC, conf.graph_node_features_not_for_training["is_flap"]].nonzero()[:,0]
            tyre_faces_vel = data.x_additional[idxs_is_BC, conf.graph_node_features_not_for_training["is_tyre"]].nonzero()[:,0]
            u_x, u_y, v_x, v_y = velocity_derivatives
        else:
            u_x, u_y, v_x, v_y = velocity_derivatives[:,0], velocity_derivatives[:,1], velocity_derivatives[:,2], velocity_derivatives[:,3]
            flap_faces_vel = flap_faces
            tyre_faces_vel = tyre_faces

        if denormalize:
            u_x, u_y = vmap(denormalize_label, in_dims=(0,None,None))(torch.stack((u_x, u_y)), "x-velocity", conf)
            v_x, v_y = vmap(denormalize_label, in_dims=(0,None,None))(torch.stack((v_x, v_y)), "y-velocity", conf)
            k = denormalize_label(turbulent_values[:,0], "turb-kinetic-energy", conf)
            w = denormalize_label(turbulent_values[:,1], "turb-kinetic-energy", conf)
            k, w = torch.clamp(k, min=0), torch.clamp(w, min=conf.w_min_for_clamp)
        else:
            # If there is no need to denormalize, it means it comes from csv --> no need to clamp
            k, w = turbulent_values[:,0], turbulent_values[:,1]
            
        viscosity = (conf.air_kinematic_viscosity + k/w).view(-1,1).repeat(1,2) # FIXME: constants

        def compute_stress_euclidian_components(idxs, idxs_vel, viscosity, inward_normal_areas, u_x, u_y, v_x, v_y, k):
                '''(jacU + jacU.T) dot n --> gives a tensor of stresses, we need the x and y component of its tangent part'''
                # FIXME: constants + physics
                k, viscosity = k[idxs], viscosity[idxs]
                n_x, n_y = inward_normal_areas[idxs, 0], inward_normal_areas[idxs, 1]
                u_x, u_y, v_x, v_y = u_x[idxs_vel], u_y[idxs_vel], v_x[idxs_vel], v_y[idxs_vel]
                # TODO: do i need to sum normal stress from newton law with reynold's stress?
                # TODO: STILL NEED TO TAKE THE TANGENT FIRST!
                x_comp = 2*n_x*u_x + n_y*(u_x+v_y) # x component of stress vector (without turbulence)
                y_comp = 2*n_y*u_y + n_x*(u_x+v_y) # y component of stress vector (without turbulence)
                # x_comp = 2*n_x*(u_x-k/3) + n_y*(u_x+v_y) # x component of stress vector
                # y_comp = 2*n_y*(u_y-k/3) + n_x*(u_x+v_y) # y component of stress vector
                return torch.stack((x_comp, y_comp)).T*viscosity

        shear_stress_flap = torch.sum(compute_stress_euclidian_components(
            flap_faces, flap_faces_vel, viscosity, data.inward_normal_areas, u_x, u_y, v_x, v_y, k
        ), dim=0)

        shear_stress_tyre = torch.sum(compute_stress_euclidian_components(
            tyre_faces, tyre_faces_vel, viscosity, data.inward_normal_areas, u_x, u_y, v_x, v_y, k
        ), dim=0)
    
    pressure_forces = {
        "flap": pressure_forces_flap,
        "tyre": pressure_forces_tyre,}
    pressure_forces["car"] = sum(pressure_forces.values())

    shear_stress_forces = {
        "flap": shear_stress_flap,
        "tyre": shear_stress_tyre,}
    shear_stress_forces["car"] = sum(shear_stress_forces.values())
    
    return pressure_forces, shear_stress_forces


class MeshCompleteInfo:
    def __init__(
            self,
            conf: Config,
            path: str,
            mesh: meshio.Mesh,
            cell_center_positions,
            CcCc_edges_bidir,
            vertices_in_cells,
            cell_volumes,
            face_center_positions, 
            FcFc_edges, 
            vertices_in_faces,
            CcFc_edges,
            face_areas,
    ) -> None:
        self.conf: Config = conf
        self.path = path
        self.name = path.split(os.sep)[-1].removesuffix(".pkl")
        self.group = 1 if "2dtc_001R" in self.name else 2
        self.mesh = mesh
        self.cell_center_positions = cell_center_positions
        self.CcCc_edges_bidir = CcCc_edges_bidir
        self.vertices_in_cells = vertices_in_cells
        self.cell_volumes = cell_volumes
        self.face_center_positions = face_center_positions 
        self.FcFc_edges = FcFc_edges 
        self.vertices_in_faces = vertices_in_faces
        self.CcFc_edges = CcFc_edges
        self.face_areas = face_areas

        self.face_center_features, self.face_center_features_mask, self.face_center_additional_features, self.inlet_points_positions = \
            self.get_face_BC_attributes()
        
        self.dist_from_BC = None
        self.vertex_labels = None
        self.face_center_labels = None
        self.cell_center_labels = None

        self.radial_attributes = self.get_radial_attributes()

        self.face_center_features = np.concatenate((self.face_center_features, self.radial_attributes), axis=1)


    def set_conf(self, conf):
        self.conf = conf

    def add_labels(self, labels_csv_filename, mode:Literal["vertex","element"]="element"):
        
        match mode:
            case "element":
                if self.conf.dim == 2:
                    self.face_center_labels = get_labels(
                        self.face_center_positions, 
                        labels_csv_filename, 
                        self.conf, 
                        check_biunivocity=True)
                elif self.conf.dim == 3:
                    raise NotImplementedError("Implement dim = 3")
                    self.cell_center_labels = get_labels(
                            self.cell_center_positions, 
                            labels_csv_filename, 
                            self.conf, 
                            check_biunivocity=True)
            case "vertex":
                self.vertex_labels = get_labels(
                    self.mesh.points, 
                    labels_csv_filename, 
                    self.conf, 
                    check_biunivocity=True)

    def add_labels_from_graph(
            self, 
            data: Union[Data, torch.Tensor, np.ndarray], 
            which_element_has_labels: Literal["vertex", "face", "cell"], 
            ordered_column_names: Optional[Union[list[str], str]] = None,
            overwrite = False,
            ):
        
        match data:
            case Data():
                labels = data.y.numpy()
            case torch.Tensor():
                labels = data.numpy()
            case np.ndarray():
                labels = data

        if len(labels.shape) == 1:
            labels = np.expand_dims(labels, axis=1)
        
        if ordered_column_names is None:
            ordered_column_names = self.conf.graph_node_final_features
        elif isinstance(ordered_column_names, str):
            ordered_column_names = [ordered_column_names]
    
        assert labels.shape[1] == len(ordered_column_names), f"Shapes do not match: data has {labels.shape[1]} columns and ordered_column_names are {len(ordered_column_names)}"

        match which_element_has_labels:
            case "vertex":
                assert labels.shape[0] == self.mesh.points.shape[0], f"Data label shape {labels.shape} does not match number of points in mesh {self.mesh.points.shape[0]}"
                obj = self.vertex_labels
            case "face":
                assert labels.shape[0] == self.face_center_positions.shape[0], f"Data label shape {data.y.shape} does not match number of faces in mesh {self.face_center_positions.shape[0]}"
                obj = self.face_center_labels
            case "cell":
                assert labels.shape[0] == self.cell_center_positions.shape[0], f"Data label shape {data.y.shape} does not match number of cells in mesh {self.cell_center_positions.shape[0]}"
                obj = self.cell_center_labels
        

        if obj is None:
            obj = pd.DataFrame(labels, columns=ordered_column_names)
        else:
            for i, column in enumerate(ordered_column_names):
                if overwrite:
                    obj[column] = labels[:,i]
                else:
                    if column not in set(obj.columns):
                        obj[column] = labels[:,i]


    def get_face_BC_attributes(self, conf:Optional[Config]=None):
        if conf is None:
            conf: Config = self.conf
        
        mesh, face_center_positions, vertices_in_faces, face_areas = \
            self.mesh, self.face_center_positions, self.vertices_in_faces, self.face_areas
        
        point_positions = mesh.points

        face_center_attr_BC = np.zeros((len(face_center_positions), len(conf.graph_node_feature_dict)))
        face_center_attr_BC_mask = np.zeros((len(face_center_positions), len(conf.graph_node_feature_mask))).astype(bool)
        face_center_additional_attrs = np.zeros((len(face_center_positions), len(conf.graph_node_features_not_for_training)))

        # Add general "is_BC" mask
        face_center_attr_BC_mask[:,conf.graph_node_feature_mask["is_BC"]] = True # set all "is_BC?" masks = True and then set it to 0 in "interior" below

        face_spatial_dir = [point_positions[v[1]]-point_positions[v[0]] for v in vertices_in_faces]
        face_spatial_dir_norm = np.array([vec/np.linalg.norm(vec) for vec in face_spatial_dir])

        # face tangent versor components (x, y)
        tmp = face_spatial_dir_norm[:,0]
        face_center_attr_BC[:,conf.graph_node_feature_dict["tangent_versor_x"]] = np.where(tmp>=0, face_spatial_dir_norm[:,0], -face_spatial_dir_norm[:,0])
        face_center_attr_BC[:,conf.graph_node_feature_dict["tangent_versor_y"]] = np.where(tmp>=0, face_spatial_dir_norm[:,1], -face_spatial_dir_norm[:,1])
        # face_center_attr_BC[:,conf.graph_node_feature_dict["tangent_versor_angle"]] = np.arctan2(face_spatial_dir_norm[:,1], face_spatial_dir_norm[:,0])/np.pi
        face_center_attr_BC[:, conf.graph_node_feature_dict["face_area"]] = face_areas

        elem_info = mesh.info["elements"]
        zone_id_bc_type = {elem: elem_info[elem]["bc_type"] if "bc_type" in elem_info[elem].keys() else -1 for elem in elem_info}
        cellblock_idx_bc_type = [zone_id_bc_type[key] for key in mesh.info["zone_id_list_cellblocks"]]
        cellblock_idx_name = [mesh.info["global"][str(key)]["zone_name"] if str(key) in mesh.info["global"].keys() else "" for key in mesh.info["zone_id_list_cellblocks"] ]

        inlet_points = []

        for i, tmp in enumerate(zip(mesh.cells, cellblock_idx_bc_type, cellblock_idx_name)):
            faceblock, bc_type, name = tmp

            faces_of_faceblock_idxs = [map_vertex_pair_to_face_idx(vertex_pair, vertices_in_faces)[0] for vertex_pair in faceblock.data]
            points_of_faceblock_idxs = np.unique(np.concatenate([faceblock.data[:,0], faceblock.data[:,1]]))
            points_of_faceblock_positions = point_positions[points_of_faceblock_idxs]

            face_center_additional_attrs[faces_of_faceblock_idxs, conf.graph_node_features_not_for_training["component_id"]] = i

            match bc_type:
                case 2: # interior, no condition
                    face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_mask["is_BC"]] = False # set "is_BC" masks = False
                case 3: # wall, speed fixed depending on the name
                    face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_n"]] = 0
                    face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_mask["v_n"]] = True
                    face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["dp_dn"]] = 0
                    face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_mask["dp_dn"]] = True

                    if "ground" in name: # same tangential velocity as the air entering the domain
                        face_center_additional_attrs[faces_of_faceblock_idxs, conf.graph_node_features_not_for_training["ground"]] = True
                        face_center_attr_BC[faces_of_faceblock_idxs,conf.graph_node_feature_dict["tangent_versor_x"]] = 1
                        face_center_attr_BC[faces_of_faceblock_idxs,conf.graph_node_feature_dict["tangent_versor_y"]] = 0
                        face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_t"]] = conf.air_speed
                        face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_mask["v_t"]] = True

                    elif "tyre" in name: # in 2D, it rotates around the center with angular speed omega
                        face_center_additional_attrs[faces_of_faceblock_idxs, conf.graph_node_features_not_for_training["tyre"]] = True
                        Cx, Cy, R, sigma = hyperLSQ(points_of_faceblock_positions[:,:2])
                        omega = conf.air_speed / R
                        rays = face_center_positions[faces_of_faceblock_idxs][:,:2] - [Cx, Cy]
                        direction = np.cross(face_center_attr_BC[faces_of_faceblock_idxs,:2], rays)

                        # for all faces, x_tangent versor is positive. Except here, where the tyre is rotating, we add information
                        # and give the correct tangent feature according to the rotation
                        face_center_attr_BC[faces_of_faceblock_idxs,conf.graph_node_feature_dict["tangent_versor_x"]] = np.where(direction>=0, 
                            face_center_attr_BC[faces_of_faceblock_idxs,conf.graph_node_feature_dict["tangent_versor_x"]],
                            -face_center_attr_BC[faces_of_faceblock_idxs,conf.graph_node_feature_dict["tangent_versor_x"]])
                        face_center_attr_BC[faces_of_faceblock_idxs,conf.graph_node_feature_dict["tangent_versor_y"]] = np.where(direction>=0, 
                            face_center_attr_BC[faces_of_faceblock_idxs,conf.graph_node_feature_dict["tangent_versor_y"]],
                            -face_center_attr_BC[faces_of_faceblock_idxs,conf.graph_node_feature_dict["tangent_versor_y"]])

                        v_t = np.linalg.norm(rays, axis=1) * omega

                        # TODO: do we add a np.dot(v_t, face_spatial_dir_norm[:,:2]) to only get the component really tangent?
                        ### OSS: mean angle deviation is ~ 0.00116
                        # v_t_directional = np.cross([0,0,-omega], rays)[:,:2]
                        # v_t_versor = np.array([v/np.linalg.norm(v) for v in v_t_directional])
                        # face_versor = face_spatial_dir_norm[faces_of_faceblock_idxs][:,:2]
                        # right_direction_face_versor = np.array([f*s for f,s in zip(face_versor, np.sign(direction))])
                        # dot_products_v_f = [np.dot(f,v) for f, v in zip(right_direction_face_versor, v_t_versor)]
                        # error_in_angle = [np.arccos(d) for d in dot_products_v_f]
                        
                        face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_t"]] = v_t
                        face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_mask["v_t"]] = True

                        face_center_additional_attrs[faces_of_faceblock_idxs, conf.graph_node_features_not_for_training["is_car"]] = 1
                        face_center_additional_attrs[faces_of_faceblock_idxs, conf.graph_node_features_not_for_training["is_tyre"]] = 1
                    elif any(x in name for x in ["w0", "default-exterior"]): # fixed wall, doesn't move
                        if "w0" in name:
                            face_center_additional_attrs[faces_of_faceblock_idxs, conf.graph_node_features_not_for_training["main_flap"]] = True
                        else:
                            face_center_additional_attrs[faces_of_faceblock_idxs, conf.graph_node_features_not_for_training["second_flap"]] = True

                        face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_t"]] = 0
                        face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_mask["v_t"]] = True

                        face_center_additional_attrs[faces_of_faceblock_idxs, conf.graph_node_features_not_for_training["is_car"]] = 1
                        face_center_additional_attrs[faces_of_faceblock_idxs, conf.graph_node_features_not_for_training["is_flap"]] = 1
                    else:
                        raise NotImplementedError(f"Didn't implement this kind of wall yet: {name}")
                case 5: # pressure-outlet

                    # vertical, going up
                    face_center_attr_BC[faces_of_faceblock_idxs,conf.graph_node_feature_dict["tangent_versor_x"]] = 0
                    face_center_attr_BC[faces_of_faceblock_idxs,conf.graph_node_feature_dict["tangent_versor_y"]] = 1

                    face_center_additional_attrs[faces_of_faceblock_idxs, conf.graph_node_features_not_for_training["p_outlet"]] = True
                    face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["p"]] = conf.relative_atmosferic_pressure
                    face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_mask["p"]] = True
                    face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["dv_dn"]] = 0
                    face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_mask["dv_dn"]] = True
                case 7: # simmetry, normal derivative = 0
                    
                    # TODO: to follow better the tyre, we could go up if left of the tyre and down if right of the tyre
                    face_center_attr_BC[faces_of_faceblock_idxs,conf.graph_node_feature_dict["tangent_versor_x"]] = np.where(
                        face_center_attr_BC[faces_of_faceblock_idxs,conf.graph_node_feature_dict["tangent_versor_x"]] > 0.8,
                        1,  # horizontal
                        0   # vertical
                    )

                    face_center_attr_BC[faces_of_faceblock_idxs,conf.graph_node_feature_dict["tangent_versor_y"]] = np.where(
                        face_center_attr_BC[faces_of_faceblock_idxs,conf.graph_node_feature_dict["tangent_versor_y"]] > 0.8,
                        1,  # vertical, going up
                        0   # horizontal
                    )

                    face_center_additional_attrs[faces_of_faceblock_idxs, conf.graph_node_features_not_for_training["simmetry"]] = True
                    face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_n"]] = 0
                    face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_mask["v_n"]] = True
                    
                    face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["dp_dn"]] = 0
                    face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_mask["dp_dn"]] = True
                case 10: # velocity_inlet

                    # vertical, going up
                    face_center_attr_BC[faces_of_faceblock_idxs,conf.graph_node_feature_dict["tangent_versor_x"]] = 0
                    face_center_attr_BC[faces_of_faceblock_idxs,conf.graph_node_feature_dict["tangent_versor_y"]] = 1

                    inlet_points.append(points_of_faceblock_positions)
                    face_center_additional_attrs[faces_of_faceblock_idxs, conf.graph_node_features_not_for_training["v_inlet"]] = True
                    face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_t"]] = 0
                    face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_mask["v_t"]] = True
                    face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_n"]] = conf.air_speed
                    face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_mask["v_n"]] = True
                    face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["dp_dn"]] = 0
                    face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_mask["dp_dn"]] = True

                case _:
                    raise NotImplementedError("Didn't implement this kind of BC yet")
                
        if len(inlet_points) >= 1:
            inlet_points = np.stack(inlet_points)[0]
        else:
            print("WARNING, no inlet points found")
        
        return face_center_attr_BC, face_center_attr_BC_mask, face_center_additional_attrs, inlet_points

    def get_triangulated_cells(self):
        vertices_in_cells, pos, CcFc_edges = self.vertices_in_cells, self.mesh.points, self.CcFc_edges

        # faces_in_cells = []
        # for i in range(len(vertices_in_cells)): # n_cells
        #     faces_in_cells.append(CcFc_edges[CcFc_edges[:,0]==i,1])

        all_triangles = []
        cell_idxs = []
        # faces_enclosing_triangles = []

        for i, vertices in enumerate(vertices_in_cells):
            if len(vertices) > 3:
                tri = Delaunay(pos[vertices, :2], qhull_options="QJ")

                for simplex in tri.simplices:
                    all_triangles.append(tri.points[simplex])
                    cell_idxs.append(i)
                    # faces_enclosing_triangles.append(faces)
            else:
                all_triangles.append(pos[vertices, :2])
                cell_idxs.append(i)
                # faces_enclosing_triangles.append(faces)
        
        return np.stack(all_triangles), np.stack(cell_idxs)


    def update_path(self, path):
        self.path = path
        self.name = path.split(os.sep)[-1].removesuffix(".pkl")

    def get_graph(self) -> Data:
        return torch.load(os.path.join(self.conf.EXTERNAL_FOLDER_GRAPHS, self.name+".pt"))
    
    def save_to_disk(self, filename):
        '''
        Pickle dumps the object to the filename.

        To reload it:
        with open(filename, 'rb') as f:
            meshCompleteInfoInstance = pickle.load(f)
        '''
        with open(filename, "wb") as f:
            pickle.dump(self, f, -1)
        
        self.update_path(filename)


    def get_distance_from_BC(self, mode: Literal["vertex", "face", "cell"]="face"):
        # if not mode=="face":
        # raise NotImplementedError("Need to change implementation")

        is_BC = self.face_center_features_mask[:,-1]
        edges = self.FcFc_edges # TODO: bidirectional?

        n_nodes = len(is_BC)
        A = np.zeros((n_nodes, n_nodes)).astype(np.float64)
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            A[i,j] = 1.
        
        A += A.T # add bidirectional edges

        rust_g = PyGraph.from_adjacency_matrix(A)
        dist_m = distance_matrix(rust_g, parallel_threshold=10000)

        dist_from_bc_per_node = np.min(dist_m[:, is_BC], axis=1)
        self.dist_from_BC = dist_from_bc_per_node
        return dist_from_bc_per_node


    def get_radial_attributes(self):
        # x_min = np.min(self.face_center_positions[:,0])
        # x_max = np.max(self.face_center_positions[:,0])
        # y_min = np.min(self.face_center_positions[:,1])
        # y_max = np.max(self.face_center_positions[:,1])
        
        n_points = self.face_center_positions.shape[0]
        radial_attributes = np.zeros((n_points, self.conf.n_theta_bins*self.conf.quantile_values)) + self.conf.default_radial_attribute_value
        th_bin_size = 2*np.pi/(self.conf.n_theta_bins)

        BC_positions = self.face_center_positions[self.face_center_features_mask[:,self.conf.graph_node_feature_mask["is_BC"]]==True]
        
        all_thetas = cdist(self.face_center_positions, BC_positions, lambda n, BC: np.arctan2(BC[1]-n[1], BC[0]-n[0]) ) # 3-dimensional

        for i in range(n_points):
            thetas = all_thetas[i]
            for j in range(self.conf.n_theta_bins):
                th_min, th_max = (-np.pi + j*th_bin_size), (-np.pi + (j+1)*th_bin_size)
                attr_idx_min = j*self.conf.quantile_values
                attr_idx_max = (j+1)*self.conf.quantile_values

                point_idxs = np.logical_and(th_min <= thetas, thetas <= th_max)

                if point_idxs.sum() > 0:
                    vectors_inside_bin = BC_positions[point_idxs] - self.face_center_positions[i]
                    vector_norms_inside_bin = np.linalg.norm(vectors_inside_bin, axis=1)
                    attributes_for_bin = np.quantile(vector_norms_inside_bin, self.conf.distance_quantiles_angular_bin)
                    radial_attributes[i, attr_idx_min:attr_idx_max] = attributes_for_bin

                    # plt.scatter(vectors_inside_bin[:,0], vectors_inside_bin[:,1])
                    # plt.scatter(attributes_for_bin, np.tan((th_min+th_max)/2)*attributes_for_bin)
                    # x = np.linspace(-5, 15, 100)
                    # plt.plot(x, np.tan(th_min)*x)
                    # plt.plot(x, np.tan(th_max)*x)
                    # plt.xlim((x_min-self.face_center_positions[i, 0]), (x_max-self.face_center_positions[i, 0]))
                    # plt.ylim((y_min-self.face_center_positions[i, 1]), (y_max-self.face_center_positions[i, 1]))
                    # plt.show()

        return radial_attributes


    def plot_mesh(self, 
                what_to_plot = None,
                labels: Optional[torch.Tensor] = None,
                run_name: Optional[str] = None,
                return_meshes: Optional[bool] = None,
                conf = None,
                ):
        '''
        what_to_plot whould be a list of tuples (tup[0], tup[1], tup[2]):
            - tup[0]: Literal["vertex", "face", "cell"] --> can be "vertex", "face" or "cell"

            - tup[1]: Literal["label", "features"] --> should be "label". If tup[0] is "face" you can also have tup[1] = "features"

            - tup[2] should be:
                - if tup[1] is label, an element of conf.features_to_keep
                - if tup[1] is feature, a key of conf.graph_node_feature_dict
                - additional special value "streamlines" in case --> ("cell", "label", "velocity") --> automatically add streamlines
        '''
        assert self.conf["dim"] == 2, "Implement dim = 3"
        assert (what_to_plot is not None) or (labels is not None) or (return_meshes is not None), "Nothing to plot specified"
        # TODO: should we create permanent objects to avoid recomputation?

        if self.vertex_labels is not None:
            vertex_pyv_mesh = toughio.from_meshio(self.mesh).to_pyvista()

        face_list_for_pyvista = np.concatenate(
                [[len(v), *v] for v in self.vertices_in_faces]
            )
        facetype_list_for_pyvista = [pyvista.CellType.LINE for _ in self.vertices_in_faces]

        face_pyv_mesh = pyvista.UnstructuredGrid(
                            face_list_for_pyvista,
                            facetype_list_for_pyvista,
                            self.mesh.points
                        )
        
        if self.face_center_labels is not None:
            
            cell_list_for_pyvista = np.concatenate(
                        [[len(v), *v] for v in self.vertices_in_cells]
                    )

            celltype_list_for_pyvista = [pyvista.CellType.POLYGON for _ in self.vertices_in_cells]

            cell_pyv_mesh = pyvista.UnstructuredGrid(
                cell_list_for_pyvista,
                celltype_list_for_pyvista,
                self.mesh.points,
            )

            self.faces_in_cells = pd.DataFrame(data=self.CcFc_edges, columns=["cell_idx", "face_idx"]).groupby("cell_idx")["face_idx"].apply(list)
            # TODO: improve from mean to weighted avg depending on distance
            self.cell_center_labels = pd.DataFrame(
                [np.mean(self.face_center_labels.iloc[faces_idx], axis=0) for faces_idx in self.faces_in_cells]
            )

        if return_meshes is not None:
            tmp = []
            if self.vertex_labels is not None:
                tmp.append(vertex_pyv_mesh)
            tmp.append(face_pyv_mesh)
            if self.face_center_labels is not None:
                tmp.append(cell_pyv_mesh)
            return tmp


        if labels is not None:
            # columns = self.conf.labels_to_keep_for_training # TODO: update all MeshComplete files from scratch
            columns = conf["labels_to_keep_for_training"]
            face_pred_labels = pd.DataFrame(labels, columns=columns)
            cell_pred_lables = pd.DataFrame(
                [np.mean(face_pred_labels.iloc[faces_idx], axis=0) for faces_idx in self.faces_in_cells]
            )

            for i, lab in enumerate(columns):
                cell_pyv_mesh.cell_data[lab] = self.cell_center_labels[lab]
                cell_pyv_mesh.cell_data[lab+"_pred"] = cell_pred_lables[lab]
                cell_pyv_mesh.cell_data[lab+"_diff"] = self.cell_center_labels[lab] - cell_pred_lables[lab]
                face_pyv_mesh.cell_data[lab] = self.face_center_labels[lab]
                face_pyv_mesh.cell_data[lab+"_pred"] = face_pred_labels[lab]
                face_pyv_mesh.cell_data[lab+"_diff"] = self.face_center_labels[lab] - face_pred_labels[lab]

            if run_name is not None:
                off_screen = True
            else:
                off_screen = False
            pl = pyvista.Plotter(shape=(3, len(columns)), off_screen=off_screen)
            for i, lab in enumerate(columns):
                pl.subplot(0,i)
                pl.add_mesh(cell_pyv_mesh.copy(), scalars=lab, 
                    lighting=False, 
                    scalar_bar_args={"title":f"GT_{lab}"},
                    cmap="Spectral")
                pl.camera_position = "xy"

                pl.subplot(1,i)
                pl.add_mesh(cell_pyv_mesh.copy(), scalars=lab+"_pred", 
                    lighting=False, 
                    scalar_bar_args={"title":f"PRED_{lab}"}, 
                    cmap="Spectral")
                pl.camera_position = "xy"

                pl.subplot(2,i)
                pl.add_mesh(cell_pyv_mesh.copy(), scalars=lab+"_diff", 
                    lighting=False, 
                    scalar_bar_args={"title":f"DIFF_{lab}"}, 
                    cmap="Spectral")
                pl.camera_position = "xy"
            pl.link_views()
            if run_name is not None:
                if not os.path.isdir(self.conf["test_htmls_comparisons"]):
                    os.mkdir(self.conf["test_htmls_comparisons"])
                if not os.path.isdir(self.conf["test_imgs_comparisons"]):
                    os.mkdir(self.conf["test_imgs_comparisons"])

                if not os.path.isdir(os.path.join(self.conf["test_htmls_comparisons"], run_name)):
                    os.mkdir(os.path.join(self.conf["test_htmls_comparisons"], run_name))
                if not os.path.isdir(os.path.join(self.conf["test_imgs_comparisons"], run_name)):
                    os.mkdir(os.path.join(self.conf["test_imgs_comparisons"], run_name))

                pl.camera.zoom(1.6)
                pl.export_html(os.path.join(self.conf["test_htmls_comparisons"], run_name, self.name+"_cell.html"))
                pl.screenshot(
                    filename=os.path.join(self.conf["test_imgs_comparisons"], run_name, self.name+"_cell.png"),
                    window_size=(1920,1200))
            # pl.enable_anti_aliasing() # BREAKS EVERYTHING do NOT use
            else:
                pl.show()

            pl = pyvista.Plotter(shape=(3, len(columns)), off_screen=off_screen)
            for i, lab in enumerate(columns):
                pl.subplot(0,i)
                pl.add_mesh(face_pyv_mesh.copy(), scalars=lab, 
                    lighting=False, 
                    scalar_bar_args={"title":f"GT_{lab}"},
                    cmap="Spectral")
                pl.camera_position = "xy"

                pl.subplot(1,i)
                pl.add_mesh(face_pyv_mesh.copy(), scalars=lab+"_pred", 
                    lighting=False, 
                    scalar_bar_args={"title":f"PRED_{lab}"}, 
                    cmap="Spectral")
                pl.camera_position = "xy"

                pl.subplot(2,i)
                pl.add_mesh(face_pyv_mesh.copy(), scalars=lab+"_diff", 
                    lighting=False, 
                    scalar_bar_args={"title":f"DIFF_{lab}"}, 
                    cmap="Spectral")
                pl.camera_position = "xy"
            pl.link_views()
            if run_name is not None:
                pl.camera.zoom(1.6)
                pl.export_html(os.path.join(self.conf["test_htmls_comparisons"], run_name, self.name+"_face.html"))
                pl.screenshot(
                    filename=os.path.join(self.conf["test_imgs_comparisons"], run_name, self.name+"_face.png"),
                    window_size=(1920,1200))
            # pl.enable_anti_aliasing() # BREAKS EVERYTHING do NOT use
            else:
                pl.show()
            print("done")

        else:
            for tup in what_to_plot:
                match tup[0]:
                    case "vertex":
                        assert self.vertex_labels is not None, "You did not use 'add_labels' with mode='vertex' on this mesh, no labels present for vertices"
                        assert tup[1] == "label", "Points do not have features (they only have LABELS)"
                        vertex_pyv_mesh.point_data[tup[2]] = self.vertex_labels[tup[2]]
                        
                        pl = pyvista.Plotter()
                        pl.add_mesh(vertex_pyv_mesh, scalars=tup[2], lighting=False,
                                    scalar_bar_args={"title":tup[2]}, cmap="Spectral")
                        pl.camera_position = "xy"
                        pl.enable_anti_aliasing()
                        pl.show()

                    case "face":
                        if tup[1] == "label":
                            assert self.face_center_labels is not None, "You did not use 'add_labels' with mode='element' on this mesh, no labels present for faces or cells"
                            face_pyv_mesh.cell_data[tup[2]] = self.face_center_labels[tup[2]]
                        elif tup[1] == "feature":
                            face_pyv_mesh.cell_data[tup[2]] = self.face_center_features[:,self.conf["graph_node_feature_dict"][tup[2]]]
                        else:
                            raise ValueError(f"tup[1] can be only 'label' or 'feature', you wrote {tup[1]}")
                        
                        pl = pyvista.Plotter()
                        pl.add_mesh(face_pyv_mesh, scalars=tup[2], lighting=False,
                                    scalar_bar_args={"title":tup[2]}, cmap="Spectral")
                        pl.camera_position = "xy"
                        pl.enable_anti_aliasing()
                        pl.show()

                    case "cell":
                        
                        assert tup[1] == "label", "Cells do not have features (they only have LABELS)"
                        assert self.face_center_labels is not None, "You did not use 'add_labels' with mode='element' on this mesh, no labels present for faces or cells"
                        
                        if not tup[2] == "streamlines":
                            # TODO: implement velocity magnitude if needed
                            cell_pyv_mesh.cell_data[tup[2]] = self.cell_center_labels[tup[2]]
                            
                            pl = pyvista.Plotter()
                            pl.add_mesh(cell_pyv_mesh, scalars=tup[2], lighting=False,
                                        scalar_bar_args={"title":tup[2]}, cmap="Spectral")
                            pl.camera_position = "xy"
                            pl.enable_anti_aliasing()
                            pl.show()
                        else:
                            raise NotImplementedError("Streamlines still not working")
                            velocity = np.concatenate([
                                self.cell_center_labels[self.conf.active_vectors_2d].to_numpy(),
                                np.zeros([len(self.cell_center_labels),1])], 
                            axis=1)
                            cell_pyv_mesh.cell_data["velocity"] = velocity
                            cell_pyv_mesh.set_active_vectors("velocity", preference="cell")

                            lines = cell_pyv_mesh.streamlines_from_source(
                                source=pyvista.PointSet(self.inlet_points_positions),
                                vectors="velocity",
                            )

                            pl = pyvista.Plotter()
                            pl.add_mesh(
                                lines,
                                render_lines_as_tubes=True,
                                line_width=5,
                                lighting=False,
                            )
                            pl.add_mesh(
                                cell_pyv_mesh, 
                                scalars="velocity", 
                                lighting=False,
                                cmap="Spectral", 
                                opacity=0.3
                            )
                            pl.camera_position = "xy"
                            pl.enable_anti_aliasing()
                            pl.show()
                    case _:
                        raise NotImplementedError()
                            


    def plot_all_results(self, data, model_output, run_name, conf):
        # if self.vertex_labels is not None:
        #     vertex_pyv_mesh = toughio.from_meshio(self.mesh).to_pyvista()
        labels = model_output[0]

        face_list_for_pyvista = np.concatenate(
                [[len(v), *v] for v in self.vertices_in_faces]
            )
        facetype_list_for_pyvista = [pyvista.CellType.LINE for _ in self.vertices_in_faces]

        face_pyv_mesh = pyvista.UnstructuredGrid(
                            face_list_for_pyvista,
                            facetype_list_for_pyvista,
                            self.mesh.points
                        )
        
        cell_list_for_pyvista = np.concatenate(
                    [[len(v), *v] for v in self.vertices_in_cells]
                )

        celltype_list_for_pyvista = [pyvista.CellType.POLYGON for _ in self.vertices_in_cells]

        cell_pyv_mesh = pyvista.UnstructuredGrid(
            cell_list_for_pyvista,
            celltype_list_for_pyvista,
            self.mesh.points,
        )

        self.faces_in_cells = pd.DataFrame(data=self.CcFc_edges, columns=["cell_idx", "face_idx"]).groupby("cell_idx")["face_idx"].apply(list)
        # TODO: improve from mean to weighted avg depending on distance
        self.cell_center_labels = pd.DataFrame(
            [np.mean(self.face_center_labels.iloc[faces_idx], axis=0) for faces_idx in self.faces_in_cells]
        )

        ### comparison labels
        columns = conf["labels_to_keep_for_training"]
        face_pred_labels = pd.DataFrame(labels, columns=columns)
        cell_pred_lables = pd.DataFrame(
            [np.mean(face_pred_labels.iloc[faces_idx], axis=0) for faces_idx in self.faces_in_cells]
        )

        for i, lab in enumerate(columns):
            cell_pyv_mesh.cell_data[lab] = self.cell_center_labels[lab]
            cell_pyv_mesh.cell_data[lab+"_pred"] = cell_pred_lables[lab]
            cell_pyv_mesh.cell_data[lab+"_diff"] = self.cell_center_labels[lab] - cell_pred_lables[lab]
            face_pyv_mesh.cell_data[lab] = self.face_center_labels[lab]
            face_pyv_mesh.cell_data[lab+"_pred"] = face_pred_labels[lab]
            face_pyv_mesh.cell_data[lab+"_diff"] = self.face_center_labels[lab] - face_pred_labels[lab]

        off_screen = True
        pl = pyvista.Plotter(shape=(3, len(columns)), off_screen=off_screen)
        for i, lab in enumerate(columns):
            pl.subplot(0,i)
            pl.add_mesh(cell_pyv_mesh.copy(), scalars=lab, 
                lighting=False, 
                scalar_bar_args={"title":f"GT_{lab}"},
                cmap="Spectral")
            pl.camera_position = "xy"

            pl.subplot(1,i)
            pl.add_mesh(cell_pyv_mesh.copy(), scalars=lab+"_pred", 
                lighting=False, 
                scalar_bar_args={"title":f"PRED_{lab}"}, 
                cmap="Spectral")
            pl.camera_position = "xy"

            pl.subplot(2,i)
            pl.add_mesh(cell_pyv_mesh.copy(), scalars=lab+"_diff", 
                lighting=False, 
                scalar_bar_args={"title":f"DIFF_{lab}"}, 
                cmap="Spectral")
            pl.camera_position = "xy"
        pl.link_views()
        if not os.path.isdir(self.conf["test_htmls_comparisons"]):
            os.mkdir(self.conf["test_htmls_comparisons"])
        if not os.path.isdir(self.conf["test_imgs_comparisons"]):
            os.mkdir(self.conf["test_imgs_comparisons"])

        if not os.path.isdir(os.path.join(self.conf["test_htmls_comparisons"], run_name)):
            os.mkdir(os.path.join(self.conf["test_htmls_comparisons"], run_name))
        if not os.path.isdir(os.path.join(self.conf["test_imgs_comparisons"], run_name)):
            os.mkdir(os.path.join(self.conf["test_imgs_comparisons"], run_name))

        pl.camera.zoom(1.6)
        pl.export_html(os.path.join(self.conf["test_htmls_comparisons"], run_name, self.name+"_cell.html"))
        pl.screenshot(
            filename=os.path.join(self.conf["test_imgs_comparisons"], run_name, self.name+"_cell.png"),
            window_size=(1920,1200))

        pl = pyvista.Plotter(shape=(3, len(columns)), off_screen=off_screen)
        for i, lab in enumerate(columns):
            pl.subplot(0,i)
            pl.add_mesh(face_pyv_mesh.copy(), scalars=lab, 
                lighting=False, 
                scalar_bar_args={"title":f"GT_{lab}"},
                cmap="Spectral")
            pl.camera_position = "xy"

            pl.subplot(1,i)
            pl.add_mesh(face_pyv_mesh.copy(), scalars=lab+"_pred", 
                lighting=False, 
                scalar_bar_args={"title":f"PRED_{lab}"}, 
                cmap="Spectral")
            pl.camera_position = "xy"

            pl.subplot(2,i)
            pl.add_mesh(face_pyv_mesh.copy(), scalars=lab+"_diff", 
                lighting=False, 
                scalar_bar_args={"title":f"DIFF_{lab}"}, 
                cmap="Spectral")
            pl.camera_position = "xy"
        pl.link_views()
        pl.camera.zoom(1.6)
        pl.export_html(os.path.join(self.conf["test_htmls_comparisons"], run_name, self.name+"_face.html"))
        pl.screenshot(
            filename=os.path.join(self.conf["test_imgs_comparisons"], run_name, self.name+"_face.png"),
            window_size=(1920,1200))

        ### residuals on boundaries
        flag_boundaries = False
        idxs_is_BC = (data.x_mask[:,-1] == 1)
        
        for k, v in model_output[1].items():
            v = v.detach().numpy()
            if v.shape[0] == idxs_is_BC.sum():
                new_k = k.removeprefix("debug_only_")
                tmp = np.zeros_like(face_pyv_mesh.cell_data["pressure"])
                tmp[idxs_is_BC] = v
                face_pyv_mesh.cell_data[new_k] = tmp
                flag_boundaries = True

        if flag_boundaries:
            cmap = "Greys"
            pl = pyvista.Plotter(shape=(2, 3), off_screen=off_screen)
            pl.subplot(0,0)
            pl.add_mesh(face_pyv_mesh.copy(), lighting=False, cmap=cmap, 
                        scalars="boundary", scalar_bar_args={"title":"Total BC residual"},)
            pl.camera_position = "xy"

            pl.subplot(0,1)
            pl.add_mesh(face_pyv_mesh.copy(), lighting=False, cmap=cmap, 
                        scalars="BC_v_t", scalar_bar_args={"title":"Component: v_t"},)
            pl.camera_position = "xy"

            pl.subplot(0,2)
            pl.add_mesh(face_pyv_mesh.copy(), lighting=False, cmap=cmap, 
                        scalars="BC_v_n", scalar_bar_args={"title":"Component: v_n"},)
            pl.camera_position = "xy"

            pl.subplot(1,0)
            pl.add_mesh(face_pyv_mesh.copy(), lighting=False, cmap=cmap, 
                        scalars="BC_p", scalar_bar_args={"title":"Component: p"},)
            pl.camera_position = "xy"

            pl.subplot(1,1)
            pl.add_mesh(face_pyv_mesh.copy(), lighting=False, cmap=cmap, 
                        scalars="BC_dv_dn", scalar_bar_args={"title":"Component: dv_dn"},)
            pl.camera_position = "xy"

            pl.subplot(1,2)
            pl.add_mesh(face_pyv_mesh.copy(), lighting=False, cmap=cmap, 
                        scalars="BC_dp_dn", scalar_bar_args={"title":"Component: dp_dn"},)
            pl.camera_position = "xy"
            pl.link_views()

            pl.camera.zoom(1.6)
            pl.export_html(os.path.join(self.conf["test_htmls_comparisons"], run_name, self.name+"_boundaries.html"))
            pl.screenshot(
                filename=os.path.join(self.conf["test_imgs_comparisons"], run_name, self.name+"_boundaries.png"),
                window_size=(1920,1200))
        
        ### residuals on domain 
        # # change this
        cell_shape = cell_pyv_mesh.cell_data["pressure"].shape[0]
        tmp_idx = data.idx_of_triangulated_cell.detach().numpy()
        present_residuals_domain = []
        for k, v in model_output[1].items():
            v = np.abs(v.detach().numpy())
            if v.shape[0] == tmp_idx.shape[0] and len(v.shape)==1:
                tmp = [np.mean(v[tmp_idx[i]]) for i in range(cell_shape)]
                cell_pyv_mesh.cell_data[k] = tmp
                present_residuals_domain.append(k)
        
        if len(present_residuals_domain) > 0:
            cmap = "Greys"
            pl = pyvista.Plotter(shape=(1, len(present_residuals_domain)), off_screen=off_screen)
            
            for i, k in enumerate(present_residuals_domain):
                pl.subplot(0,i)
                pl.add_mesh(cell_pyv_mesh.copy(), lighting=False, cmap=cmap, 
                            scalars=k, scalar_bar_args={"title":f"Domain residual: {k}"},)
                pl.camera_position = "xy"

            pl.link_views()
            pl.camera.zoom(1.6)
            pl.export_html(os.path.join(self.conf["test_htmls_comparisons"], run_name, self.name+"_domain_res.html"))
            pl.screenshot(
                filename=os.path.join(self.conf["test_imgs_comparisons"], run_name, self.name+"_domain_res.png"),
                window_size=(1920,1200))

def print_memory_state_gpu(text, conf):
    if conf.device == "cuda":
        print(f"{text} - Alloc: {torch.cuda.memory_allocated()/1024**3:.2f}Gb - Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}Gb")


def get_input_to_model(batch):
    '''Used because model summary doesn't work if you give in input arbitrary objects'''
    return {
        "x": batch.x,
        "x_mask": batch.x_mask,
        "x_additional": batch.x_additional,
        "edge_index": batch.edge_index,
        "edge_attr": batch.edge_attr,
        "batch": batch.batch,
        "pos": batch.pos,
        "domain_sampling_points": getattr(batch, "domain_sampling_points", None),
        "boundary_sampling_points": getattr(batch, "boundary_sampling_points", None),
        "index_boundary_sampled": getattr(batch, "index_boundary_sampled", None),
        "new_edges_index": getattr(batch, "new_edges_index", None),
        "new_edge_attributes": getattr(batch, "new_edge_attributes", None),
        "x_additional_boundary": getattr(batch, "x_additional_boundary", None),
        "num_domain_sampling_points": getattr(batch, "num_domain_sampling_points", None),
        "num_boundary_sampling_points": getattr(batch, "num_boundary_sampling_points", None),
    }


def plot_continuity(pos: torch.Tensor, res: torch.Tensor):
    points = np.stack([
        pos[:,0].detach().cpu().numpy(), 
        pos[:,1].detach().cpu().numpy(), 
        res["continuity"].detach().cpu().numpy()]).T
    
    range_x = points[:,0].max() - points[:,0].min()
    range_y = points[:,1].max() - points[:,1].min()
    range_z = points[:,2].max() - points[:,2].min()

    points[:,2] = points[:,2]/range_z*max(range_x, range_y)
    pointcloud = pyvista.PolyData(points)
    pointcloud.plot()

def get_maximum_difference(labels_of_faces_in_cell):
    sorted_labels = labels_of_faces_in_cell.sort(dim=0)[0]
    return sorted_labels[-1,:] - sorted_labels[0,:]

def convert_mesh_complete_info_obj_to_graph(
        conf:Config,
        meshCI: MeshCompleteInfo, # MeshCompleteInfo object
        complex_graph=False,
        filename_output_graph=None, 
        ):
    
    '''Given a MeshCompleteInfo instance, returns a graph and saves it to memory'''
    if filename_output_graph == None:
        print("Warning: no output location specified, graph will NOT be saved to disk")

    if not complex_graph:
        if conf.dim == 2:
            graph_nodes_positions = meshCI.face_center_positions # not used because not relative

            graph_node_attr = normalize_features(meshCI.face_center_features, conf)
            graph_node_attr_mask = meshCI.face_center_features_mask
            graph_node_additional_attr = meshCI.face_center_additional_features

            FcFc_edges_bidir = np.concatenate([meshCI.FcFc_edges, np.flip(meshCI.FcFc_edges, axis=1)], axis=0)
            graph_edges = FcFc_edges_bidir

            graph_edge_relative_displacement_vector = np.array([graph_nodes_positions[p2]-graph_nodes_positions[p1] for (p1, p2) in graph_edges])
            graph_edge_norm = np.expand_dims(np.linalg.norm(graph_edge_relative_displacement_vector, axis=1), axis=1)
            graph_edge_attr = np.concatenate([graph_edge_relative_displacement_vector, graph_edge_norm], axis=1)

            data = Data(
                edge_index=torch.tensor(graph_edges).t().contiguous(), 
                pos=torch.tensor(graph_nodes_positions, dtype=torch.float32),
                edge_attr=torch.tensor(graph_edge_attr, dtype=torch.float32),
                x=torch.tensor(graph_node_attr, dtype=torch.float32),
                y=None
            )
            data.name = meshCI.name
            data.meshComplete_corresponding_path = meshCI.path
            
            data.x_mask = torch.tensor(graph_node_attr_mask, dtype=torch.bool)
            data.x_additional = torch.Tensor(graph_node_additional_attr)

            data.n_faces = torch.tensor(meshCI.face_center_positions.shape[0])
            data.n_face_edges = torch.tensor(len(FcFc_edges_bidir))
            face_label_dim = len(conf.features_to_keep)

            tmp = meshCI.face_center_labels
            data.y = torch.tensor(tmp[conf.labels_to_keep_for_training].values, dtype=torch.float32)
            remaining_columns = sorted(set(conf.features_to_keep).difference(conf.labels_to_keep_for_training))
            data.y_additional = torch.tensor(tmp[remaining_columns].values, dtype=torch.float32)
            # data.y_mask = torch.tensor(np.ones([n_faces, face_label_dim]), dtype=torch.bool)
            # data.turbolence = torch.tensor(tmp[conf.labels_to_keep_for_training_turbulence].values, dtype=torch.float32)

            data.inward_normal_areas = torch.zeros((data.x.shape[0], 2))

            surface_faces = data.x_additional[:, conf.graph_node_features_not_for_training["is_car"]].nonzero()[:,0]
            data.inward_normal_areas[surface_faces, :] = get_inward_normal_areas(
                faces_idxs=surface_faces,
                face_areas=data.x[:, conf.graph_node_feature_dict["face_area"]],
                faces_x_component=data.x[:, conf.graph_node_feature_dict["tangent_versor_x"]],
                faces_y_component=data.x[:, conf.graph_node_feature_dict["tangent_versor_y"]],
                CcFc_edges=meshCI.CcFc_edges,
                cell_center_positions=meshCI.cell_center_positions,
                face_center_positions=meshCI.face_center_positions,)

            data.force_on_component = get_forces(conf, data, pressure_values=data.y[:,2],
                velocity_derivatives=data.y_additional[:,2:], turbulent_values=data.y[:,3:])
            data.CcFc_edges = torch.tensor(meshCI.CcFc_edges) # useful for sampling inside cells
            
            tmp1, tmp2 = meshCI.get_triangulated_cells()
            data.triangulated_cells, data.idx_of_triangulated_cell = torch.tensor(tmp1), torch.tensor(tmp2)
            
            tmp = [data.CcFc_edges[data.CcFc_edges[:,0]==i,1] for i in range(data.CcFc_edges[:,0].max()+1)]
            data.faces_in_cell = torch.nn.utils.rnn.pad_sequence(
                tmp, 
                padding_value=-1
            )
            data.len_faces = torch.tensor([len(f) for f in tmp])

            sampling_weights = torch.stack(
                [get_maximum_difference(data.y[faces_in_cells]) for faces_in_cells in tmp])
            
            data.sampling_weights = sampling_weights # TODO: some cells could be "flat" --> do we wand to add an epsilon?

        else:
            raise NotImplementedError("Implement dim = 3")
    else:
        raise NotImplementedError("Still many things to do")
        if conf.dim == 2:
            graph_nodes_positions = np.concatenate([meshCI.cell_center_positions, meshCI.face_center_positions])

            # shift face indices
            FcFc_edges = meshCI.FcFc_edges
            CcFc_edges = meshCI.CcFc_edges
            CcCc_edges_bidir = meshCI.CcCc_edges_bidir
            n_cells = meshCI.cell_center_positions.shape[0]
            n_faces = meshCI.face_center_positions.shape[0]

            FcFc_edges[:,:] += n_cells
            CcFc_edges[:,1] += n_cells # first indices refer to cell centers and so do not need to be shifted

            # Add graph_edge bidirectionality
            FcFc_edges_bidir = np.concatenate([FcFc_edges, np.flip(FcFc_edges, axis=1)], axis=0)
            CcFc_edges_bidir = np.concatenate([CcFc_edges, np.flip(CcFc_edges, axis=1)], axis=0)

            graph_edges = np.concatenate([CcCc_edges_bidir, FcFc_edges_bidir, CcFc_edges_bidir])

            graph_edge_attr = np.concatenate([np.ones((len(CcCc_edges_bidir), 1)) * conf.edge_type_feature["cell_cell"],
                                            np.ones((len(FcFc_edges_bidir), 1)) * conf.edge_type_feature["face_face"],
                                            np.ones((len(CcFc_edges_bidir), 1)) * conf.edge_type_feature["cell_face"]])

            graph_node_attr = np.concatenate([np.zeros([n_cells, len(conf.graph_node_feature_dict)]),
                                            meshCI.face_center_features])

            graph_node_attr_mask = np.concatenate([np.zeros([n_cells, len(conf.graph_node_feature_dict)]),
                                                meshCI.face_center_ord_features_mask])
            data = Data(
                edge_index=torch.tensor(graph_edges).t().contiguous(), 
                pos=torch.tensor(graph_nodes_positions, dtype=torch.float32),
                edge_attr=torch.tensor(graph_edge_attr, dtype=torch.float32),
                x=torch.tensor(graph_node_attr, dtype=torch.float32),
                y=None
            )
            
            data.x_mask = torch.tensor(graph_node_attr_mask, dtype=torch.bool)

            data.n_cells = torch.tensor(n_cells)
            data.n_faces = torch.tensor(n_faces)

            data.n_cell_edges = torch.tensor(len(CcCc_edges_bidir))
            data.n_face_edges = torch.tensor(len(FcFc_edges_bidir))
            data.n_cell_face_edges = torch.tensor(len(CcFc_edges_bidir))

            face_label_dim = len(conf.features_to_keep)

            data.y = torch.tensor(np.concatenate([np.zeros([n_cells, face_label_dim]), 
                                        meshCI.face_center_labels]), dtype=torch.float32)
    
            data.y_mask = torch.tensor(np.concatenate([np.zeros([n_cells, face_label_dim]), 
                                                    np.ones([n_faces, face_label_dim])]), dtype=torch.bool)
            
        else:
            raise NotImplementedError("Implement dim = 3")


    if filename_output_graph:
        torch.save(data, filename_output_graph)

    return data


def plot_test_images_from_model(conf, model, run_name, test_dataloader):

    for batch in tqdm(test_dataloader):
        y = model(**get_input_to_model(batch))

        # if isinstance(pred_batch, tuple):
        #     residuals = pred_batch[1]
        #     pred_batch = pred_batch[0]

        for i in range(len(batch)):
            data = batch[i]
            assert batch.ptr.shape[0]==2, "Can only print if batch is one"
            plot_gt_pred_label_comparison(data, y, conf, run_name=os.path.basename(run_name))
