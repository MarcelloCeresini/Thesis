import os
import pickle
from typing import Literal, Optional, Union
import itertools

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
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
from torch_geometric.utils import to_dense_adj
from rustworkx import PyGraph
from rustworkx import distance_matrix

from config_pckg.config_file import Config
import read_mesh_meshio_forked
from mesh_exploration import plot_2d_cfd

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

    return cell_mesh.centers, CcCc_edges_bidir, vertices_in_cells


def get_face_data(face_mesh: meshio.Mesh, vertices_in_cells):

    vertices_in_faces = np.concatenate([c.data for c in face_mesh.cells], axis=0) # face_list[face_idx] = [list_of_node_idxs_in_that_face]
    tmp = set([frozenset([vertex_pair[0], vertex_pair[1]]) for vertex_pair in vertices_in_faces])
    vertices_in_faces = np.stack([np.array(list(fset)) for fset in tmp])

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
    
    return face_center_positions, FcFc_edges, vertices_in_faces, CcFc_edges


def get_face_BC_attributes(mesh: meshio.Mesh, face_center_positions, vertices_in_faces, conf: Config):
    point_positions = mesh.points

    face_center_attr_BC = np.zeros((len(face_center_positions), len(conf.graph_node_feature_dict)))
    face_center_attr_BC_mask = np.zeros((len(face_center_positions), len(conf.graph_node_feature_dict)+1)).astype(bool)

    # Add general "is_BC" mask
    face_center_attr_BC_mask[:,-1] = True # set all "is_BC?" masks = True and then set it to 0 in "interior" below

    face_spatial_dir = [point_positions[v[1]]-point_positions[v[0]] for v in vertices_in_faces]
    face_spatial_dir_norm = np.array([vec/np.linalg.norm(vec) for vec in face_spatial_dir])

    # face tangent versor components (x, y)
    face_center_attr_BC[:,:2] = face_spatial_dir_norm[:,:2]
    face_center_attr_BC_mask[:,:2] = True

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

        face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["component_id"]] = i
        face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_dict["component_id"]] = True
                                                        
        match bc_type:
            case 2: # interior, no condition
                face_center_attr_BC_mask[faces_of_faceblock_idxs, -1] = False # set "is_BC" masks = False
            case 3: # wall, speed fixed depending on the name
                face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_n"]] = 0
                face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_n"]] = True
                face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["dp_dn"]] = 0
                face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_dict["dp_dn"]] = True

                if "ground" in name: # same tangential velocity as the air entering the domain
                    direction = face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["tangent_versor_x"]]
                    if conf.flag_directional_BC_velocity:
                        face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_t"]] = conf.air_speed * np.sign(direction)
                    else:
                        face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_t"]] = conf.air_speed

                    face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_t"]] = True

                elif "tyre" in name: # in 2D, it rotates around the center with angular speed omega
                    Cx, Cy, R, sigma = hyperLSQ(points_of_faceblock_positions[:,:2])
                    omega = conf.air_speed / R
                    rays = face_center_positions[faces_of_faceblock_idxs][:,:2] - [Cx, Cy]
                    direction = np.cross(face_center_attr_BC[faces_of_faceblock_idxs,:2], rays)

                    if conf.flag_directional_BC_velocity:
                        v_t = np.linalg.norm(rays, axis=1) * omega * np.sign(direction) # v_t = r * omega
                    else:
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
                    face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_t"]] = True
                elif any(x in name for x in ["w0", "default-exterior"]): # fixed wall, doesn't move
                    face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_t"]] = 0
                    face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_t"]] = True
                else:
                    raise NotImplementedError(f"Didn't implement this kind of wall yet: {name}")
            case 5: # pressure-outlet
                face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["p"]] = conf.relative_atmosferic_pressure
                face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_dict["p"]] = True
                face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["dv_dn"]] = 0
                face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_dict["dv_dn"]] = True
            case 7: # simmetry, normal derivative = 0
                pass
                face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_n"]] = 0
                face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_n"]] = True
                
                face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["dp_dn"]] = 0
                face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_dict["dp_dn"]] = True
            case 10: # velocity_inlet
                inlet_points.append(points_of_faceblock_positions)
                # TODO: dp_dn = 0
                face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_t"]] = 0
                face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_t"]] = True
                face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_n"]] = conf.air_speed
                face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_dict["v_n"]] = True
                face_center_attr_BC[faces_of_faceblock_idxs, conf.graph_node_feature_dict["dp_dn"]] = 0
                face_center_attr_BC_mask[faces_of_faceblock_idxs, conf.graph_node_feature_dict["dp_dn"]] = True

            case _:
                raise NotImplementedError("Didn't implement this kind of BC yet")
            
    if len(inlet_points) >= 1:
        inlet_points = np.stack(inlet_points)[0]
    else:
        print("WARNING, no inlet points found")
    
    return face_center_attr_BC, face_center_attr_BC_mask, inlet_points
    

def get_labels(positions, csv_filename, conf, check_biunivocity):
    '''Returns a [len(positions), N_features] matrix'''
    eps = conf.epsilon_for_point_matching

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
                
        if check_biunivocity:
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

    cell_center_positions, CcCc_edges_bidir, vertices_in_cells = get_cell_data(mesh, conf)
    
    face_center_positions, FcFc_edges, vertices_in_faces, CcFc_edges = get_face_data(mesh, vertices_in_cells)
    
    mesh_complete_instance = MeshCompleteInfo(
        conf,
        filename_input_msh,
        mesh,
        cell_center_positions,
        CcCc_edges_bidir,
        vertices_in_cells,
        face_center_positions,
        FcFc_edges,
        vertices_in_faces,
        CcFc_edges
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


def normalize_labels(labels, conf_dict, conf):
    labels = labels.copy()
    # TODO: to use v_mag correctly, we should compute v_mag POINTWISE and then aggregate
    # instead here we compute mean of x-v and y-v for each graph, then compute v_mag as norm (graph-wise)
    v_mag = np.linalg.norm(labels[["x-velocity", "y-velocity"]], axis=1)

    if conf_dict["graph_wise"]:
        v_mag_mean, v_mag_std = v_mag.mean(), v_mag.std()
        vx_mean, vx_std = labels["x-velocity"].mean(), labels["x-velocity"].std()
        vy_mean, vy_std = labels["y-velocity"].mean(), labels["y-velocity"].std()
        p_mean, p_std = labels["pressure"].mean(), labels["pressure"].std()
    else: # else it is dataset_wise
        v_mag_mean, v_mag_std = conf.train_set_normalization_constants["v_mag_mean"], conf.train_set_normalization_constants["v_mag_std"]
        vx_mean, vx_std = conf.train_set_normalization_constants["vx_mean"], conf.train_set_normalization_constants["vx_std"]
        vy_mean, vy_std = conf.train_set_normalization_constants["vy_mean"], conf.train_set_normalization_constants["vy_std"]
        p_mean, p_std = conf.train_set_normalization_constants["p_mean"], conf.train_set_normalization_constants["p_std"]

    if conf_dict["no_shift"]:
        v_mag_mean = 0
        vx_mean = 0
        vy_mean = 0
        p_mean = 0

    match conf_dict["main"]:
        case "None":
            return labels
        case "Z-Normalization":
            labels["pressure"] = (labels["pressure"]-p_mean)/p_std
            match conf_dict["velocity_mode"]:
                case "component_wise":
                    labels["x-velocity"] = (labels["x-velocity"]-vx_mean)/vx_std
                    labels["y-velocity"] = (labels["y-velocity"]-vy_mean)/vy_std
                    return labels
                case "magnitude_wise":
                    v_mag_norm = (v_mag-v_mag_mean)/v_mag_std
                    labels["x-velocity"] = (labels["x-velocity"]/v_mag)*v_mag_norm    # inside parentheses it becomes "versor component", then you renormalize it with the product
                    labels["y-velocity"] = (labels["y-velocity"]/v_mag)*v_mag_norm
                case _:
                    raise NotImplementedError()
            return labels
        case "Physical":
            labels[["x-velocity", "y-velocity"]] /= conf.air_speed
            labels["pressure"] /= (conf.air_speed**2)/2
            return labels
        case _:
            raise NotImplementedError()


def denormalize_labels(labels, conf):
    '''
    Only implemented for physical normalization for now
    '''
    labels[:,:2] *= conf.air_speed          # x-velocity and y-velocity
    labels[:,2] *= (conf.air_speed**2)/2    # pressure
    return labels


def plot_gt_pred_label_comparison(data: Data, pred: torch.Tensor, conf, run_name: Optional[str]= None):
    with open(os.path.join(conf.EXTERNAL_FOLDER_MESHCOMPLETE_W_LABELS, data.name+".pkl"), "rb") as f:
        meshCI = pickle.load(f)
        
    print(f"Plotting {meshCI.name}")
    pred = denormalize_labels(pred, conf)
    meshCI.set_conf(conf)
    meshCI.plot_mesh(labels=pred, run_name = run_name)


def convert_mesh_complete_info_obj_to_graph(
        conf:Config,
        meshCI, # MeshCompleteInfo object
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
            graph_node_attr_mask = meshCI.face_center_ord_features_mask

            FcFc_edges_bidir = np.concatenate([meshCI.FcFc_edges, np.flip(meshCI.FcFc_edges, axis=1)], axis=0)
            graph_edges = FcFc_edges_bidir
            
            features_to_remove = set(conf.graph_node_feature_dict.keys()).difference(set(conf.graph_node_final_features))
            idxs_to_remove = [conf.graph_node_feature_dict[f] for f in features_to_remove]
            idxs_to_keep = []
            for idx in range(graph_node_attr.shape[-1]):
                if idx not in idxs_to_remove:
                    idxs_to_keep.append(idx)

            graph_node_attr = graph_node_attr[:, np.array(idxs_to_keep)]
            idxs_to_keep = np.concatenate((idxs_to_keep, np.expand_dims(graph_node_attr_mask.shape[1]-1, axis=0)))
            graph_node_attr_mask = graph_node_attr_mask[:, np.array(idxs_to_keep)]

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

            data.n_faces = torch.tensor(meshCI.face_center_positions.shape[0])
            data.n_face_edges = torch.tensor(len(FcFc_edges_bidir))
            face_label_dim = len(conf.features_to_keep)

            tmp = normalize_labels(meshCI.face_center_labels, conf.label_normalization_mode, conf)
            data.y = torch.tensor(tmp[conf.labels_to_keep_for_training].values, dtype=torch.float32)
            # data.y_mask = torch.tensor(np.ones([n_faces, face_label_dim]), dtype=torch.bool)

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

class MeshCompleteInfo:
    def __init__(
            self,
            conf: Config,
            path: str,
            mesh: meshio.Mesh,
            cell_center_positions,
            CcCc_edges_bidir,
            vertices_in_cells,
            face_center_positions, 
            FcFc_edges, 
            vertices_in_faces,
            CcFc_edges,
    ) -> None:
        self.conf = conf
        self.path = path
        self.name = path.split(os.sep)[-1].removesuffix(".pkl")
        self.group = 1 if "2dtc_001R" in self.name else 2
        self.mesh = mesh
        self.cell_center_positions = cell_center_positions
        self.CcCc_edges_bidir = CcCc_edges_bidir
        self.vertices_in_cells = vertices_in_cells
        self.face_center_positions = face_center_positions 
        self.FcFc_edges = FcFc_edges 
        self.vertices_in_faces = vertices_in_faces
        self.CcFc_edges = CcFc_edges

        self.face_center_features, self.face_center_ord_features_mask, self.inlet_points_positions = \
            get_face_BC_attributes(mesh, face_center_positions, vertices_in_faces, conf)
        
        self.dist_from_BC = None
        self.vertex_labels = None
        self.face_center_labels = None
        self.cell_center_labels = None

        self.radial_attributes = self.get_radial_attributes()


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


    def get_min_distance_from_BC(self, mode: Literal["vertex", "face", "cell"]="face"):
        if not mode=="face":
            raise NotImplementedError("Only implemented for 'face' for now")

        is_BC = self.face_center_ord_features_mask[-1]
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
        n_points = self.face_center_positions.shape[0]
        radial_attributes = np.zeros((n_points, self.conf.n_theta_bins))

        th_bin_size = 2*np.pi/(self.conf.n_theta_bins+1)

        BC_positions = self.face_center_positions[self.face_center_ord_features_mask[-1]==True]
        vectors_node_to_BC = cdist(self.face_center_positions, BC_positions, lambda x,y: y-x) # 3-dimensional

        for i in range(n_points):
            thetas = np.arctan2(vectors_node_to_BC[i,:,1], vectors_node_to_BC[i,:,0]) # y, x
            for j in range(self.conf.n_theta_bins):
                th_min, th_max = -np.pi + j*th_bin_size, -np.pi + (j+1)*th_bin_size

                idxs = (th_min <= thetas <= th_max)
                if idxs.shape[0] > 0:
                    vectors_inside_bin = vectors_node_to_BC[i, idxs, :]
                    vec_norms = map(lambda x: np.linalg.norm(x), vectors_inside_bin) 
                    
                    radial_attributes[j, :] = (
                        np.min(vec_norms), 
                        np.max(vec_norms)
                        )
                else:
                    radial_attributes[j, :] = (self.conf.default_radial_attribute_value, self.conf.default_radial_attribute_value)

        return radial_attributes


    def plot_mesh(self, 
                what_to_plot = None,
                labels: Optional[torch.Tensor] = None,
                run_name: Optional[str] = None,
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
        assert self.conf.dim == 2, "Implement dim = 3"
        assert (what_to_plot is not None) or (labels is not None), "Nothing to plot specified"
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

        if labels is not None:
            # columns = self.conf.labels_to_keep_for_training # TODO: update all MeshComplete files from scratch
            columns = ['x-velocity', 'y-velocity', 'pressure']
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
            pl = pyvista.Plotter(shape=(3, 3), off_screen=off_screen)
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
                if not os.path.isdir(os.path.join(self.conf.test_htmls_comparisons, run_name)):
                    os.mkdir(os.path.join(self.conf.test_htmls_comparisons, run_name))
                if not os.path.isdir(os.path.join(self.conf.test_imgs_comparisons, run_name)):
                    os.mkdir(os.path.join(self.conf.test_imgs_comparisons, run_name))
                pl.camera.zoom(1.6)
                pl.export_html(os.path.join(self.conf.test_htmls_comparisons, run_name, self.name+"_cell.html"))
                pl.screenshot(
                    filename=os.path.join(self.conf.test_imgs_comparisons, run_name, self.name+"_cell.png"),
                    window_size=(1920,1200))
            # pl.enable_anti_aliasing() # BREAKS EVERYTHING do NOT use
            else:
                pl.show()

            pl = pyvista.Plotter(shape=(3, 3), off_screen=off_screen)
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
                pl.export_html(os.path.join(self.conf.test_htmls_comparisons, run_name, self.name+"_face.html"))
                pl.screenshot(
                    filename=os.path.join(self.conf.test_imgs_comparisons, run_name, self.name+"_face.png"),
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
                            face_pyv_mesh.cell_data[tup[2]] = self.face_center_features[:,self.conf.graph_node_feature_dict[tup[2]]]
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


def print_memory_state_gpu(text, conf):
    if conf.device == "cuda":
        print(f"{text} - Alloc: {torch.cuda.memory_allocated()/1024**3:.2f}Gb - Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}Gb")


def get_input_to_model(batch):
    '''Used because model summary doesn't work if you give in input arbitrary objects'''
    return {
        "x": batch.x,
        "x_mask": batch.x_mask,
        "edge_index": batch.edge_index,
        "edge_attr": batch.edge_attr,
        "batch": batch.batch,
        "pos": batch.pos,
    }