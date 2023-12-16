import os
from typing import Literal, Optional

import meshio
import pyvista
import toughio
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from icecream import ic
import torch
from torch_geometric.data import Data
import pandas as pd

from config_pckg.config_file import Config


def read_mesh(filename, mode:Literal["meshio", "pyvista", "toughio"], plot=True):
    '''Reads mesh given mode'''
    # mesh = meshio.read(filename)
    match mode:
        case "meshio":
            return meshio.ansys.read(filename)
        case "pyvista":
            # Useful to plot the mesh but not useful for mesh manipulation
            # See: https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.UnstructuredGrid.html#pyvista.UnstructuredGrid
            mesh = pyvista.read(filename)
            if plot:
                mesh.plot()
            return mesh
        case "toughio":
            # Parent classes to meshio Mesh type with (it seems) useful utilities
            # https://toughio.readthedocs.io/en/latest/mesh.html
            return toughio.read_mesh(filename)


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


def match_mesh_and_feature_pts(mesh, features, check_biunivocity=True):
    eps = 1e-5
    feature_to_mesh_scale = 1000
    map_mesh_to_feature = np.zeros([len(mesh.points), 2])
    if mesh.points[0].shape[0] == 2:
        mesh_m, mesh_old_idxs = sort_matrix(mesh.points)
        features_m, features_old_idxs = sort_matrix(features[["    x-coordinate", "    y-coordinate"]].to_numpy())
        
        mesh_abs = np.abs(mesh_m)
        features_low_bound = np.abs(features_m*feature_to_mesh_scale*(1-eps))
        features_high_bound = np.abs(features_m*feature_to_mesh_scale*(1+eps))

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
    return np.concatenate(list(map(get_edges_from_component, mesh.cells)))


def convert_msh_csv_to_graph(filename_input_msh, filename_input_csv, filename_output_graph):
    '''Given the ansys .msh file and the .csv feature file, saves in memory the complete graph from torch_geometric'''
    conf = Config()

    mesh = read_mesh(filename_input_msh, mode="meshio")
    features = pd.read_csv(filename_input_csv)

    map_mesh_to_feature = match_mesh_and_feature_pts(mesh, features)

    # TODO: insert this into Config()
    features_to_remove = ['nodenumber', '    x-coordinate', '    y-coordinate', 'boundary-normal-dist']
    mesh_features = features[features.columns.difference(features_to_remove)].iloc[map_mesh_to_feature]

    ##### To get full graph node-node connectivity
    edges = get_all_edges(mesh)

    data = Data(x=torch.tensor(mesh_features.to_numpy()),
                edge_index=torch.tensor(edges).t().contiguous(), 
                pos=torch.tensor(mesh.points))
    
    torch.save(data, filename_output_graph)

    return data