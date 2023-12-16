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
# from submodules import pymesh
import utils

def get_mesh_component(mesh, mesh_component_idx):
    '''Returns a new mesh with only one of the "cells" components of the old mesh'''
    return toughio.Mesh(
        points=mesh.points, 
        cells=[mesh.cells[mesh_component_idx]]
    )


def subplot_mesh_components(mesh, idxs_cellblocks = [], debug=False):
    '''Plots in a row all specified cell components. If not specified, plots all'''
    if len(idxs_cellblocks) == 0:
        idxs_cellblocks = range(len(mesh.cells))
    
    pl = pyvista.Plotter(shape=(1, len(idxs_cellblocks)))

    c = 0
    for i in idxs_cellblocks:
        pl.subplot(0,c)
        c+=1

        mesh_part = get_mesh_component(mesh, i)
        if debug:
            ic(i, mesh_part.cells[0].data.shape)
        mesh_part = mesh_part.to_pyvista()
        actor = pl.add_mesh(mesh_part)

    pl.show()


def plot_pts(pts):
    '''Plot pointcloud indipendently of dimension'''
    fig = plt.figure()
    if len(pts.shape) == 2:
        ax = fig.add_subplot()
        ax.scatter(pts[:,0], pts[:,1], alpha=0.2)
    elif len(pts.shape) == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], alpha=0.2)
    plt.show()


def not_of_box(pt):
    '''Used to get points NOT near/on the simulation box for this specific example'''
    if -5<pt[0]<5 and 0<pt[1]<10 and 0.01<pt[2]<10:
        return True
    else:
        return False


def get_free_points(mesh: Optional[meshio.Mesh | toughio.Mesh], p=1, plot=False, debug=False):
    '''
    Return points in input mesh not present in any cellblock

    Inputs:
        - p: percentage of free points to return (to alleviate computation)
        - plot: if we want to plot the free points
    '''
    pts = mesh.points
    p = max(1, min(0, p))

    indexes_in_faces = np.unique(np.concatenate(
        list(map(lambda x: np.unique(x.data), mesh.cells))
    ))

    num_points = pts.shape[0]
    num_points_in_faces = indexes_in_faces.shape[0]

    if debug:
        ic(num_points)
        ic(num_points_in_faces)

    if num_points == num_points_in_faces:
        ic("No free air points, all points are connected in the mesh")
        return

    pts = np.delete(pts, indexes_in_faces, axis=0)

    if p != 1:
        pts = pts[np.random.choice(pts.shape[0], int(len(pts)*p), replace=False), :]

    if plot:
        plot_pts(pts)

    return pts


def get_idxs_in_compontents(mesh: Optional[toughio.Mesh | meshio.Mesh]):
    '''Returns the unique indices of all points inside each cellblock'''
    return list(map(lambda x: set(np.unique(x.data)), mesh.cells))


def get_intersection_matrix_components(indexes_in_faces):
    '''Returns, for each component, the indices of the points that intersect with all other components'''
    intersections_matrix = []
    for i in range(len(mesh.cells)):
        intersections_for_cb1 = []
        idxs_cb1 = indexes_in_faces[i]
        for j in range(i+1, len(mesh.cells)):
            idxs_cb2 = indexes_in_faces[j]
            intersections_for_cb1.append(list(idxs_cb1.intersection(idxs_cb2)))
        intersections_matrix.append(intersections_for_cb1)
    return intersections_matrix


def get_intersection_two_compontents(intersections_matrix, n1: int, n2: int):
    '''Returns the indices of the points present in both component'''
    if n1 == n2:
        ic("Same component")
        return None
    elif n2 > n1:
        pass
    else:
        tmp = n1
        n1 = n2
        n2 = tmp
    return intersections_matrix[n1][n2-n1-1]


# def plot_air_mesh(air_pts, plot_ch=False):
#     tri = Delaunay(air_pts, incremental=False, qhull_options="") # QJ is to force all points to be connected

#     mesh_air = toughio.Mesh(
#         points=air_pts, 
#         cells=[("quad", tri.simplices)]
#     ).to_pyvista()
#     mesh_air.plot()
    
#     if plot_ch:
#         ch = tri.convex_hull
#         fig = plt.figure()
#         ax = fig.add_subplot(projection='3d')
#         ax.scatter(ch[:,0], ch[:,1], ch[:,2], alpha=0.2)
#         plt.show()


# def plot_air_mesh_pyvista(air_pts):
#     cloud = pyvista.PolyData(air_pts)
#     mesh = cloud.delaunay_3d()
#     mesh.plot()


# def create_3d_point_cloud(size):
#     list_pts = []
#     for i in range(size):
#         for j in range(size):
#             for k in range(size):
#                 list_pts.append([i,j,k])
#     return np.array(list_pts)


conf = Config()
mesh_filename = os.path.join(conf.DATA_DIR, "initial_exploration", "raw", "2dtc_001R001_001_s01.msh") # 2D mesh, binary
features_filename = os.path.join(conf.DATA_DIR, "initial_exploration", "raw", "2dtc_001R001_001_s01_nodal_values.csv")
# filename = os.path.join(conf.DATA_DIR, "initial_exploration", "raw", "profilo_end.msh") # 3D mesh, ascii (broken for now)

mesh = utils.read_mesh(mesh_filename, mode="meshio")
features = pd.read_csv(features_filename)

map_mesh_to_feature = utils.match_mesh_and_feature_pts(mesh, features)

###### To visualize compontents of the mesh
# visualize_mesh_component(mesh, 1, read_from_file=False)
# subplot_mesh_components(mesh)

###### To visualize points that are in mesh.points but not inside cellblocks (so, without connectivity)
# free_idxs = get_free_points(mesh)
# plot_pts(mesh.points[free_idxs])

##### To visualize borders between components
# idxs_in_components = get_idxs_in_compontents(mesh)
# idxs_intersection = get_intersection_two_compontents(get_intersection_matrix_components(idxs_in_components), 0, 1)
# pts = mesh.points[idxs_intersection]
# plot_pts(pts)

##### Specific to the example "2dtc_001R001_001_s01" --> one component is fully inside the 
##### intersection between the two components in the mesh EXCEPT the four points in the extreme outer vertices
# outer_component = idxs_in_components[0]
# difference = outer_component.difference(idxs_intersection)
# plot_pts(mesh.points[list(difference)])



##### Investigate feature "boundary-normal-dist"
# ic(features.shape)
# boundaries = features[features["boundary-normal-dist"]==0]
# ic(boundaries.shape)
# non_boundaries = features[features["boundary-normal-dist"]!=0]
# plot_pts(features[["    x-coordinate", "    y-coordinate"]].iloc[boundaries.index].to_numpy())

features_to_remove = ['nodenumber', '    x-coordinate', '    y-coordinate', 'boundary-normal-dist']
mesh_features = features[features.columns.difference(features_to_remove)].iloc[map_mesh_to_feature]

##### To get full graph node-node connectivity
edges = utils.get_all_edges(mesh)

data = Data(x=torch.tensor(mesh_features.to_numpy()),
            edge_index=torch.tensor(edges).t().contiguous(), 
            pos=torch.tensor(mesh.points))

print(data)






