import os
from typing import Literal, Optional, Union
from time import time
import itertools
import sys

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


def subplot_mesh_components(mesh, idxs_cellblocks = [], for_loop = False, debug=False):
    '''Plots in a row all specified cell components. If not specified, plots all'''
    if len(idxs_cellblocks) == 0:
        idxs_cellblocks = range(len(mesh.cells))
    
    if for_loop:
        for i in idxs_cellblocks:
            pl = pyvista.Plotter()
            zone_id = mesh.info['zone_id_list_cellblocks'][i]
            elem_info = mesh.info["elements"][zone_id]
            zone_name = mesh.info['global'][str(zone_id)]['zone_name'] if str(zone_id) in mesh.info["global"].keys() else "None"
            pl.add_title(f"Name: {zone_name} - Index: {i} - Type: {mesh.cells[i].type} - BC: {conf.bc_dict[elem_info['bc_type']]} - #Elements: {mesh.cells[i].data.shape[0]}")
            mesh_part = get_mesh_component(mesh, i).to_pyvista()
            pl.add_mesh(mesh_part, color="r")
            pl.add_mesh(mesh, color="grey", opacity=0.2)
            pl.camera_position = "xy"
            pl.show(window_size=(int(3000), int(1700)))

    else:
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


def plot_pts(pts, fig = False, show = True):
    '''Plot pointcloud indipendently of dimension'''
    fig = plt.figure()
    
    if len(pts.shape) == 2:
        ax = fig.add_subplot()
        ax.scatter(pts[:,0], pts[:,1], alpha=0.2)
    elif len(pts.shape) == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], alpha=0.2)
    
    return fig, ax



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


def plot_2d_cfd(mesh: Union[meshio.Mesh, pyvista.UnstructuredGrid], mesh_features: pd.DataFrame, conf: Config, plot_streamlines=False):
    # https://docs.pyvista.org/version/stable/user-guide/data_model.html
    # https://docs.pyvista.org/version/stable/examples/99-advanced/openfoam-tubes.html
    if isinstance(mesh, meshio.Mesh):
        pyv_mesh = toughio.from_meshio(mesh).to_pyvista()
        # surf = pyvista.wrap(pyv_mesh.points).reconstruct_surface(progress_bar=True)
        # surf.plot()
    elif isinstance(mesh, pyvista.UnstructuredGrid):
        pyv_mesh = mesh
    

    velocity = np.concatenate([
                    mesh_features[conf.active_vectors_2d].to_numpy(),
                    np.zeros([len(mesh_features),1])], 
                axis=1)

    for feature in mesh_features:
        pyv_mesh.point_data[feature] = mesh_features[feature]

    pyv_mesh.point_data["velocity"] = velocity
    pyv_mesh.points[:,2] *= 0 # sanity check
    pyv_mesh.set_active_vectors("velocity")

    ### Plot pressure
    pl = pyvista.Plotter()
    pl.add_mesh(pyv_mesh, scalars='        pressure', lighting=False, scalar_bar_args={'title': 'Pressure'}, cmap="Spectral")
    pl.camera_position = 'xy'
    pl.enable_anti_aliasing()
    pl.show()

    pl = pyvista.Plotter(shape=("1/2"))
    pl.subplot(2)
    pl.add_mesh(pyv_mesh, scalars='velocity', lighting=False, scalar_bar_args={'title': 'Velocity Magnitude'}, cmap="Spectral")             
    pl.camera_position = 'xy'
    pl.subplot(0)
    pl.add_mesh(pyv_mesh, scalars='velocity', component=0, lighting=False, scalar_bar_args={'title': 'x-Velocity'}, cmap="Spectral")
    pl.camera_position = 'xy'
    pl.subplot(1)
    pl.add_mesh(pyv_mesh, scalars='velocity', component=1, lighting=False, scalar_bar_args={'title': 'y-Velocity'}, cmap="Spectral")
    pl.camera_position = 'xy'
    pl.show()

    pl = pyvista.Plotter(shape=("1/1"))
    pl.subplot(0)
    pl.add_mesh(pyv_mesh, scalars='  turb-diss-rate', lighting=False, scalar_bar_args={'title': 'turb-diss-rate'}, cmap="Spectral")
    pl.camera_position = 'xy'
    pl.subplot(1)
    pl.add_mesh(pyv_mesh, scalars='turb-kinetic-energy', lighting=False, scalar_bar_args={'title': 'turb-kinetic-energy'}, cmap="Spectral")
    pl.camera_position = 'xy'
    pl.show()

    # #### Plot velocity
    
    # #### Plot categorical data in mesh
    # pl.add_mesh(mesh, scalars="somethin", categproes=True)

    if plot_streamlines:
        # TODO: need to find a better way to get boundary conditions
        outer_component_idxs = list(get_idxs_in_compontents(mesh)[0])
        inlet_idxs = np.array([idx for idx in outer_component_idxs if (mesh.points[idx][0]< -1.725) and (0.01 < mesh.points[idx][1] < 1.95)])
        # inlet_idxs = np.random.choice(range(len(mesh.points)), size=100)
        inlet_pts = mesh.points[inlet_idxs]
        velocity_at_inlet = velocity[inlet_idxs]

        # move_them to avoid ReasonForTermination=1 "OUT_OF_DOMAIN" (starting streamlines from the mesh bounds probably leads to problems?)
        inlet_pts[:,0] += 5e-3

        ###### Plot where are the inlet points
        # fig, ax = plot_pts(mesh.points, show=False)
        # ax.scatter(inlet_pts[:,0], inlet_pts[:,1])    

        pset = pyvista.PointSet(np.concatenate([
            inlet_pts, 
            np.zeros([len(inlet_pts), 1])], axis=1))
        
        lines = pyv_mesh.streamlines_from_source(
            source=pset,
            vectors="velocity",
            max_time=int(1e4),
            step_unit="l",
            min_step_length=1e-7,
            initial_step_length=1e-4,
            # max_step_length=1e2,
            # interpolator_type="cell",
            # integration_direction="backward",
            max_steps=int(1e4),
            surface_streamlines=True
        )

        len_lines = list(map(lambda x: len(x.points), lines.cell))
        if len_lines:
            ic(lines.points.shape, min(len_lines), np.mean(len_lines), max(len_lines))
            ic(np.unique(lines["ReasonForTermination"], return_counts=True))
            tic = time()
            # lines = lines.clip_box(pyv_mesh.bounds, invert=False)
            ic(time()-tic)
            ic(lines.points.shape)
        else:
            raise ValueError("No lines produced")

        for i, cell in enumerate(lines.cell):
            if lines["ReasonForTermination"][i] in [1, 3]:
                print(cell.points[-6:, :2])

        pl = pyvista.Plotter()
        pl.add_mesh(
            lines,
            render_lines_as_tubes=True,
            line_width=3,
            lighting=False,
            scalar_bar_args={'title': 'Flow Velocity'},
            scalars='velocity',
        )
        pl.add_mesh(pyv_mesh, color="grey", opacity=0.1)
        pl.enable_anti_aliasing()
        pl.show()

        return inlet_pts

######## TO CREATE MESH FOR FREE POINTS ############
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
######## ######################################### ############

def get_adjacency_list(mesh, idx_component):
    return(mesh.info["elements"][mesh.info["zone_id_list_cellblocks"][idx_component]]["adj_data"])

# def get_cell_from_lines(cell_list, adjacency_list, cell, max_n_vertices = 4):
    
#     if len(cell) > max_n_vertices + 1:
#         # if we expect triangles and quadrilaterals it makes no sense to search more than size=4
#         return
#     elif len(cell) >= 3:
#         if cell[-1] == cell[-3]:
#             # step backward, not interesting to reconstruct cells
#             return
#         elif cell[-1] == 0:
#             # adjacency with 0 means no adjacency
#             return
#         elif cell[0] == cell[-1]:
#             # good cycle, cell complete
#             cell_list.append(cell[:-1])
#             return
    
#     idxs_l = np.argwhere(adjacency_list[:,0]==cell[-1])
#     idxs_r = np.argwhere(adjacency_list[:,1]==cell[-1])
#     tmp = [adjacency_list[idx, 1] for idx in idxs_l] + [adjacency_list[idx, 0] for idx in idxs_r]
#     for vertex in tmp:
#         get_cell_from_lines(cell_list, adjacency_list, np.concatenate([cell, vertex]), max_n_vertices) # recursive call
#     # elif candidate_next_vertex == cell[-2]:
#     #     # backward step, do not search further
#     #     return 
#     # else:
#     #     cell = np.concatenate([cell, candidate_next_vertex])
#     #     idxs = np.argwhere(adjacency_list[:,0]==cell[-1])
#     #     tmp = [adjacency_list[idx,1] for idx in idxs]
#     #     for candidate_next_vertex in tmp:
#     #         cell = get_cell_from_lines(cell_list, adjacency_list, cell, candidate_next_vertex, max_size=max_size)
#     #         if cell:
#     #             cell_list.append(cell)


# def recreate_2d_faces_recursive(mesh):
#     '''Doesn't work because adjacency list is not for faces but cells'''
#     edge_list = np.concatenate([c.data for c in mesh.cells])
#     adjacency_list = np.concatenate([get_adjacency_list(mesh, i) for i in range(len(mesh.cells))])
#     cells_vertices = []
#     for i, edge in enumerate(edge_list):
#         cell_list = []

#         get_cell_from_lines(cell_list, adjacency_list, edge, max_n_vertices=4)
#         if len(cell_list) > 0:
#             cells_vertices += cell_list
        
#         if i > 100:
#             break
    
#     cellblock = {
#         "triangle": [],
#         "quad": []
#     }

#     for face in cells_vertices:
#         if len(face) == 3:
#             cellblock["triangle"].append(face)
#         elif len(face) == 4:
#             cellblock["quad"].append(face)
#         else:
#             raise ValueError("len cannot be != 3 and != 4")

#     resulting_mesh = toughio.Mesh(mesh.points, cells=[(key, np.stack(cellblock[key])) for key in cellblock])
#     resulting_mesh.to_pyvista().plot()
#     return cells_vertices


def make_cell(a):
    cell = list(a[0,:])
    for i in range(1,len(a)):
        idx = np.argwhere(a[:,0]==cell[i])[0][0]
        cell += [a[idx,1]]
    assert cell[0] == cell[-1]
    return cell[:-1]


def recreate_cells(mesh):
    edge_list = np.concatenate([c.data for c in mesh.cells])
    adjacency_list = np.concatenate([get_adjacency_list(mesh, i) for i in range(len(mesh.cells))])
    cell_types_list = mesh.info["elements"][8]["cell_type_cumulative"]

    unique_vals, unique_counts = np.unique(cell_types_list, return_counts=True)

    cellblock = {
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
        cellblock[conf.cell_type_dict[cell_type]].append(cell)
        cell_vertices_list.append(cell)

    # TODO: assert counts and vals correspond
    for val, count in zip(unique_vals, unique_counts):
        assert len(cellblock[conf.cell_type_dict[val]]) == count, "Msh definition of cells doesn't correspond to reconstructed cells"

    real_adjacency = adjacency_list[np.logical_and(adjacency_list[:,0] != 0, adjacency_list[:,1] != 0)]
    cell_connectivity = np.concatenate([real_adjacency, np.flip(real_adjacency, axis=1)], axis=0)
    cell_connectivity = np.unique(cell_connectivity, axis=0) # Sanity check
    
    # Bring all connections in [0, n_cells-1] (instead now they were in [0, n_cells], where 0 meant "no_connection")
    cell_connectivity -= 1

    return cellblock, cell_vertices_list, cell_connectivity


def map_vertex_pair_to_face_idx(vertex_pair, face_to_node_correspondance, face_idxs):
    return face_idxs[np.nonzero(np.logical_and(face_to_node_correspondance[:,0] == vertex_pair[0], 
                                               face_to_node_correspondance[:,1] == vertex_pair[1]))][0]


conf = Config()

mesh_filename = os.path.join(conf.DATA_DIR, "raw", "2dtc_001R001_001_s01_ascii.msh") # 2D mesh, ASCII
labels_filename = os.path.join(conf.DATA_DIR, "raw", "2dtc_001R001_001_s01_cell_values.csv")
final_data_filename = os.path.join(conf.DATA_DIR, "interim", "2dtc_001R001_001_s01_ascii_W_LABELS.pt") 

utils.convert_msh_to_graph(mesh_filename, conf,
                           filename_output_graph=final_data_filename,
                           labels_csv_filename=labels_filename)

sys.exit()

# mesh_filename = os.path.join(conf.DATA_DIR, "initial_exploration", "raw", "2dtc_001R001_001_s01.msh") # 2D mesh, binary
features_filename = os.path.join(conf.DATA_DIR, "initial_exploration", "raw", "2dtc_001R001_001_s01_nodal_values.csv")
# filename = os.path.join(conf.DATA_DIR, "initial_exploration", "raw", "profilo_end.msh") # 3D mesh, ascii (broken for now)

mesh = utils.read_mesh(mesh_filename, mode="meshio", conf=conf, plot=False)
features = pd.read_csv(features_filename)

# recreate_2d_faces(mesh)

# #### Plot "cell centers"
# pl = pyvista.Plotter()
# pl.add_mesh(mesh, color="grey", opacity=0.2)
# pl.add_mesh(mesh.cell_centers())
# pl.show()
toughio_mesh = toughio.Mesh(mesh.points, mesh.cells)
cellblock, cell_vertices_list, cell_connectivity = recreate_cells(mesh)
cell_mesh = toughio.Mesh(mesh.points, [(key, np.stack(val)) for key, val in cellblock.items()])
centers = cell_mesh.centers

#### CELL INFO
cell_center_positions = centers
cell_center_edges = cell_connectivity
cell_center_to_node_correspondance = cell_vertices_list
n_cells = len(cell_center_positions)


##### POINT INFO
point_positions = toughio_mesh.points
face_to_node_correspondance = np.concatenate([c.data for c in toughio_mesh.cells],axis=0)
face_to_node_correspondance = np.concatenate([face_to_node_correspondance, np.flip(face_to_node_correspondance, axis=1)], axis=0)
face_to_node_correspondance = np.unique(face_to_node_correspondance, axis=0) # Sanity check

assert len(face_to_node_correspondance) % 2 == 0, "Impossible to have odd n_faces (because we count them bidirectionally)"
face_vertices_to_face_idx_map = np.repeat(np.arange(int(len(face_to_node_correspondance)/2)), 2) # Map both direction of the edge to the same face

cellblock_faces_idx = []
for c in toughio_mesh.cells:
    data = c.data
    tmp = np.stack([map_vertex_pair_to_face_idx(face, face_to_node_correspondance, face_vertices_to_face_idx_map) for face in data])
    tmp += n_cells # shift all FACE indexes by n_cells
    cellblock_faces_idx.append(tmp)

face_center_positions = toughio_mesh.centers
graph_nodes_positions = np.concatenate([cell_center_positions, face_center_positions])

cell_center_to_face_center_edges = []
face_center_edges = []
for i, cell in enumerate(cell_center_to_node_correspondance):
    cell_face_ids = []
    for vertex_pair in np.lib.stride_tricks.sliding_window_view(cell+[cell[0]], 2):
        face_id = n_cells + map_vertex_pair_to_face_idx(vertex_pair, face_to_node_correspondance, face_vertices_to_face_idx_map)
        cell_center_to_face_center_edges.append([i, face_id])
        cell_face_ids.append(face_id)
    face_center_edges += list(itertools.combinations(cell_face_ids, 2))
    
cell_center_to_face_center_edges = np.concatenate([cell_center_to_face_center_edges, np.flip(cell_center_to_face_center_edges, axis=1)], axis=0)
face_center_edges = np.concatenate([face_center_edges, np.flip(face_center_edges, axis=1)], axis=0)

graph_edges = np.concatenate([cell_center_edges, face_center_edges, cell_center_to_face_center_edges])
graph_edge_attr = np.concatenate([np.ones((len(cell_center_edges, 1))) * conf.cell_type_dict["cell_cell"],
                                  np.ones((len(face_center_edges, 1))) * conf.cell_type_dict["face_face"],
                                  np.ones((len(cell_center_to_face_center_edges, 1))) * conf.cell_type_dict["cell_face"],
                                  ])



graph_without_features = Data(
    edge_index=torch.tensor(graph_edges).t().contiguous(), 
    pos=torch.tensor(graph_nodes_positions),
    edge_attr=torch.tensor(graph_edge_attr),
    x=None,
    y=None
)

A = mesh.info["elements"]
zone_id_bc_type = {key: conf.bc_dict[A[key]["bc_type"]] for key in A if "bc_type" in A[key].keys()}
cellblock_idx_bc_type = [zone_id_bc_type[key] for key in mesh.info["zone_id_list_cellblocks"]]
cellblock_idx_name = [mesh.info["global"][str(key)]["zone_name"] if str(key) in mesh.info["global"].keys() else "" for key in mesh.info["zone_id_list_cellblocks"] ]

colors = {
    "velocity-inlet":                             "g",
    'pressure-outlet, exhaust-fan, outlet-vent':  "b",
    "wall":                                       "r",
    "symmetry":                                     "y",
    "interior":                                     "grey"
}

if False: # plot Cell/face centers
    pl = pyvista.Plotter()
    pl.add_mesh(cell_mesh.to_pyvista(), color="grey", opacity=0.2, label="Cells")
    pl.add_mesh(pyvista.PolyData(centers), color="b", opacity=0.1, label="Cell centers")
    pl.add_mesh(pyvista.PolyData(toughio_mesh.centers), color="r", opacity=0.1, label="Face centers")
    pl.camera_position = "xy"
    pl.add_legend()
    pl.show()

if False: # plot
    pl = pyvista.Plotter()
    for i, c in enumerate(mesh.cells):
        mesh_part = toughio.Mesh(mesh.points, [c]).to_pyvista()
        pl.add_mesh(mesh_part, 
                    color=colors[cellblock_idx_bc_type[i]], 
                    label=cellblock_idx_name[i]+" "+cellblock_idx_bc_type[i], 
                    line_width=0.5 if cellblock_idx_bc_type[i]=="interior" else 5)
    pl.camera_position = "xy"
    pl.add_legend()
    pl.show()

map_mesh_to_feature = utils.match_mesh_and_feature_pts(mesh, features, conf)

###### To visualize compontents of the mesh
# visualize_mesh_component(mesh, 1, read_from_file=False)
# subplot_mesh_components(mesh, for_loop=True)

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

# #### Investigate feature "boundary-normal-dist"
# ic(features.shape)
# boundaries = features[features["boundary-normal-dist"]==0]
# ic(boundaries.shape)
# non_boundaries = features[features["boundary-normal-dist"]!=0]
# plot_pts(features[["    x-coordinate", "    y-coordinate"]].iloc[boundaries.index].to_numpy())

reduced_features = features[features.columns.difference(conf.features_to_remove)]
mesh_features = reduced_features[reduced_features.columns.difference(conf.features_coordinates)].iloc[map_mesh_to_feature]
mesh_coords = reduced_features[conf.features_coordinates].iloc[map_mesh_to_feature]

# #### To get full graph node-node connectivity
# edges = utils.get_all_edges(mesh)

# data = Data(x=torch.tensor(mesh_features.to_numpy()),
#             edge_index=torch.tensor(edges).t().contiguous(), 
#             pos=torch.tensor(mesh_coords.to_numpy()))

# from pyvista import examples
# block = examples.download_openfoam_tubes()

plot_2d_cfd(mesh, mesh_features, conf)