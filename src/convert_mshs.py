import copy
import os
import pickle
from time import time
from typing import Optional
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from scipy.spatial import Delaunay

from config_pckg.config_file import Config
from utils import convert_msh_to_mesh_complete_info_obj, convert_mesh_complete_info_obj_to_graph, get_coefficients, MeshCompleteInfo


def convert_all_msh_to_meshComplete(conf: Config, output_dir: str, input_dir: Optional[str] = None, input_filenames: Optional[list[str]] = None):

    if input_dir is not None:
        input_filenames = glob.glob(os.path.join(input_dir, "*.msh"))
    else:
        assert len(input_filenames) > 0, "No input files"
    
    obj_list = []

    for filename in (pbar := tqdm(input_filenames)):

        msh_name = filename.split(os.sep)[-1].removesuffix("_ascii.msh")
        pbar.set_description(f"Parsing {msh_name}")

        if msh_name in conf.problematic_files:
            continue
        
        if output_dir is not None:
            meshComplete_filename = os.path.join(output_dir, msh_name+".pkl")
        else:
            meshComplete_filename = None

        # if msh_name < "2dtc_002R077_001_s01":
        #     continue
        # else:
        #     meshCompleteInstance = convert_msh_to_mesh_complete_info_obj(conf, filename, meshComplete_filename)
        meshCompleteInstance = convert_msh_to_mesh_complete_info_obj(conf, filename, meshComplete_filename)


def get_triangulated_cells(vertices_in_cells, pos):
    all_triangles = []

    for vertices in vertices_in_cells:
        if len(vertices) > 3:
            tri = Delaunay(pos[vertices, :2], qhull_options="QJ")

            for simplex in tri.simplices:
                all_triangles.append(tri.points[simplex])
        else:
            all_triangles.append(pos[vertices, :2])
    return np.stack(all_triangles)


def get_maximum_difference(labels_of_faces_in_cell):
    sorted_labels = labels_of_faces_in_cell.sort(dim=0)[0]
    return sorted_labels[-1,:] - sorted_labels[0,:]


def relative_error(pred: torch.Tensor, label: np.ndarray):
    label=label.values[0]
    pred=pred.numpy()
    return np.abs((label-pred)/label)

if __name__ == "__main__":
    conf = Config()

    # df = pd.read_csv(os.path.join(conf.DATA_DIR, "parsed_files.csv"), index_col=1)
    #################### convert all
    input_filepaths = glob.glob(os.path.join(conf.EXTERNAL_FOLDER_MSH, "*.msh"))
    input_filenames = [p.split(os.sep)[-1].removesuffix("_ascii.msh") for p in input_filepaths]
    acceptable_input_filenames = sorted(list(set(input_filenames).difference(conf.problematic_files)))

    acceptable_input_filepaths = [os.path.join(conf.EXTERNAL_FOLDER_MSH, p+"_ascii.msh") for p in acceptable_input_filenames]
    meshComplete_paths = sorted(glob.glob(os.path.join(conf.EXTERNAL_FOLDER_MESHCOMPLETE_W_LABELS, "*.pkl")))
    graph_paths = sorted(glob.glob(os.path.join(conf.EXTERNAL_FOLDER_GRAPHS, "*.pt")))
    
    # df_errors = pd.DataFrame({}, columns=[
    #     "last_iter",
    #     "cd_main_pressure", "cl_main_pressure", "cd_main_total", "cl_main_total",
    #     "cd_tyre_pressure", "cl_tyre_pressure", "cd_tyre_total", "cl_tyre_total"
    # ])
    
    dataset_files = sorted(glob.glob(os.path.join(conf.standard_dataset_dir, "*.pt")))

    n_edges = 0
    n_nodes = 0
    n_cells = 0
    n_boundary = 0
    
    for i, (path_msh, name, path_m, path_g, path_dataset_f) in tqdm(enumerate(zip(
        acceptable_input_filepaths, acceptable_input_filenames, meshComplete_paths, graph_paths, dataset_files)), 
        total=len(meshComplete_paths)):

        assert path_g.split(".")[0].split(os.sep)[-1] == path_m.split(".")[0].split(os.sep)[-1]

        # name_df_entry = name.removeprefix("2dtc_").removesuffix("_001_s01")
        # coeffs = df[df.index == name_df_entry]

        # if coeffs.shape[0] != 0:
        # meshCI_new = convert_msh_to_mesh_complete_info_obj(conf, path_msh, compute_radial_attributes=False)
        # meshCI_new.add_labels(os.path.join(conf.EXTERNAL_FOLDER_CSV, name+"_cell_values_at300.csv"))
        # meshCI_new.save_to_disk(path_m)
        with open(path_m, "rb") as f:
            meshCI_old = copy.copy(pickle.load(f))

        n_edges += meshCI_old.FcFc_edges.shape[0]
        n_nodes += meshCI_old.face_areas.shape[0]
        n_cells += meshCI_old.cell_center_positions.shape[0]
        n_boundary += meshCI_old.face_center_features_mask[:,-1].sum()
        pass
        # meshCI_new.radial_attributes = meshCI_old.radial_attributes
        # meshCI_new.face_center_features = np.concatenate((meshCI_new.face_center_features, meshCI_new.radial_attributes), axis=1)
        # meshCI_new.face_center_labels = meshCI_old.face_center_labels
        
        # data = convert_mesh_complete_info_obj_to_graph(conf, meshCI_new, filename_output_graph=path_g)
        
        #     cd_main_pressure, cl_main_pressure = data.components_coefficients["main_flap"][0]
        #     cd_tyre_pressure, cl_tyre_pressure = data.components_coefficients["tyre"][0]
        #     # relative_error_wrt_p = 
        #     cd_main_total, cl_main_total = sum(data.components_coefficients["main_flap"])
        #     cd_tyre_total, cl_tyre_total = sum(data.components_coefficients["tyre"])

        #     df_errors.loc[len(df_errors.index)] = [
        #         coeffs["last_iter"].values[0],
        #         relative_error(cd_main_pressure, coeffs["main_cd"]),
        #         relative_error(cl_main_pressure, coeffs["main_cl"]),
        #         relative_error(cd_main_total, coeffs["main_cd"]),
        #         relative_error(cl_main_total, coeffs["main_cl"]),
        #         relative_error(cd_tyre_pressure, coeffs["tyre_cd"]),
        #         relative_error(cl_tyre_pressure, coeffs["tyre_cl"]),
        #         relative_error(cd_tyre_total, coeffs["tyre_cd"]),
        #         relative_error(cl_tyre_total, coeffs["tyre_cl"]),
        #         ]
        # else:
        #     print("No history found")


        # relative_error_wrt_p = 

        # print(data)
        # data.faces_in_cell = data.faces_in_cell.T
        # data.components_coefficients = get_coefficients(conf, data, pressure_values=data.y[:,2],
        #     velocity_derivatives=data.y_additional[:,2:], turbulent_values=data.y[:,3:])
        # pass
        # data_graph = copy.copy(torch.load(path_g))
        # dataset_file = copy.copy(torch.load(path_dataset_f))
        # print(data_graph.y.mean(dim=0))
        # print(dataset_file.y.mean(dim=0))
        # pass
        # omega = data.y[...,4] / (0.09*data.y[...,3])
        # data.y[...,4] = omega
        # tmp = [data.CcFc_edges[data.CcFc_edges[:,0]==i,1] for i in range(data.CcFc_edges[:,0].max()+1)]
        # data.faces_in_cell = torch.nn.utils.rnn.pad_sequence(
        #     tmp, 
        #     padding_value=-1
        # ).T
        # data.n_cells = data.len_faces.shape[0]
        # torch.save(data, path_g)
        # if i > 48:
        #     with open(path_m, "rb") as f: # 48 double flap
        #         meshCI: MeshCompleteInfo = copy.copy(pickle.load(f))
            
        # # meshCI.plot_mesh(what_to_plot=[("face", "feature", "v_t")], conf=conf)
        #     if np.where(meshCI.face_center_additional_features[:,meshCI.conf.graph_node_features_not_for_training["second_flap"]])[0].shape[0] > 0:
        #         meshCI.plot_mesh(plot_boundary_group=True)
        #         break
    # df_errors.to_csv(os.path.join(conf.DATA_DIR, "relative_errors.csv"))

    print("REMEMBER TO COPY IT TO THE PC FOLDER")
    
    print(f"Edges {n_edges / i}")
    print(f"Nodes {n_nodes / i}")
    print(f"Cells {n_cells / i}")
    print(f"B faces {n_boundary / i}")


def initial_trial():
    conf = Config()

    msh_name = "2dtc_001R001_001_s01"

    msh_filename = os.path.join(conf.DATA_DIR, "msh", msh_name+"_ascii.msh") # 2D mesh, ASCII
    tmp_mesh_info_complete_filename = os.path.join(conf.DATA_DIR, "mesh_complete_info", msh_name+".pkl")
    labels_filename = os.path.join(conf.DATA_DIR, "csv", msh_name+"_cell_values.csv")
    final_data_filename = os.path.join(conf.DATA_DIR, "raw", msh_name+".pt")

    tic = time()
    meshCompleteInstance = convert_msh_to_mesh_complete_info_obj(
        conf,
        msh_filename
    )
    print("msh -> meshComplete:         ", time()-tic)
    # To save "barebone graph" without labels
    meshCompleteInstance.save_to_disk(tmp_mesh_info_complete_filename)
    # or put the path inside the convert function directly

    # To add labels and then save it
    tic=time()
    meshCompleteInstance.add_labels(labels_filename, mode="element")
    print("add labels to meshComplete:  ", time()-tic)

    # meshCompleteInstance.save_to_disk(tmp_mesh_info_complete_filename)

    # To load it from disk:
    # with open(tmp_mesh_info_complete_filename, 'rb') as f:
    #     meshCompleteInfoInstance = pickle.load(f)

    # To plot it
    # meshCompleteInstance.plot_mesh([
    #         ("face", "label", "pressure"),
    #         ("face", "label", "x-velocity"),
    #         ("face", "label", "y-velocity"),
    #         ("face", "feature", "v_t"),
    #         ("cell", "label", "pressure"),
    #         ("cell", "label", "x-velocity"),
    #         ("cell", "label", "y-velocity"),
    #         ("cell", "label", "streamlines"),
    # ])

    # To transform it to graph
    tic = time()
    data = convert_mesh_complete_info_obj_to_graph(
            conf,
            meshCompleteInstance,
            complex_graph=False,
            filename_output_graph=final_data_filename
    )
    print("meshComplete -> graph:       ", time()-tic)

    # To load it from disk:
    data = torch.load(final_data_filename)

    # To visualize data from graph:
    tic = time()
    meshCompleteInstance.add_labels_from_graph(
        data=data,
        which_element_has_labels="face"
    )
    print("graph + meshComplete -> add_labels:  ", time()-tic)
    
    meshCompleteInstance.plot_mesh([
        ("cell", "label", "pressure"),
        ("cell", "label", "x-velocity"),
        ("cell", "label", "y-velocity"),
    ])
