import os
import pickle
from time import time
from typing import Optional
import glob
import numpy as np
from tqdm import tqdm

import torch
from scipy.spatial import Delaunay

from config_pckg.config_file import Config
from utils import convert_msh_to_mesh_complete_info_obj, convert_mesh_complete_info_obj_to_graph


def convert_all_msh_to_meshComplete(conf: Config(), output_dir: str, input_dir: Optional[str] = None, input_filenames: Optional[list[str]] = None):

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


if __name__ == "__main__":
    conf = Config()

    #################### 1
    # meshComplete_objs = convert_all_msh_to_meshComplete(
    #     conf, 
    #     output_dir=conf.EXTERNAL_FOLDER_MESHCOMPLETE,
    #     input_dir=conf.EXTERNAL_FOLDER_MSH, 
    # )
    
    #################### 2
    # input_filenames = glob.glob(os.path.join(conf.EXTERNAL_FOLDER_MESHCOMPLETE, "*.pkl"))
    # for filename in (pbar := tqdm(input_filenames)):
    #     with open(filename, "rb") as f:
    #         obj = pickle.load(f)
    #     name = obj.name.split(".")[0].removesuffix("_ascii")
    #     pbar.set_description(f"Adding labels to : {name}")
    #     if name not in conf.problematic_files:
    #         obj.add_labels(os.path.join(conf.EXTERNAL_FOLDER_CSV, name+"_cell_values_at300.csv"))
    #         obj.save_to_disk(os.path.join(conf.EXTERNAL_FOLDER_MESHCOMPLETE_W_LABELS, name+".pkl"))

    #################### 3
    meshComplete_paths = sorted(glob.glob(os.path.join(conf.EXTERNAL_FOLDER_MESHCOMPLETE_W_LABELS, "*.pkl")))
    graph_paths = sorted(glob.glob(os.path.join(conf.EXTERNAL_FOLDER_GRAPHS, "*.pt")))
    
    for path_m, path_g in tqdm(zip(meshComplete_paths, graph_paths)):
        assert path_g.split(".")[0].split(os.sep)[-1] == path_m.split(".")[0].split(os.sep)[-1]
        with open(path_m, "rb") as f:
            obj = pickle.load(f)

        data = torch.load(path_g)

        triangulated_cells = get_triangulated_cells(obj.vertices_in_cells, obj.mesh.points)

        data.triangulated_cells = torch.tensor(triangulated_cells)
        torch.save(data, path_g)



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
