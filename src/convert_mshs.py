import os
import pickle
from time import time
from typing import Optional
import glob
from tqdm import tqdm

import torch

from config_pckg.config_file import Config
from utils import convert_msh_to_mesh_complete_info_obj, convert_mesh_complete_info_obj_to_graph


def convert_all_msh_to_meshComplete(conf: Config(), input_dir: Optional[str] = None, output_dir: Optional[str] = None, input_filenames: Optional[list[str]] = None):

    if input_dir is not None:
        input_filenames = glob.glob(os.path.join(input_dir, "*.msh"))
    else:
        assert len(input_filenames) > 0, "No input files"
    
    obj_list = []

    for filename in (pbar := tqdm(input_filenames)):
        msh_name = filename.split(os.sep)[-1].removesuffix("_ascii.msh")
        pbar.set_description(f"Parsing {msh_name}")

        if output_dir is not None:
            meshComplete_filename = os.path.join(output_dir, msh_name+".pkl")
        else:
            meshComplete_filename = None
        
        meshCompleteInstance = convert_msh_to_mesh_complete_info_obj(conf, filename, meshComplete_filename)
        
        obj_list.append(meshCompleteInstance)

    return obj_list


if __name__ == "__main__":
    conf = Config()

    # meshComplete_objs = convert_all_msh_to_meshComplete(conf, 
    #                                                     input_dir=conf.EXTERNAL_FOLDER_MSH, 
    #                                                     output_dir=conf.EXTERNAL_FOLDER_MESHCOMPLETE)
    
    # # msh_filename = os.path.join(conf.EXTERNAL_FOLDER_MSH, "2dtc_002R002_001_s01.msh")
    # # convert_all_msh_to_meshComplete(conf, input_filenames=[msh_filename])

    # for obj in (pbar := tqdm(meshComplete_objs)):
    #     pbar.set_description(f"Adding labels to : {obj.name}")
    #     if obj.name not in conf.problematic_files:
    #         obj.add_labels(os.path.join(conf.EXTERNAL_FOLDER_CSV, obj.name+"_cell_values_at300.csv"))
    #         obj.save_to_disk(os.path.join(conf.EXTERNAL_FOLDER_MESHCOMPLETE_W_LABELS, obj.name+".pkl"))

    meshComplete_paths = sorted(glob.glob(os.path.join(conf.EXTERNAL_FOLDER_MESHCOMPLETE_W_LABELS, "*.pkl")))
    for path in (pbar := tqdm(meshComplete_paths[55:])):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        pbar.set_description(f"Creating graph of : {obj.name}")
        convert_mesh_complete_info_obj_to_graph(
            conf,
            obj,
            filename_output_graph=os.path.join(conf.EXTERNAL_FOLDER_GRAPHS, obj.name+".pt")
        )
    


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
