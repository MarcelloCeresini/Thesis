import os
import pickle
from time import time

import torch

from config_pckg.config_file import Config
from utils import convert_msh_to_mesh_complete_info_obj, convert_mesh_complete_info_obj_to_graph


if __name__ == "__main__":
    conf = Config()

    msh_name = "2dtc_001R001_001_s01"

    msh_filename = os.path.join(conf.DATA_DIR, "msh", msh_name+"_ascii.msh") # 2D mesh, ASCII
    tmp_mesh_info_complete_filename = os.path.join(conf.DATA_DIR, "mesh_complete_info", msh_name+".pkl")
    labels_filename = os.path.join(conf.DATA_DIR, "csv", msh_name+"_cell_values.csv")
    final_data_filename = os.path.join(conf.DATA_DIR, "raw", msh_name+".pt")


    tic = time()
    meshCompleteInstance = convert_msh_to_mesh_complete_info_obj(
        conf,
        msh_filename,
        tmp_mesh_info_complete_filename
    )
    print("msh -> meshComplete:             ", time()-tic)
    # To save "barebone graph" without labels
    meshCompleteInstance.save_to_disk(tmp_mesh_info_complete_filename)

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