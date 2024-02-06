import glob
import os
import pickle

import numpy as np
import utils
from tqdm import tqdm

import torch
import torch_geometric
from torch_geometric.data import Data
# import torchinfo
from torch_geometric.nn import summary
from torch_geometric.utils.convert import to_networkx
import networkx, rustworkx, igraph
from time import time


import utils
from config_pckg.config_file import Config
from data_pipeline.data_loading import get_data_loaders
from utils import convert_mesh_complete_info_obj_to_graph, plot_gt_pred_label_comparison
from models.models import get_model_instance


def get_training_data(run_name, conf):
    with open(os.path.join(conf.ROOT_DIR, "log", run_name, "full_conf.pkl"), "rb") as f:
        full_conf = pickle.load(f)
    model = get_model_instance(full_conf)
    model.load_state_dict(torch.load(os.path.join(conf.DATA_DIR, "model_runs", f"{run_name}.pt")))
    model.eval()

    return model, full_conf


def get_last_training(conf):
    dirlist = sorted(os.listdir(os.path.join(conf.ROOT_DIR, "log")))
    model, full_conf = get_training_data(dirlist[-1], conf)
    return model, full_conf


if __name__ == "__main__":
    conf = Config()

    print("Getting dataloaders")
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(
        conf, load_from_disk=True)
    

    
    ############
    # dirlist = sorted(os.listdir(os.path.join(conf.ROOT_DIR, "log")))
    # run_name = dirlist[-1]
    # with open(os.path.join(conf.ROOT_DIR, "log", run_name, "full_conf.pkl"), "rb") as f:
    #     full_conf = pickle.load(f)
    # model = get_model_instance(full_conf)
    # model.load_state_dict(
    #     torch.load(
    #         os.path.join(conf.ROOT_DIR, "data", "model_checkpoints", run_name, "85_ckpt.pt")
    # ))
    
    # model, old_full_conf = get_last_training(conf)

    # model = get_model_instance(conf.get_tensorboard_logging_info())
    
    ############

    # model, full_conf = get_last_training(conf)
    data = test_dataloader.dataset[0]

    adj = torch_geometric.utils.to_dense_adj(data.edge_index)[0].numpy().astype(np.float64)
    rust_g = rustworkx.PyGraph.from_adjacency_matrix(adj)
    print(time())
    distance_matrix = rustworkx.distance_matrix(rust_g, parallel_threshold=10000)
    # a = to_networkx(data)
    print(time())
    ########
    bc_nodes_list = []
    other_nodes = []
    n_nodes = data.x.shape[0]
    for i, mask in enumerate(data.x_mask):
        if sum(mask) >= 4:
            bc_nodes_list.append(i)
        else:
            other_nodes.append(i)

    dist_from_bc_per_node = np.min(distance_matrix[:, np.array(bc_nodes_list)], axis=1)
    max_dist_from_bc = np.max(dist_from_bc_per_node)

    data["dist_from_bc"] = dist_from_bc_per_node
    data["max_dist_from_bc"] = max_dist_from_bc


    # max_dist = 0
    # c = 0
    # for i, node in tqdm(enumerate(other_nodes), total=len(other_nodes)):
    #     min_dist_from_this_to_bc = n_nodes
    #     for j, bc_node in enumerate(bc_nodes_list):
    #         c+=1
    #         tmp = networkx.shortest_path_length(a, node, bc_node)
    #         if tmp < min_dist_from_this_to_bc:
    #             min_dist_from_this_to_bc = tmp
    #             if min_dist_from_this_to_bc < max_dist:
    #                 break
    #     if min_dist_from_this_to_bc > max_dist:
    #         max_dist = min_dist_from_this_to_bc
    #         print()
    #         print(f"{c/((i+1)*len(bc_nodes_list))*100:.2f}% - {i} - {max_dist}")
    
    # print(f"{c/(len(other_nodes)*len(bc_nodes_list))*100:.2f}%")
    ######


    # for batch in test_dataloader:
    #     input_to_model = batch.x, batch.x_mask, batch.edge_index, batch.edge_attr, batch.batch
    #     pred_batch = model(*input_to_model)
    #     for i in range(len(batch)):
    #         with torch.no_grad():
    #             data = batch[i]
    #             pred = pred_batch[batch.ptr[i]:batch.ptr[i+1], :]
    #             plot_gt_pred_label_comparison(data, pred, conf)
    #             break
    #     break

    
