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
from utils import convert_mesh_complete_info_obj_to_graph, plot_gt_pred_label_comparison, get_input_to_model
from models.models import get_model_instance


def get_training_data(run_name, conf, from_checkpoints:bool):
    with open(os.path.join(conf.ROOT_DIR, "log", run_name, "full_conf.pkl"), "rb") as f:
        full_conf = pickle.load(f)
    model = get_model_instance(full_conf)

    if not from_checkpoints:
        model.load_state_dict(torch.load(os.path.join(conf.DATA_DIR, "model_runs", f"{run_name}.pt")))
    else:
        checkpoints = sorted(os.listdir(os.path.join(conf.DATA_DIR, "model_checkpoints", run_name)))
        idx_max_ckpt = np.argmax([int(x.split("_")[0]) for x in checkpoints])
        model.load_state_dict(torch.load(os.path.join(conf.DATA_DIR, "model_checkpoints", run_name, checkpoints[idx_max_ckpt])))

    model.eval()

    return model, full_conf


def get_last_training(conf, from_checkpoints: bool =False):
    if not from_checkpoints:
        dirlist = sorted(os.listdir(os.path.join(conf.DATA_DIR, "model_runs")))
        run_name = dirlist[-1].split(".")[0]
    else:
        dirlist = sorted(os.listdir(os.path.join(conf.DATA_DIR, "model_checkpoints")))
        run_name = dirlist[-1]
    model, full_conf = get_training_data(run_name, conf, from_checkpoints=from_checkpoints)
    return model, full_conf, run_name


if __name__ == "__main__":
    conf = Config()

    print("Getting dataloaders")
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(
            conf, 
            save_to_disk=False,
            load_from_disk=True,
        )
    print("done")
    
    pass
    
    ######## print results of last training
    # model, model_conf, run_name = get_last_training(conf, from_checkpoints=False)
    # model.cpu()
    # print(f"Last training: {run_name}")

    # for batch in tqdm(test_dataloader):
    #     pred_batch = model(**get_input_to_model(batch))
    #     for i in range(len(batch)):
    #         with torch.no_grad():
    #             data = batch[i]
    #             pred = pred_batch[batch.ptr[i]:batch.ptr[i+1], :]
    #             plot_gt_pred_label_comparison(data, pred, conf, run_name=run_name)

    ######## try if the model works
    # full_conf = conf.get_tensorboard_logging_info()
    # model = get_model_instance(full_conf) # build model
    # model.to(conf.device)
    # for batch in train_dataloader:
    #     break
    # model_summary = summary(model, **get_input_to_model(batch), leaf_module=None) # run one sample through model
    # print(model_summary)


    ########## Convert msh to graph
    

    ######### update graphs