import glob
import os
import pickle

import numpy as np
import utils
from tqdm import tqdm

import wandb
import torch
import torch_geometric
from torch_geometric.data import Data
from torch.optim import Adam
# import torchinfo
from torch_geometric.nn import summary
from torch_geometric.utils.convert import to_networkx
import networkx, rustworkx
from time import time
import matplotlib.pyplot as plt

import utils
from config_pckg.config_file import Config
from data_pipeline.data_loading import get_data_loaders, CfdDataset
from utils import convert_mesh_complete_info_obj_to_graph, plot_gt_pred_label_comparison, get_input_to_model
from models.models import get_model_instance, PINN
from train import test


def get_training_data(run_name, conf, from_checkpoints:bool):
    with open(os.path.join(conf.DATA_DIR, "model_runs", run_name+"_full_conf.pkl"), "rb") as f:
        model_conf = torch.load(f)

    model = get_model_instance(model_conf)

    if not from_checkpoints:
        model.load_state_dict(torch.load(os.path.join(conf.DATA_DIR, "model_runs", f"{run_name}.pt")))
    else:
        checkpoints = sorted(os.listdir(os.path.join(conf.DATA_DIR, "model_checkpoints", run_name)))
        idx_max_ckpt = np.argmax([int(x.split("_")[0]) for x in checkpoints])
        if run_name.split("_")[-1] == "opt":
            tmp = torch.load(os.path.join(conf.DATA_DIR, "model_checkpoints", run_name, checkpoints[idx_max_ckpt]))
            epoch = tmp["epoch"]
            model.load_state_dict(tmp["model_state_dict"])
            opt = Adam()
            opt.load_state_dict(tmp["optimizer_state_dict"])

    model.eval()

    return model, model_conf


def get_last_training(conf, from_checkpoints: bool =False):
    if not from_checkpoints:
        dirlist = sorted(glob.glob(os.path.join(conf.DATA_DIR, "model_runs", "*.pt")))
        run_name = dirlist[-1].split(".")[0]
    else:
        dirlist = sorted(os.listdir(os.path.join(conf.DATA_DIR, "model_checkpoints")))
        run_name = dirlist[-1]
    model, model_conf = get_training_data(run_name, conf, from_checkpoints=from_checkpoints)
    return model, model_conf, run_name



def get_run_from_id(run_id):
    wandb.init(id=run_id, resume="must")
    conf = wandb.config

    dirlist = glob.glob(os.path.join(conf.DATA_DIR, "model_runs", "*.pt"))
    run_names = [v.split(os.sep)[-1] for v in dirlist]
    run_ids = [v.split("-")[-1].split(".")[0] for v in run_names]

    raise NotImplementedError("complete this")
    if run_id in run_ids:
        pass
        
        # model, model_conf = get_training_data(run_name, wandb.conf, from_checkpoints=from_checkpoints)
    else:
        pass


def add_test_results_from_last_checkpoint(conf, test_dataloader):
    model, model_conf, run_name = get_last_training(conf, from_checkpoints=True)

    id = run_name.split("-")[-1]
    wandb.init(id=id, resume="must")

    with torch.no_grad():
        test_loss, metric_results = test(test_dataloader, model, conf)
        print(f"Test loss: {test_loss}")
        print(f"Test metrics: {metric_results}")

        metric_results = {f"test_{k}":v for k,v in metric_results.items()}
        metric_results.update({"test_loss":test_loss})
        wandb.log(metric_results)

        utils.plot_test_images_from_model(conf, model, run_name, test_dataloader)


if __name__ == "__main__":
    
    # get_run_from_id("2awfxt5j")

    # api = wandb.Api()

    # run = api.run("/Thesis/69phokhj")

    # for i, row in run.history().iterrows():
    #     print(row["_timestamp"], row["accuracy"])
    

    utils.init_wandb(Config(), overwrite_WANDB_MODE="offline")
    conf = wandb.config
    # run_name = wandb.run.dir.split(os.sep)[-2]
    print("Getting dataloaders")
    train_dataloader, val_dataloader, test_dataloader, train_dataloader_for_metrics = get_data_loaders(conf)
    print("done")

    # model, model_conf, run_name = get_last_training(conf)
    ####### print results of last training
    # plot_test_images_from_model(conf, test_dataloader)
    # model, model_conf, run_name = get_last_training(conf, from_checkpoints=False)
    # plot_test_images_from_model(conf, model, run_name, test_dataloader)


    # ######## try if the model works
    model = get_model_instance(conf) # build model

    # conf.device = "cpu"

    model.to(conf.device)

    # plot_test_images_from_model(conf, model, run_name, test_dataloader)

    opt = Adam(
        params = model.parameters(),
        lr = conf.hyper_params["training"]["lr"],
        weight_decay = conf.hyper_params["training"]["weight_decay"],
    )
    opt.zero_grad(set_to_none=True)

    for batch in train_dataloader_for_metrics:
        batch.to(conf.device, non_blocking=True)
        y = model(**get_input_to_model(batch))
        loss = model.loss(y, batch.y, batch)
        loss[0].backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10, foreach=True)
        opt.step()
        break
    # model_summary = summary(model, **get_input_to_model(batch), leaf_module=None) # run one sample through model
    # print(model_summary)
