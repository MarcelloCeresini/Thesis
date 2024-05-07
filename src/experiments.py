import glob
import os
import pickle
import sys
from typing import Literal

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
from utils import convert_mesh_complete_info_obj_to_graph, init_wandb, plot_gt_pred_label_comparison, get_input_to_model
from models.models import get_model_instance, PINN
from train import test


def load_model_weights(conf, run_id, model):
    dirlist = sorted(glob.glob(os.path.join(conf.DATA_DIR, "model_runs", "*.pt")))
    for finished_run in dirlist:
        finished_run_id = finished_run.split("-")[-1].removesuffix(".pt")
        if run_id == finished_run_id:
            model.load_state_dict(
                torch.load(os.path.join(conf.DATA_DIR, "model_runs", finished_run)))
            return model
    
    dirlist_checkpoints = sorted(glob.glob(os.path.join(conf.DATA_DIR, "model_checkpoints", "*")))
    for checkpoint_dir in dirlist_checkpoints:
        checkpoint_run_id = checkpoint_dir.split(os.sep)[-1].split("-")[-1]
        if run_id == checkpoint_run_id:
            correct_ckpts = os.listdir(checkpoint_dir)
            highest_number = sorted([int(ckpt_name.split("_")[0]) for ckpt_name in correct_ckpts])[-1]
            checkpoint_name = glob.glob(os.path.join(checkpoint_dir, f"{str(highest_number)}_*"))[0]
            data_from_checkpoint = torch.load(checkpoint_name)
            model.load_state_dict(data_from_checkpoint["model_state_dict"])
            return model
        
    raise FileNotFoundError("Didn't find the weights neither in runs nor in checkpoints")


def get_wandb_run_from_id(run_id):
    wandb.init(entity="marcelloceresini", project="Thesis", id=run_id, resume="must")
    return wandb.config


def add_test_results(test_dataloader, model, conf, run_name):
    with torch.no_grad():
        test_loss, metric_results = test(test_dataloader, model, conf,)
        print(f"Test loss: {test_loss}")
        print(f"Test metrics: {metric_results}")

        metric_results = {f"test_{k}":v for k,v in metric_results.items()}
        metric_results.update({"test_loss":test_loss})
        wandb.log(metric_results)

        utils.plot_test_images_from_model(conf, model, run_name, test_dataloader)


def complete_unfinished_run(run_id):
    conf = get_wandb_run_from_id(run_id)
    
    model = get_model_instance(conf)
    run_name = wandb.run.dir.split(os.sep)[-2]
    model = load_model_weights(conf, wandb.run.id, model)
    
    print("Getting dataloaders")
    train_dataloader, val_dataloader, test_dataloader, train_dataloader_for_metrics = get_data_loaders(conf)
    print("done")

    for batch in train_dataloader_for_metrics:
        batch.to(conf["device"])
        break

    model_summary = summary(model, **get_input_to_model(batch), leaf_module=None)
    print(model_summary)

    add_test_results(test_dataloader, model, conf, run_name)


if __name__ == "__main__":

    complete_unfinished_run("0egkpy0z")
    sys.exit()

    WANDB_MODE: Literal["online", "offline"] = "online"

    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is not None:
        if gettrace():
            print('Hmm, Big Debugger is watching me --> setting wandb to offline')
            WANDB_MODE="offline"

    platform = sys.platform
    if platform == "linux" or platform == "linux2":
        WANDB_MODE="offline"

    init_wandb(Config(), overwrite_WANDB_MODE=WANDB_MODE)
    conf = wandb.config
    # model, model_conf, run_name = get_last_training(conf)
    ####### print results of last training
    # plot_test_images_from_model(conf, test_dataloader)
    # model, model_conf, run_name = get_last_training(conf, from_checkpoints=False)
    # plot_test_images_from_model(conf, model, run_name, test_dataloader)

    # ######## try if the model works
    model = get_model_instance(conf) # build model
    print("Getting dataloaders")
    train_dataloader, val_dataloader, test_dataloader, train_dataloader_for_metrics = get_data_loaders(conf)
    print("done")

    # conf.update({"device": "cpu"}, allow_update=True)
    # conf.device = "cpu"

    model.to(conf.device)

    # plot_test_images_from_model(conf, model, run_name, test_dataloader)

    

    opt = Adam(
        params = model.parameters(),
        lr = conf.hyper_params["training"]["lr"],
        weight_decay = conf.hyper_params["training"]["weight_decay"],
    )
    opt.zero_grad(set_to_none=True)

    for batch in train_dataloader:
        batch.to(conf.device, non_blocking=True)
        y = model(**get_input_to_model(batch))
        loss = model.loss(y, batch.y, batch)
        loss[0].backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10, foreach=True)
        opt.step()
        break
    # model_summary = summary(model, **get_input_to_model(batch), leaf_module=None) # run one sample through model
    # print(model_summary)
