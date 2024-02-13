from copy import deepcopy
import os
from typing import Literal
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.data as pyg_data
import wandb
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler 
from torch.masked import masked_tensor

from utils import print_memory_state_gpu, get_input_to_model
from config_pckg.config_file import Config 


def clean_labels(batch, conf: Config):
    if hasattr(batch, "y_mask"):
        if batch.y_mask is not None:
            # FIXME: still no good support for masked tensors
            labels = torch.masked.masked_tensor(batch.y, batch.y_mask)
    else:
        labels = batch.y
    return labels[:,:conf["hyperparams"]["label_dim"]]


def forward_metric_results(preds, labels, conf, metric_dict):
    for metric_obj_subdict in metric_dict.values():
        for i, column in enumerate(conf.labels_to_keep_for_training):
                metric_obj_subdict[column].forward(preds[:,i], labels[:,i])
    return metric_dict


def compute_metric_results(metric_dict, conf):
    metric_results = metric_dict.copy() # just to copy the structure
    for metric_res_subdict, metric_obj_subdict in zip(metric_results.values(), metric_dict.values()):
        for column in conf.labels_to_keep_for_training:
            metric_res_subdict[column] = metric_obj_subdict[column].compute()
    return metric_results


def write_metric_results(metric_results, writer, epoch, split="val") -> None:
    # TODO: add also global writing when global is implemented
    for metric_name, metric_res_subdict in metric_results.items():
        for column, result in metric_res_subdict.items():
            writer.add_scalar(f"{metric_name}/{split}/{column}", result, epoch)


def test(loader: pyg_data.DataLoader, model, conf):
    metric_dict = deepcopy(conf.metric_dict)

    with torch.no_grad():
        total_loss = 0
        model.eval()
        for batch in loader:
            input_to_model = get_input_to_model(batch)
            pred = model(**input_to_model)
            
            labels = clean_labels(batch, model.conf)
            loss = model.loss(pred, labels)
            total_loss += loss.item() * batch.num_graphs
            metric_dict = forward_metric_results(pred.cpu(), labels.cpu(), conf, metric_dict)

        total_loss /= len(loader.dataset)
        metric_results = compute_metric_results(metric_dict, conf)

    del metric_dict
    return total_loss, metric_results


def train(
        model,
        train_loader, 
        val_loader, 
        writer: SummaryWriter, 
        conf: Config):
    run_name = os.path.basename(os.path.normpath(writer.get_logdir()))
    os.mkdir(os.path.join(conf.DATA_DIR, "model_checkpoints", run_name))
    # with wandb.init(**conf.get_logging_info()):

    # TODO: change the name "features_to_keep" to "labels_to_keep"
    
    # wandb.watch(
    #     model,
    #     log=conf.logging["model_log_mode"],
    #     log_freq=conf.logging["n_batches_freq"],
    #     log_graph=conf.logging["log_graph"]
    # )

    opt = Adam(
        params = model.parameters(),
        lr = conf.hyper_params["training"]["lr"],
        weight_decay = conf.hyper_params["training"]["weight_decay"],
    )

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=opt, 
        patience=conf.hyper_params["training"]["patience_reduce_lr"],
        min_lr=conf.hyper_params["training"]["min_lr"],
        cooldown=conf.hyper_params["training"]["cooldown"]
    )

    scheduler_for_training_end = lr_scheduler.ReduceLROnPlateau(
        optimizer=opt,
        patience=conf.hyper_params["training"]["patience_end_training"],
    )

    best_loss = 1000000
    best_epoch = 0

    print_memory_state_gpu("Before training", conf)

    for epoch in tqdm(range(conf.hyper_params["training"]["n_epochs"]), desc="Epoch", position=0):
        total_loss = 0
        model.train()
        for batch in tqdm(train_loader, leave=False, desc="Batch in epoch", position=1):
            opt.zero_grad()
            input_to_model = get_input_to_model(batch)
            pred = model(**input_to_model)
            labels = clean_labels(batch, model.conf)
            loss = model.loss(pred, labels)

            loss.backward()
            opt.step()

            total_loss += loss.item() * batch.num_graphs

        total_loss /= len(train_loader.dataset)
        writer.add_scalar(f"{conf.hyper_params['loss']}/train", total_loss, epoch)
        writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch) # learning rate
        writer.add_scalar("epoch", epoch, epoch) # learning rate
        writer.add_scalar("num_bad_epochs/lr_scheduler", scheduler.num_bad_epochs, epoch)
        writer.add_scalar("num_bad_epochs/end_of_training", scheduler_for_training_end.num_bad_epochs, epoch)

        if epoch % conf.hyper_params["val"]["n_epochs_val"] == 0:
            val_loss, metric_results = test(val_loader, model, conf)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                model_save_path = os.path.join(conf.DATA_DIR, "model_checkpoints", run_name, f"{epoch}_ckpt.pt")
                torch.save(model.state_dict(), model_save_path)
            writer.add_scalar(f"{conf.hyper_params['loss']}/val", val_loss, epoch)
            write_metric_results(metric_results, writer, epoch)
            del metric_results
        
        if scheduler_for_training_end.num_bad_epochs >= scheduler_for_training_end.patience:
            print(f"Restoring best weights of epoch {best_epoch}")
            model.load_state_dict(
                torch.load(os.path.join(conf.DATA_DIR, "model_checkpoints", run_name, f"{best_epoch}_ckpt.pt"))
            )
            break

        scheduler.step(metrics=val_loss)
        scheduler_for_training_end.step(metrics=val_loss)
        
    return model

            