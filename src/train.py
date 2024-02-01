from copy import deepcopy
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



from config_pckg.config_file import Config 
from models.models import BaselineModel


def clean_labels(batch, conf: Config):
    if hasattr(batch, "y_mask"):
        if batch.y_mask is not None:
            # FIXME: still no good support for masked tensors
            labels = torch.masked.masked_tensor(batch.y, batch.y_mask)
    else:
        labels = batch.y
    return labels[:,:len(conf.labels_to_keep_for_training)]


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
            input_to_model = batch.x, batch.x_mask, batch.edge_index, batch.edge_attr, batch.batch
            pred = model(*input_to_model)
            
            labels = clean_labels(batch, model.conf)
            loss = model.loss(pred, labels)
            total_loss += loss.item() * batch.num_graphs
            metric_dict = forward_metric_results(pred, labels, conf, metric_dict)

        total_loss /= len(loader.dataset)
        metric_results = compute_metric_results(metric_dict, conf)

    return total_loss, metric_results


def train(
        model,
        train_loader, 
        val_loader, 
        writer: SummaryWriter, 
        conf: Config):

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

    # TODO: implement schedules
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=opt, 
        patience=conf.hyper_params["training"]["patience_reduce_lr"],
        min_lr=conf.hyper_params["training"]["min_lr"]
    )
    best_loss = 1000
    last_best_epoch = 0

    for epoch in tqdm(range(conf.hyper_params["training"]["n_epochs"]), desc="Epoch", position=0):
        total_loss = 0
        model.train()
        for batch in tqdm(train_loader, leave=False, desc="Batch in epoch", position=1):
            opt.zero_grad()
            input_to_model = batch.x, batch.x_mask, batch.edge_index, batch.edge_attr, batch.batch
            pred = model(*input_to_model)
            # pred = model(batch)
            # FIXME: change masked tensor with manual mask (slow implementation by pytorch)
            labels = clean_labels(batch, model.conf)
            loss = model.loss(pred, labels)

            loss.backward()
            opt.step()

            total_loss += loss.item() * batch.num_graphs

        total_loss /= len(train_loader.dataset)
        writer.add_scalar("loss/train", total_loss, epoch)

        if epoch % conf.hyper_params["val"]["n_epochs_val"] == 0:
            val_loss, metric_results = test(val_loader, model, conf)

            scheduler.step(val_loss)

            # TODO: complete model checkpoins and "save best"
            # if val_loss < best_loss:
            #     best_loss = val_loss
            #     model_save_path = os.path.join(conf.DATA_DIR, "model_checkpoints", f"{last_best_epoch}ckpt_{run_name}.pt")
            #     torch.save(model.state_dict(), model_save_path)
            writer.add_scalar("loss/val", val_loss, epoch)
            write_metric_results(metric_results, writer, epoch)


    # with open(os.path.join(config['training']['save_training_info_dir'], 
    #                         config['training']['base_stats_save_name']), 'wb') as f:
    #     pickle.dump(metrics_test, f)

        
    return model

            