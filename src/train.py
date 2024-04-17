from copy import deepcopy
import os
from typing import Literal, Optional
from tqdm import tqdm
from datetime import datetime
import sys

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.data as pyg_data
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler 
from torch.masked import masked_tensor
from pandas import json_normalize
import torchmetrics
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook

from utils import print_memory_state_gpu, get_input_to_model, get_coefficients, normalize_label
from config_pckg.config_file import Config 
import loss_pckg



def clean_labels(batch, conf: Config):
    if hasattr(batch, "y_mask"):
        if batch.y_mask is not None:
            # FIXME: still no good support for masked tensors
            labels = torch.masked.masked_tensor(batch.y, batch.y_mask)
    else:
        labels = batch.y
    return labels[:,:conf["output_dim"]]


def forward_metric_results(preds, labels, conf, metric_dict):
    for metric_obj_subdict in metric_dict.values():
        for i, column in enumerate(conf["labels_to_keep_for_training"]):
                metric_obj_subdict[column].forward(preds[:,i], labels[:,i])
    return metric_dict


def compute_metric_results(metric_dict, conf):
    metric_results = metric_dict.copy() # just to copy the structure
    for metric_res_subdict, metric_obj_subdict in zip(metric_results.values(), metric_dict.values()):
        for column in conf["labels_to_keep_for_training"]:
            metric_res_subdict[column] = metric_obj_subdict[column].compute()
    return metric_results


def write_metric_results(metric_results, writer, epoch, split="val") -> None:
    # TODO: add also global writing when global is implemented
    for metric_name, metric_res_subdict in metric_results.items():
        if isinstance(metric_res_subdict, dict):
            for column, result in metric_res_subdict.items():
                writer.add_scalar(f"{metric_name}/{split}/{column}", result, epoch)
        else:
            writer.add_scalar(metric_name, metric_res_subdict, epoch)


def test(loader: pyg_data.DataLoader, model, conf, loss_weights: dict={}):
    metric_dict = deepcopy(conf["metric_dict"])
    for metric_name in metric_dict:
        metric_dict[metric_name] = {k:eval(v)() for k,v in metric_dict[metric_name].items()}
    metric_aero = eval(deepcopy(conf["metric_aero"]))(conf)

    device = list(model.parameters())[0].device

    with torch.no_grad():
        total_loss = 0
        total_loss_dict = {}
        total_optional_values = {}
        total_loss_dict_reweighted = {}

        model.eval()
        for batch in loader:
            batch.to(device, non_blocking=True)
            pred = model(**get_input_to_model(batch))
            labels = clean_labels(batch, model.conf)
            loss = model.loss(pred, labels, data=batch)

            if isinstance(loss, tuple):
                standard_loss = loss[0]
                loss_dict = {k:v*conf["standard_weights"].get(k,1) for k,v in loss[1].items()}
                optional_values = {k:v for k,v in loss[2].items()}

                for k in loss_dict: 
                    total_loss_dict[k] = total_loss_dict.get(k, 0) + \
                                            loss_dict[k].item()*batch.num_graphs
                for k in optional_values:
                    total_optional_values[k] = total_optional_values.get(k,0) + \
                                            optional_values[k].item()*batch.num_graphs
                if conf["dynamic_loss_weights"]:
                    loss = sum(loss_dict[k]*loss_weights.get(k,1) for k in loss_dict)
                else:
                    loss = sum(loss_dict[k] for k in loss_dict)
            
            total_loss += loss.item() * batch.num_graphs
            metric_dict = forward_metric_results(pred[0].cpu(), labels.cpu(), conf, metric_dict)
            
            ptr_num_sampled_boundary = torch.tensor([batch.num_boundary_sampling_points[:i].sum() 
                                                for i in range(batch.num_boundary_sampling_points.shape[0]+1)])
            
            
            for i in range(len(batch)):
                data = batch[i]
                
                if conf.DEBUG_BYPASS_MODEL:
                    UserWarning("YOU ARE NOT USING PREDICTIONS TO COMPUTE AERO METRICS")
                    pred = (labels, pred[1], 
                        tuple(normalize_label(data.y_additional[data.index_boundary_sampled, i].view(-1) , "x-velocity", conf)
                            for i in range(2,6)))

                pred_supervised_pts_pressure = pred[0][batch.ptr[i]:batch.ptr[i+1], 2]
                assert batch.ptr.shape[0] == 2, "Check derivatives for batch size higher than 1"
                if conf.flag_BC_PINN and conf.output_turbulence:
                    pred_vel_derivatives = torch.stack(
                        [p[ptr_num_sampled_boundary[i]:ptr_num_sampled_boundary[i+1]] for p in pred[2]])
                    pred_turb_values = pred[0][batch.ptr[i]:batch.ptr[i+1], 3:]
                    pred_coefficients = get_coefficients(conf, data, pred_supervised_pts_pressure, 
                        velocity_derivatives=pred_vel_derivatives, turbulent_values=pred_turb_values, 
                        denormalize=True, from_boundary_sampling=True)
                else:
                    pred_coefficients = get_coefficients(conf, data, pred_supervised_pts_pressure, denormalize=True)
                
                metric_aero.forward(pred=pred_coefficients,
                                    label=data.components_coefficients)
                
            batch.cpu()

            gettrace = getattr(sys, 'gettrace', None)
            if gettrace is not None:
                if gettrace():
                    print('Hmm, Big Debugger is watching me --> breaking in TEST')
                    break

        total_loss /= len(loader.dataset)
        for k in total_loss_dict: total_loss_dict[k] /= len(loader.dataset)
        for k in total_optional_values: total_optional_values[k] /= len(loader.dataset)
        if conf["dynamic_loss_weights"]:
            for k in total_loss_dict: 
                total_loss_dict_reweighted[k] = total_loss_dict[k]*loss_weights.get(k,1)
        
        metric_results = compute_metric_results(metric_dict, conf)
        metric_aero_dict = metric_aero.compute()
        
        # metric_results.update({"efficiency_MAPE/"+k:v for k,v in metric_aero_dict.items()})
        # metric_aero_dict_flattened = json_normalize(metric_aero_dict, sep="_").to_dict(orient="records")[0]
        # metric_results.update(metric_aero_dict_flattened)
        metric_results.update(metric_aero_dict)
        metric_results.update(total_loss_dict)
        metric_results.update(total_optional_values)
        metric_results.update(total_loss_dict_reweighted)

    return total_loss, metric_results


def train(
        model: torch.nn.Module,
        train_loader, 
        val_loader,
        dataloader_train_for_metrics,
        conf: Config,
        run_name: str,
        **kwargs):
    
    trigger_sync: Optional[TriggerWandbSyncHook] = kwargs.get("trigger_sync", None)

    if not os.path.isdir(os.path.join(conf["DATA_DIR"], "model_checkpoints")):
        os.mkdir(os.path.join(conf["DATA_DIR"], "model_checkpoints"))
    os.mkdir(os.path.join(conf["DATA_DIR"], "model_checkpoints", run_name))

    wandb.watch(
        model,
        log=conf["logging"]["model_log_mode"],
        log_freq=conf["logging"]["n_batches_freq"],
        log_graph=conf["logging"]["log_graph"]
    )

    opt = Adam(
        params = model.parameters(),
        lr = conf["hyper_params"]["training"]["lr"],
        weight_decay = conf["hyper_params"]["training"]["weight_decay"],
    )

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=opt, 
        patience=conf["hyper_params"]["training"]["patience_reduce_lr"],
        min_lr=conf["hyper_params"]["training"]["min_lr"],
        cooldown=conf["hyper_params"]["training"]["cooldown"],
        threshold=0, 
        threshold_mode='abs',
        mode="min",
    )

    scheduler_for_training_end = lr_scheduler.ReduceLROnPlateau(
        optimizer=opt,
        patience=conf["hyper_params"]["training"]["patience_end_training"],
        threshold=0, 
        threshold_mode='abs',
        mode="min",
    )

    best_epoch = 0
    # print_memory_state_gpu("Before training", conf)
    loss_weights = {}
    mean_grads = {}

    for epoch in tqdm(range(conf["hyper_params"]["training"]["n_epochs"]), desc="Epoch", position=0):
        total_loss = 0
        total_loss_dict = {}
        total_optional_values = {}
        total_loss_dict_reweighted = {}

        model.train()
        for batch in tqdm(train_loader, leave=False, desc="Batch in epoch", position=1):
            opt.zero_grad(set_to_none=True)
            
            batch.to(conf.device)
            pred = model(**get_input_to_model(batch))
            labels = clean_labels(batch, model.conf)
            loss = model.loss(pred, labels, batch)

            if isinstance(loss, tuple):
                standard_loss = loss[0]
                
                loss_dict = {k:v*conf.standard_weights.get(k,1) for k,v in loss[1].items()}
                optional_values = {k:v for k,v in loss[2].items()}
                
                for k in loss_dict: 
                    total_loss_dict[k] = total_loss_dict.get(k, 0) + \
                                            loss_dict[k].item()*batch.num_graphs # set it with initialization = 0
                
                for k in optional_values: 
                    total_optional_values[k] = total_optional_values.get(k,0) + \
                                            optional_values[k].item()*batch.num_graphs
                # TODO: do it each BATCH? or each EPOCH? in any case, log each EPOCH?
                if conf.dynamic_loss_weights:
                    assert conf.main_loss_component_dynamic in loss_dict, f"{conf.main_loss_component_dynamic} not in loss_dict keys: {list(loss_dict.keys())}"
                    for k in loss_dict:
                        loss_dict[k].backward(retain_graph=True)
                        # for param in model.named_parameters():
                        #     print(f"True - {param[0]}" if isinstance(param[1].grad, torch.Tensor) else f"False - {param[0]}")
                        mean_grads[k] = mean_grads.get(k, 0) + \
                                            torch.cat([param.grad.view(-1) for param in model.parameters() 
                                                if isinstance(param.grad, torch.Tensor)]).abs().mean()
                        opt.zero_grad()
                    # TODO: could store grads and compute differences between them to avoid the last
                    # backward pass with the total loss (because we are summing them)
                    loss = sum(loss_dict[k]*loss_weights.get(k,1) for k in loss_dict)
                else:
                    loss = sum(loss_dict[k] for k in loss_dict)
            
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=conf.gradient_clip_value, foreach=True)
            opt.step()
            batch.cpu()

            if torch.isnan(loss):
                print(loss_dict)

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).sum() > 0:
                        print(f"NaN - {name}")
                        input("Press something to go onwards")

            total_loss += loss.item() * batch.num_graphs
            gettrace = getattr(sys, 'gettrace', None)
            if gettrace is not None:
                if gettrace():
                    print('Hmm, Big Debugger is watching me --> breaking in TRAIN')
                    break

        total_loss /= len(train_loader.dataset)
        for k in total_loss_dict: total_loss_dict[k] /= len(train_loader.dataset)
        for k in total_optional_values: total_optional_values[k] /= len(train_loader.dataset)

        log_dict = {
            "standard_loss": standard_loss,
            "train_loss": total_loss,
            "lr": opt.param_groups[0]["lr"],
            "epoch": epoch,
            "num_bad_epochs/lr_scheduler": scheduler.num_bad_epochs,
            "num_bad_epochs/end_of_training": scheduler_for_training_end.num_bad_epochs,
        }
        log_dict.update(total_loss_dict)
        log_dict.update(total_optional_values)

        wandb.log(log_dict, epoch)
        
        if trigger_sync is not None:
            trigger_sync()
        
        # torch.cuda.empty_cache()
        if epoch % conf["hyper_params"]["val"]["n_epochs_val"] == 0:
            val_loss, metric_results = test(val_loader, model, conf, loss_weights)
            torch.cuda.empty_cache()

            metric = sum([metric_results["MAE"][k] for k in conf.physical_labels])
            if metric < scheduler.best:
                best_epoch = epoch
                
                model_save_path = os.path.join(conf.DATA_DIR, "model_checkpoints", run_name, f"{epoch}_ckpt_opt.pt")
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    }, model_save_path)
                
                # torch.save(model.state_dict(), model_save_path)
            
            wandb.log({"val_loss": val_loss}, epoch)
            wandb.log({"metric": metric}, epoch)
            wandb.log({"best_metric": scheduler.best}, epoch)
            wandb.log({f"val_{k}":v for k,v in metric_results.items()}, epoch)

            train_standard_loss, train_metric_results = test(dataloader_train_for_metrics, model, conf, loss_weights)
            train_standard_metric = sum([metric_results["MAE"][k] for k in conf.physical_labels])

            wandb.log({"train_standard_loss": train_standard_loss}, epoch)
            wandb.log({"train_standard_metric": train_standard_metric}, epoch)
            wandb.log({f"train_standard_{k}":v for k,v in train_metric_results.items()}, epoch)

        # scheduler.step(metrics=val_loss)
        # scheduler_for_training_end.step(metrics=val_loss)
        scheduler.step(metrics=metric)
        scheduler_for_training_end.step(metrics=metric)
        
        if scheduler_for_training_end.num_bad_epochs >= scheduler_for_training_end.patience:
            print(f"Restoring best weights of epoch {best_epoch}")
            tmp = torch.load(os.path.join(conf.DATA_DIR, "model_checkpoints", run_name, f"{best_epoch}_ckpt_opt.pt"))
            model.load_state_dict(tmp["model_state_dict"])
            break

        if conf.dynamic_loss_weights: 
            for k in total_loss_dict: 
                total_loss_dict_reweighted[k] = total_loss_dict[k]*loss_weights.get(k,1)
            wandb.log({f"weight_{k}":v for k,v in loss_weights.items()}, epoch)
            wandb.log({f"reweighted_{k}":v for k,v in total_loss_dict_reweighted.items()}, epoch)
            for k in loss_dict:
                loss_weights[k] = (1-conf.lambda_dynamic_weights) * loss_weights.get(k, conf.standard_weights.get(k, 1)/conf.gamma_loss) + \
                                    conf.lambda_dynamic_weights/conf.gamma_loss * (mean_grads[conf.main_loss_component_dynamic]/mean_grads[k])

    return model