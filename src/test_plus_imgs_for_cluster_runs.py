import pickle, sys, os
from datetime import datetime
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.data as pyg_data
from torch_geometric.nn import summary
from pandas import json_normalize
from tqdm import tqdm
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook


from config_pckg.config_file import Config
from data_pipeline.data_loading import get_data_loaders
from train import train, test
from utils import clean_labels, plot_gt_pred_label_comparison, print_memory_state_gpu, \
    get_input_to_model, init_wandb, plot_test_images_from_model, TriggerWandbSyncHookForWindows
from models.models import get_model_instance

import glob


runs_to_recover = [
    "568h6oc2", # elated-pine-110
    
]


if __name__ == "__main__":
    
    for run_id in runs_to_recover:
        try:
            wandb.init(
                project = "Thesis",
                run_id = run_id,
                resume="must",
            )
            conf = wandb.config
            
            conf.device = "cpu"
            
            train_dataloader, val_dataloader, test_dataloader, dataloader_train_for_metrics = \
                get_data_loaders(conf)
                
            model = get_model_instance(conf)
            model.to(conf.device)
            
            run_name = wandb.run.dir.split(os.sep)[-2]
            pardir = os.path.join(conf.DATA_DIR, "model_runs")
            model_save_path = os.path.join(pardir, f"{run_name}.pt")
            ckpt_dir = os.path.join(conf.DATA_DIR, "model_checkpoints", run_name)
            ckpts = glob.glob("*.pt", root_dir=ckpt_dir)
            for i in len(ckpts): ckpts[i] = int(ckpts[i].removesuffix(".pt"))
            
            ckpt_path = os.path.join(ckpt_dir, f"{sorted(ckpts)[-1]}.pt")
            model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
            
            with torch.no_grad():
                test_loss, metric_results = test(test_dataloader, model, conf)
                print(f"Test loss: {test_loss}")
                print(f"Test metrics: {metric_results}")

                final_metric = sum([metric_results["MAE"][k] for k in conf["physical_labels"]])
                metric_results = {f"test_{k}":v for k,v in metric_results.items()}
                metric_results.update({"test_loss":test_loss})
                metric_results.update({"test_metric":final_metric})

                wandb.log(metric_results)

                plot_test_images_from_model(conf, model, run_name, test_dataloader)
            
        except Exception as e:
            print(e)
        finally:
            wandb.finish()
            