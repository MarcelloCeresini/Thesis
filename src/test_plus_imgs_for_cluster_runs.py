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

import glob, gc

gc.enable()

runs_to_recover = [
    # "568h6oc2", # elated-pine-110
    # "qmzxcn8q", 
    "updyeabh", 
    # "pe5g6emv",
    # "zcwgrsl9",
    # "ehu8ya0l",
    # "xlqu35is",
    # "p1nbcy4p",
    # "28ig2vzi",
    # "fn0w9rb0",
    # "iez5i4fg",
    # "8mdt9lh0",
    # "qqxymndp",
    # ##### new comparisons finished
    # "m7bmm9cj",
    # "mormtj9e",
    # "cbf38gv6",
    # "peh0csfv",
    # "iugv186w",
    # "0egkpy0z",
    # "kylpiqqd",
    # "htb6vv54",
    # "1fagc6rx",
    # "0az598nq",
    # "27v3gvl0",
    # "ekffajd8",
    # "sxaimrne",
    # "pnyash2s",
    # "o8xmai4x",
    # "zt1jjmmp",
    # "7rqymg98",
    # "hm4nkv8d",
]

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# CLUSTER_DATA_DIR = os.path.join("H:","CFD-RD_SOLVER", "marcello", "Thesis-main", "data")
# EXTERNAL_FOLDER = os.path.join("K:", "CFD-WT", "ML_AI", "2D_V01_database")
# EXTERNAL_FOLDER_MESHCOMPLETE_W_LABELS = os.path.join(EXTERNAL_FOLDER, "MeshCompleteObjsWithLabels_at300")

test_imgs_comparisons = os.path.join(DATA_DIR, "test_imgs_comparisons")
test_htmls_comparisons = os.path.join(DATA_DIR, "test_htmls_comparisons")
test_vtk_comparisons = os.path.join(DATA_DIR, "test_vtk_comparisons")

test_dataloader = None

if __name__ == "__main__":
    
    for run_id in runs_to_recover:
        try:
            wandb.init(
                project = "Thesis",
                id = run_id,
                resume="must",
            )
            conf = wandb.config
            
            conf.update({
                "device": "cpu",
                "standard_dataset_dir": os.path.join(DATA_DIR, "dataset_files"),
                # "EXTERNAL_FOLDER_MESHCOMPLETE_W_LABELS": EXTERNAL_FOLDER_MESHCOMPLETE_W_LABELS,
                "test_htmls_comparisons": test_htmls_comparisons,
                "test_imgs_comparisons": test_imgs_comparisons,
                "test_vtk_comparisons": test_vtk_comparisons,
                }, allow_val_change=True)
            
            if test_dataloader is None:
                _, _, test_dataloader, _ = \
                    get_data_loaders(conf, from_cluster=False)
                
            model = get_model_instance(conf)
            model.to(conf.device)
            
            for batch in test_dataloader:
                batch.to(conf["device"])
                pred = model(**get_input_to_model(batch))
                labels = clean_labels(batch, model.conf)
                loss = model.loss(pred, labels, batch)
                break

            loss_dict = loss[1]
            loss_keys_list = [k for k in loss_dict]
            
            ###################################################################################################
            model_summary = summary(model, **get_input_to_model(batch), leaf_module=None)
            print(model_summary)
            
            run_name = wandb.run.dir.split(os.sep)[-2]
            # pardir = os.path.join(conf.DATA_DIR, "model_runs")
            # model_save_path = os.path.join(pardir, f"{run_name}.pt")
            ckpts_dir = os.path.join(CLUSTER_DATA_DIR, "model_checkpoints")
            ckpt_dir = glob.glob(f"*{run_id}", root_dir=ckpts_dir)[0]
            ckpts = glob.glob("*.pt", root_dir=os.path.join(ckpts_dir, ckpt_dir))
            best_ckpt = sorted(list(map(lambda x: int(x.removesuffix("_ckpt_opt.pt")), ckpts)))[-1]
            
            ckpt_path = os.path.join(ckpts_dir, ckpt_dir, f"{best_ckpt}_ckpt_opt.pt")
            model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
            
            with torch.no_grad():
                plot_test_images_from_model(conf, model, run_id, test_dataloader)
                
                test_loss, metric_results = test(test_dataloader, model, conf)
                print(f"Test loss: {test_loss}")
                # print(f"Test metrics: {metric_results}")

                final_metric = sum([metric_results["MAE"][k] for k in conf["physical_labels"]])
                metric_results = {f"test_{k}":v for k,v in metric_results.items()}
                metric_results.update({"test_loss":test_loss})
                metric_results.update({"test_metric":final_metric})

                wandb.log(metric_results)
            
            model = None
            
        except Exception as e:
            print("!!!EXCEPTION!!!", e)
        finally:
            wandb.finish()
            gc.collect()
            