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


from config_pckg.config_file import Config
from data_pipeline.data_loading import get_data_loaders
from train import train, test
from utils import plot_gt_pred_label_comparison, print_memory_state_gpu, \
    get_input_to_model, init_wandb, plot_test_images_from_model
from models.models import get_model_instance

def print_w_time(str):
    print(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} - {str}")


if __name__ == "__main__":

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
        torch.cuda.empty_cache()
        torch.autograd.set_detect_anomaly(True, True)

        conf = wandb.config

        run_name = wandb.run.dir.split(os.sep)[-2]
        pardir = os.path.join(conf.DATA_DIR, "model_runs")
        model_save_path = os.path.join(pardir, f"{run_name}.pt")

        if not os.path.isdir(pardir):
            os.mkdir(pardir)

        wandb.define_metric("train_loss", summary="min")
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("metric", summary="min")

        print_w_time("GETTING DATALOADERS")
        train_dataloader, val_dataloader, test_dataloader, dataloader_train_for_metrics = \
            get_data_loaders(conf)

        print_w_time("BUILDING MODEL")
        model = get_model_instance(conf)
        model.to(conf.device)

        print_w_time("WRITING GRAPH SUMMARY")
        for batch in val_dataloader:
            batch.to(conf["device"])
            break
        # writer.add_graph(model, input_to_model=input_to_model, use_strict_trace=False)
        model_summary = summary(model, **get_input_to_model(batch), leaf_module=None)

        print(model_summary)

        print_w_time("TRAINING")
        model = train(model, train_dataloader, val_dataloader, dataloader_train_for_metrics, conf, run_name)

        print_w_time("SAVING MODEL")
        if not os.path.isdir(os.path.join(conf["DATA_DIR"], "model_runs")):
            os.mkdir(os.path.join(conf["DATA_DIR"], "model_runs"))
        torch.save(model.state_dict(), model_save_path)
        model.eval()
        model.cpu()

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
            # for batch in tqdm(test_dataloader):
            #     pred_batch = model(**get_input_to_model(batch))

            #     if isinstance(pred_batch, tuple):
            #         residuals = pred_batch[1]
            #         pred_batch = pred_batch[0]

            #     for i in range(len(batch)):
            #         data = batch[i]
            #         pred = pred_batch[batch.ptr[i]:batch.ptr[i+1]]
            #         plot_gt_pred_label_comparison(data, pred, conf, run_name=run_name)
            #         # TODO: add residuals plot!
            #         # residuals = residuals[batch.ptr[i]:batch.ptr[i+1]]
            #         # plot_gt_pred_label_comparison(data, pred, conf, run_name=run_name, residuals=True)

        # FIXME: doesn't work(?)
        # artifact = wandb.Artifact(name="test_img_results", type="png")
        # artifact.add_dir(local_path=os.path.join(conf.test_imgs_comparisons, run_name))
        # wandb.run.log_artifact(artifact)

        # artifact = wandb.Artifact(name="test_html_results", type="html")
        # artifact.add_dir(local_path=os.path.join(conf.test_htmls_comparisons, run_name))
        # wandb.run.log_artifact(artifact)
