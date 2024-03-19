import os
from datetime import datetime
import pickle

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
from utils import plot_gt_pred_label_comparison, print_memory_state_gpu, get_input_to_model
from models.models import get_model_instance

def print_w_time(str):
    print(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} - {str}")


WANDB_FLAG = True

if __name__ == "__main__":

    # for msg_passing_steps in tqdm([1, 3, 5, 7, 10, 15]):
        # print_w_time(f"Starting training with {msg_passing_steps} message passing steps")
        conf = Config()
        full_conf = conf.get_logging_info()
        torch.cuda.empty_cache()
        torch.autograd.set_detect_anomaly(True, True)
        ###################
        # full_conf["model"]["message_passer"]["repeats_training"] = msg_passing_steps
        ###################

        if WANDB_FLAG:
            wandb.init(
                project="Thesis",
                config=full_conf,
            )
            run_name = wandb.run.dir.split(os.sep)[-2]
            pardir = os.path.join(conf.DATA_DIR, "model_runs")
            model_save_path = os.path.join(pardir, f"{run_name}.pt")
            if not os.path.isdir(pardir):
                os.mkdir(pardir)
            torch.save(full_conf, model_save_path.split(".")[0]+"_full_conf.pkl")

            wandb.define_metric("train_loss", summary="min")
            wandb.define_metric("val_loss", summary="min")

        else:
            hparams_flattened = json_normalize(full_conf).to_dict(orient="records")[0]
            c=0
            while not all([
                        (isinstance(v, (int, str, bool, torch.Tensor, float)) or (v is None)) 
                            for v in hparams_flattened.values()]):
                keys_to_pop = []
                new_dict = {}
                for k, v in hparams_flattened.items():
                    if isinstance(v, list):
                        for i, element in enumerate(v):
                            new_dict[k+"_"+str(i)] = element
                        keys_to_pop.append(k)
                [hparams_flattened.pop(k) for k in keys_to_pop]
                hparams_flattened.update(new_dict)
                hparams_flattened = json_normalize(hparams_flattened).to_dict(orient="records")[0]
                c+=1
                print(c)
                assert c<10, "Too many cycles"

                ### TENSORBOARD SETUP ####
                run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
                print_w_time(f"Starting run: {run_name}")

                if not os.path.isdir(os.path.join(conf.ROOT_DIR, "log")):
                    os.mkdir(os.path.join(conf.ROOT_DIR, "log"))

                writer = SummaryWriter(os.path.join(conf.ROOT_DIR, "log", run_name))
                with open(os.path.join(conf.ROOT_DIR, "log", run_name, "full_conf.pkl"), "wb") as f:
                    pickle.dump(full_conf, f)

                splits = ["train", "val"]
                layout = {
                    "Standard Layout": {
                        metric_name: ["Multiline", 
                            [f"{metric_name}/val/{col}" for col in conf.labels_to_keep_for_training]] 
                                for metric_name in conf.metrics
                    }
                }
                layout["Standard Layout"]["efficiency_MAPE"] = ["Multiline", ["efficiency_MAPE/flap", "efficiency_MAPE/tyre", "efficiency_MAPE/car"]]
                layout["Standard Layout"][f"{full_conf['hyperparams']['loss']}"] = \
                    ["Multiline", [f"{full_conf['hyperparams']['loss']}/{s}" for s in splits]]
                layout["Standard Layout"]["lr"] = ["Multiline", ["lr"]]
                layout["Standard Layout"]["epoch"] = ["Multiline", ["epoch"]]
                layout["Standard Layout"]["num_bad_epochs"] = ["Multiline", ["num_bad_epochs/lr_scheduler", "num_bad_epochs/end_of_training"]]
                layout["Standard Layout"]["GPU_occ"] = ["Multiline", ["GPU_occ/allocated", "GPU_occ/reserved"]]

                writer.add_custom_scalars(layout)
            ### TENSORBOARD SETUP END ####

        print_w_time("GETTING DATALOADERS")
        train_dataloader, val_dataloader, test_dataloader = \
            get_data_loaders(conf, n_workers=0)
        print_memory_state_gpu("After DataLoaders", conf)

        print_w_time("BUILDING MODEL")
        model = get_model_instance(full_conf)
        model.to(conf.device)
        print_memory_state_gpu("After Model.cuda()", conf)

        print_w_time("WRITING GRAPH SUMMARY")
        for batch in val_dataloader:
            batch.to(conf.device)
            break
        # writer.add_graph(model, input_to_model=input_to_model, use_strict_trace=False)
        model_summary = summary(model, **get_input_to_model(batch), leaf_module=None)
        if not WANDB_FLAG:
            writer.add_text("Model summary", "<pre>"+model_summary+"</pre>")

        print(model_summary)
        print_memory_state_gpu("After model summary", conf)

        print_w_time("TRAINING")
        if WANDB_FLAG:
            model = train(model, train_dataloader, val_dataloader, True, conf)
        else:
            model = train(model, train_dataloader, val_dataloader, writer, conf)

        print_w_time("SAVING MODEL")
        if not os.path.isdir(os.path.join(conf.DATA_DIR, "model_runs")):
            os.mkdir(os.path.join(conf.DATA_DIR, "model_runs"))
        torch.save(model.state_dict(), model_save_path)
        # TODO: save model as artifact on wandb
        model.eval()
        model.cpu()

        with torch.no_grad():
            test_loss, metric_results = test(test_dataloader, model, conf)
            print(f"Test loss: {test_loss}")
            print(f"Test metrics: {metric_results}")

            if WANDB_FLAG:
                metric_results = {f"test_{k}":v for k,v in metric_results.items()}
                metric_results.update({"test_loss":test_loss})
                final_metric = sum(metric_results["MAE"].values())
                metric_results.update({"test_metric":final_metric})

                wandb.log(metric_results)
            else:
                metric_dict = {}
                for metric_name, metric_res_subdict in metric_results.items():
                    if isinstance(metric_res_subdict, dict):
                        for column, result in metric_res_subdict.items():
                            metric_dict[f"test/{metric_name}/{column}"] = result
                    else:
                        metric_dict[f"test/{metric_name}"] = metric_res_subdict

                metric_dict[f"test/{full_conf['hyperparams']['loss']}"]= test_loss

                writer.add_hparams(
                    hparam_dict=hparams_flattened,
                    metric_dict=metric_dict
                )

            for batch in tqdm(test_dataloader):
                pred_batch = model(**get_input_to_model(batch))

                if isinstance(pred_batch, tuple):
                    residuals = pred_batch[1]
                    pred_batch = pred_batch[0]

                for i in range(len(batch)):
                    data = batch[i]
                    pred = pred_batch[batch.ptr[i]:batch.ptr[i+1], :]
                    plot_gt_pred_label_comparison(data, pred, conf, run_name=run_name)

        # FIXME: doesn't work(?)
        # artifact = wandb.Artifact(name="test_img_results", type="png")
        # artifact.add_dir(local_path=os.path.join(conf.test_imgs_comparisons, run_name))
        # wandb.run.log_artifact(artifact)

        # artifact = wandb.Artifact(name="test_html_results", type="html")
        # artifact.add_dir(local_path=os.path.join(conf.test_htmls_comparisons, run_name))
        # wandb.run.log_artifact(artifact)
