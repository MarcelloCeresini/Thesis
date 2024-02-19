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

from config_pckg.config_file import Config
from data_pipeline.data_loading import get_data_loaders
from train import train, test
from utils import plot_gt_pred_label_comparison, print_memory_state_gpu, get_input_to_model
from models.models import get_model_instance

def print_w_time(str):
    print(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} - {str}")


if __name__ == "__main__":

    for msg_passing_steps in tqdm([1, 3, 5, 7, 10, 15]):
        print_w_time(f"Starting training with {msg_passing_steps} message passing steps")
        conf = Config()

        full_conf = conf.get_tensorboard_logging_info()

        ###################
        full_conf["model"]["message_passer"]["repeats_training"] = msg_passing_steps
        ###################

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

        writer = SummaryWriter(f"./log/{run_name}")
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
        train_dataloader, val_dataloader, test_dataloader = get_data_loaders(
            conf, 
            save_to_disk=False,
            load_from_disk=True,
        )
        print_memory_state_gpu("After DataLoaders", conf)

        print_w_time("BUILDING MODEL")
        model = get_model_instance(full_conf)
        model.to(conf.device)
        print_memory_state_gpu("After Model.cuda()", conf)

        print_w_time("WRITING GRAPH SUMMARY")
        for batch in train_dataloader:
            batch.to(conf.device)
            break
        # writer.add_graph(model, input_to_model=input_to_model, use_strict_trace=False)
        model_summary = summary(model, **get_input_to_model(batch), leaf_module=None)
        writer.add_text("Model summary", "<pre>"+model_summary+"</pre>")
        print(model_summary)
        print_memory_state_gpu("After model summary", conf)

        print_w_time("TRAINING")
        model = train(model, train_dataloader, val_dataloader, writer, conf)

        print_w_time("SAVING MODEL")
        model_save_path = os.path.join(conf.DATA_DIR, "model_runs", f"{run_name}.pt")
        torch.save(model.state_dict(), model_save_path)
        model.eval()
        model.cpu()

        with torch.no_grad():
            test_loss, metric_results = test(test_dataloader, model, conf)
            print(f"Test loss: {test_loss}")
            print(f"Test metrics: {metric_results}")

            # TODO: save to wandb all the images of the test set for all trials that reach this point

            metric_dict = {}
            for metric_name, metric_res_subdict in metric_results.items():
                if isinstance(metric_res_subdict, dict):
                    for column, result in metric_res_subdict.items():
                        metric_dict[f"test/{metric_name}/{column}"] = result
                else:
                    metric_dict[f"test/{metric_name}"] = metric_res_subdict

            # metric_dict = {
            #         f"test/{metric_name}/{label_name}": label_metric_value
            #             for metric_name, labels_dict in metric_results.items()
            #                 for label_name, label_metric_value in labels_dict.items()
            #     }

            metric_dict[f"test/{full_conf['hyperparams']['loss']}"]= test_loss

            writer.add_hparams(
                hparam_dict=hparams_flattened,
                metric_dict=metric_dict
            )

            for batch in tqdm(test_dataloader):
                pred_batch = model(**get_input_to_model(batch))
                for i in range(len(batch)):
                    data = batch[i]
                    pred = pred_batch[batch.ptr[i]:batch.ptr[i+1], :]
                    plot_gt_pred_label_comparison(data, pred, conf, run_name=run_name)
