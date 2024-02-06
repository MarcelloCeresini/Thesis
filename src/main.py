import os
from datetime import datetime
import pickle

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.data as pyg_data
from torch_geometric.nn import summary
from pandas import json_normalize

from config_pckg.config_file import Config
from data_pipeline.data_loading import get_data_loaders
from train import train, test
from utils import plot_gt_pred_label_comparison, print_memory_state_gpu
from models.models import get_model_instance

def print_w_time(str):
    print(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} - {str}")

conf = Config()

full_conf = conf.get_tensorboard_logging_info()
hparams_flattened = json_normalize(full_conf).to_dict(orient="records")[0]

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
layout["Standard Layout"][f"{full_conf['hyperparams']['loss']}"] = \
    ["Multiline", [f"{full_conf['hyperparams']['loss']}/{s}" for s in splits]]
layout["Standard Layout"]["lr"] = ["Multiline", ["lr"]]
layout["Standard Layout"]["epoch"] = ["Multiline", ["epoch"]]
layout["Standard Layout"]["num_bad_epochs"] = ["Multiline", ["num_bad_epochs/lr_scheduler", "num_bad_epochs/end_of_training"]]

writer.add_custom_scalars(layout)
### TENSORBOARD SETUP ####

print_w_time("Getting dataloaders")
train_dataloader, val_dataloader, test_dataloader = get_data_loaders(conf, load_from_disk=True)
print_memory_state_gpu("After DataLoaders", conf)

print_w_time("Building model")
model = get_model_instance(full_conf)
model.to(conf.device)
print_memory_state_gpu("After Model.cuda()", conf)

print_w_time("Writing graph and summary")
for batch in train_dataloader:
    break
input_to_model = (batch.x, batch.x_mask, batch.edge_index, batch.edge_attr, batch.batch)
# writer.add_graph(model, input_to_model=input_to_model, use_strict_trace=False)
model_summary = summary(model, *input_to_model, leaf_module=None)
writer.add_text("Model summary", "<pre>"+model_summary+"</pre>")
print(model_summary)
print_memory_state_gpu("After model summary", conf)

print_w_time("Training")
model = train(model, train_dataloader, val_dataloader, writer, conf)

print_w_time("Saving model")
model_save_path = os.path.join(conf.DATA_DIR, "model_runs", f"{run_name}.pt")
torch.save(model.state_dict(), model_save_path)
model.eval()
model.cpu()

with torch.no_grad():
    test_loss, metric_results = test(test_dataloader, model, conf)
    print(f"Test loss: {test_loss}")
    print(f"Test metrics: {metric_results}")

    # TODO: save to wandb all the images of the test set for all trials that reach this point

    metric_dict = {
            f"test/{metric_name}/{label_name}": label_metric_value
                for metric_name, labels_dict in metric_results.items()
                    for label_name, label_metric_value in labels_dict.items()
        }

    metric_dict.update({f"test/{full_conf['hyperparams']['loss']}": test_loss})

    writer.add_hparams(
        hparam_dict=hparams_flattened,
        metric_dict=metric_dict
    )

