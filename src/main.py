import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.data as pyg_data
from torch_geometric.nn import summary
from pandas import json_normalize

from config_pckg.config_file import Config
from data_pipeline.data_loading import get_data_loaders
from train import train, test
from utils import plot_gt_pred_label_comparison
from models.models import BaselineModel

def print_w_time(str):
    print(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} - {str}")

conf = Config()

full_conf = conf.get_tensorboard_logging_info()
hparams_flattened = json_normalize(full_conf).to_dict(orient="records")[0]

### TENSORBOARD SETUP ####
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
print_w_time(f"Starting run: {run_name}")

writer = SummaryWriter(f"./log/{run_name}")
splits = ["train", "val"]
layout = {
    "Standard Layout": {
        metric_name: ["Multiline", 
            [f"{metric_name}/val/{col}" for col in conf.labels_to_keep_for_training]] 
                for metric_name in conf.metrics
    }
}
layout["Standard Layout"]["loss"] = ["Multiline", [f"loss/{s}" for s in splits]]
writer.add_custom_scalars(layout)
### TENSORBOARD SETUP ####

print_w_time("Getting dataloaders")
train_dataloader, val_dataloader, test_dataloader = get_data_loaders(conf, load_from_disk=True)

print_w_time("Building model")
model = eval(conf.model_structure["name"])(
    input_dim = conf.input_dim,
    output_dim = conf.output_dim,
    model_structure = conf.model_structure,
    conf = conf
).to(conf.device)

print_w_time("Writing graph and summary")
for batch in test_dataloader:
    break
input_to_model = (batch.x, batch.x_mask, batch.edge_index, batch.edge_attr, batch.batch)
writer.add_graph(model, input_to_model=input_to_model, use_strict_trace=False)
writer.add_text("Model summary", "<pre>"+summary(model, *input_to_model)+"</pre>")


print_w_time("Training")
model = train(model, train_dataloader, val_dataloader, writer, conf)

print_w_time("Saving model")
model_save_path = os.path.join(conf.DATA_DIR, "model_runs", f"{run_name}.pt")
torch.save(model.state_dict(), model_save_path)
model.eval()

with torch.no_grad():
    test_loss, metric_results = test(test_dataloader, model, conf)
    print(f"Test loss: {test_loss}")
    print(f"Test metrics: {metric_results}")

    # TODO: save to wandb all the images of the test set for all trials that reach this point

    writer.add_hparams(
        hparam_dict=hparams_flattened,
        metric_dict={
            f"test/{metric_name}/{label_name}": label_metric_value
                for metric_name, labels_dict in metric_results.items()
                    for label_name, label_metric_value in labels_dict.items()
        },
        run_name=run_name)

