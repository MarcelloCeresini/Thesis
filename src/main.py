import os
from datetime import datetime

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.data as pyg_data

from config_pckg.config_file import Config
from data_pipeline.data_loading import get_data_loaders
from train import train, test


conf = Config()

writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

# TODO: adjust for my use case
# TODO: adjust for my use case
# layout = {
#     "ABCDE": {
#         "loss": ["Multiline", ["loss/train", "loss/validation"]],
#         "accuracy": ["Multiline", ["accuracy/train", "accuracy/validation"]],
#     },
# }

# writer = SummaryWriter()
# writer.add_custom_scalars(layout)
# TODO: adjust for my use case
# TODO: adjust for my use case

print("Getting dataloaders")
train_dataloader, val_dataloader, test_dataloader = get_data_loaders(conf)

model = train(train_dataloader, val_dataloader, writer, conf)

test_loss, metric_results = test(test_dataloader, model, conf)
print(test_loss)
print(metric_results)

for 