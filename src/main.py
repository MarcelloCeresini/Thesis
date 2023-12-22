import os
from datetime import datetime

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.data as pyg_data

from config_pckg.config_file import Config
from data_pipeline.data_loading import get_data_loaders
from train import train


conf = Config()

writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

dataloader = get_data_loaders(conf)

model = train(dataloader, writer, conf)

print("AAA")