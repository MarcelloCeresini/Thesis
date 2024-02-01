import glob
import os
import pickle
import utils
from tqdm import tqdm

import torch
from torch_geometric.data import Data
# import torchinfo
from torch_geometric.nn import summary

from config_pckg.config_file import Config
from data_pipeline.data_loading import get_data_loaders
from utils import convert_mesh_complete_info_obj_to_graph, plot_gt_pred_label_comparison
from models.models import BaselineModel


if __name__ == "__main__":
    conf = Config()

    print("Getting dataloaders")
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(
        conf, load_from_disk=True)

    # model = BaselineModel(conf)
    # model.load_state_dict(torch.load(TODO:))
    # model.eval()
    
    # for batch in test_dataloader:
    #     pred_batch = model(batch)
    #     for i in range(len(batch)):
    #         data = batch[i]
    #         pred = pred_batch[batch.ptr[i]:batch.ptr[i+1], :]
    #         plot_gt_pred_label_comparison(data, pred, conf)

    for batch in test_dataloader:
        break

    model = BaselineModel(conf.input_dim, conf.output_dim, conf.model_structure, conf)
    description = summary(model, batch)
    
    print(description)