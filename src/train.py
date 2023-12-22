from typing import Literal
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.data as pyg_data
import wandb
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler 

from config_pckg.config_file import Config 
from models.models import GNNStack


def test(loader: pyg_data.DataLoader, model):
    with torch.no_grad():
        model.eval()
        for batch in loader:
            _, pred = model(batch)
            label = batch.y
            label_mask = batch.y_mask

            loss = model.loss(pred, label, label_mask)
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        return total_loss


def train(dataloader, writer: SummaryWriter(), conf: Config):

    # TODO: implement train/val/test splits
    # train_loader, val_loader, test_loader = 
    train_loader = dataloader

    # with wandb.init(**conf.get_logging_info()):

    # TODO: change the name "features_to_keep" to "labels_to_keep"
    model = GNNStack(input_dim=len(conf.graph_node_feature_dict), 
                     hidden_dim=20, 
                     output_dim=len(conf.features_to_keep))
    
    model.to(conf.device)

    # wandb.watch(
    #     model,
    #     log=conf.logging["model_log_mode"],
    #     log_freq=conf.logging["n_batches_freq"],
    #     log_graph=conf.logging["log_graph"]
    # )

    opt = Adam(
        params= model.parameters(), 
        lr= conf.hyper_params["training"]["lr"],
        weight_decay=conf.hyper_params["training"]["weight_decay"],
    )

    # TODO: implement schedules
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=conf.hyper_params["training"]["patience_reduce_lr"])

    for epoch in tqdm(range(conf.hyper_params["training"]["n_epochs"])):
        total_loss = 0
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            embedding, pred = model(batch)

            loss = model.loss(pred, batch.y, batch.y_mask)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(train_loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)

    #     if epoch % conf.hyper_params["training"]["n_epochs_val"]:
    #         val_loss = test(val_loader, model)
    #         writer.add_scalar("val_loss", val_loss, epoch)

    # test_metric = 0
    # test_loss = test(test_loader, model)
    # writer.add_scalar("test_loss", test_loss, epoch)

    # with open(os.path.join(config['training']['save_training_info_dir'], 
    #                         config['training']['base_stats_save_name']), 'wb') as f:
    #     pickle.dump(metrics_test, f)

        
    return model

            