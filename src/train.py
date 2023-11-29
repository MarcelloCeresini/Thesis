import wandb
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler 

from config_pckg.config_file import Config as Cfg
from data_pipeline import get_data_loaders


if __name__ == "__main__":

    cfg = Cfg()

    ROOT_DIR = cfg.ROOT_DIR

    train_loader, val_loader, test_loader = get_data_loaders(cfg)

    with wandb.init(**cfg.get_logging_info()):

        model = Model()
        model.to(cfg.device)

        wandb.watch(
            model,
            log=cfg.logging["model_log_mode"],
            log_freq=cfg.logging["n_batches_freq"],
            log_graph=cfg.logging["log_graph"]
        )

        optimizer = Adam(
            params=[{
                "params": None, # model.module1.parameters()
                "lr": cfg.hyper_params["training"]["module1"]["lr"]
            },{
                "params": None, # model.module2.parameters()
                "lr": cfg.hyper_params["training"]["module2"]["lr"]
            },],
            weight_decay=cfg.hyper_params["training"]["weight_decay"]
        )

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.hyper_params["training"]["patience_reduce_lr"])

        model = train_loop( model,
                            cfg,
                            training_loader=dataset_base_train,
                            validation_loader=dataset_base_val,
                            optimizer=optimizer_base,
                            scheduler=scheduler_base,
                            device=device)

            # Evaluation also on base test dataset
            metrics_test = Evaluate(
                model, 
                dataset_base_test, 
                prefix="test/",
                device=device,
                config=config,
                confidence_threshold=config['eval']['threshold_classification_scores']
            )(is_novel=False)

            # TODO: add timestamp so that it doesn't overwrite the same metrics over and over
            with open(os.path.join(config['training']['save_training_info_dir'], 
                                   config['training']['base_stats_save_name']), 'wb') as f:
                pickle.dump(metrics_test, f)