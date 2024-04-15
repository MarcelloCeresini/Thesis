import wandb
from main import main

if __name__ == "__main__":

    sweep_config = {
        "method": "random"
    }

    metric = {
        "name": "metric",
        "goal": "minimize"
    }

    sweep_config["metric"] = metric

    parameters_dict = {
        "bool_radial_attributes": {
            "values": [True, False]
        },
        "PINN_mode": {
            "values": [
                "supervised_only", 
                "supervised_with_sampling", 
                "continuity_only", 
                "full_laminar", 
                "turbulent_kw"
        ]},
        "output_turbulence": {
            "values": [True, False]
        },
        "activation_for_max_normalized_features":{
            "values": [True, False]
        },
        "bool_bootstrap_bias": {
            "values": [True, False]
        },
        "general_sampling-use_sampling_weights": {
            "values": [True, False]
        },
        "standard_weights-supervised_on_sampled": {
            "values": [1, 10, 100]
        },
        "standard_weights-boundary": {
            "values": [1, 10, 100, 1000]
        },
        "standard_weights-continuity": {
            "values": [1, 10, 100, 1000]
        },
        "standard_weights-momentum_x": {
            "values": [1, 10, 100, 1000]
        },
        "standard_weights-momentum_y": {
            "values": [1, 10, 100, 1000]
        },
        "dynamic_loss_weights":{
            "values": [True, False],
            "probabilities": [0.1, 0.9]
        },
        "lambda_dynamic_weights":{
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.5,
        },
        "gamma_loss":{
            "values": [1, 5, 10],
        }
    }

    sweep_config["parameters"] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="Thesis")

    print(sweep_id)

    wandb.agent(sweep_id, function=main)