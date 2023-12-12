from enum import Enum
import os
from typing import Annotated, Dict, Literal, Optional, TypedDict, Union
from annotated_types import Gt

import yaml
from pydantic import BaseModel

class Config():

    def __init__(self, mode=None):
        '''
        Create a config class. To make it more readable, all fixed hyperparameters can be stored in a YAML file
        To change only some parameters between runs, we use a match case
        '''
        self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.DATA_DIR = os.path.join(self.ROOT_DIR, "data")
        self.PROJECT_NAME = os.path.basename(self.ROOT_DIR)
        self.device: Literal["cpu", "cuda"] = "cpu"

        with open(os.path.join(self.ROOT_DIR, "src", "config_pckg", "hyperparams.yaml"), "r") as stream:
            try:
                self.hyper_params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        match mode:
            case None:
                pass
            case _:
                raise NotImplementedError()

        model_logging = ModelLogging(
            model_log_mode= "all",
            n_batches_freq=0,
            log_graph=False
        )


    def get_wandb_logging_info(self) -> Dict:
        '''
        Return a Dict with the information for wandb logging and the hyperparameters that are important to log
        '''
        return_dict = {
            "project": self.PROJECT_NAME,
            "group": None,
            "entity": None
        }

        hyperparams_to_save = {}

        return_dict.update({"config":hyperparams_to_save})

        return return_dict
    
class ModelLogging(TypedDict):
            model_log_mode: Optional[Literal["all, weights, gradients"]]
            n_batches_freq: Annotated[int, Gt(0)]
            log_graph: bool