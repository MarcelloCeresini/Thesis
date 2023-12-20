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

        self.epsilon_for_point_matching = 1e-5
        self.mesh_to_features_scale_factor = 1e-3

        self.active_vectors_2d = ['      x-velocity', '      y-velocity']
        self.features_to_remove = ['nodenumber', 'boundary-normal-dist']
        self.features_coordinates = ['    x-coordinate', '    y-coordinate']

        self.bc_dict = {
            2:	"interior",
            3:	"wall",
            4:	"pressure-inlet, inlet-vent, intake-fan",
            5:	"pressure-outlet, exhaust-fan, outlet-vent",
            7:	"symmetry",
            8:	"periodic-shadow",
            9:	"pressure-far-field",
            10:	"velocity-inlet",
            12:	"periodic",
            14:	"fan, porous-jump, radiator",
            20:	"mass-flow-inlet",
            24:	"interface",
            31:	"parent (hanging node)",
            36:	"outflow",
            37:	"axis",
        }

        self.cell_type_dict = {
            '1': "triangle", # triangular
            '2': "tetrahedral",
            '3': "quad", # quadrilateral
        }

        self.edge_type_feature={
            "cell_cell": 1,
            "face_face": 2,
            "cell_face": 3,
        }

        self.air_speed = 50 # m/s
        self.atmosferic_pressure = 1.01325e5 # Pa

        self.graph_node_feature_dict = {
            "tangent_versor_x": 0,  # x-component of versor tangent to the face
            "tangent_versor_y": 1,  # y......
            "v_t": 2,               # velocity along the tangent versor
            "v_n": 3,               # velocity along the perpendicular versor
            "p": 4,                 # pressure on the face
            "dt_v_t": 5,            # derivative along the tangent versor of v_t
            "dt_v_n": 6,            # ...
            "dt_p": 7,              # ...
            "dn_v_t": 8,            # ...
            "dn_v_n": 9,            # ...
            "dn_p": 10,             # ...
            "component_id": 11,     # Maybe useful?
        }


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