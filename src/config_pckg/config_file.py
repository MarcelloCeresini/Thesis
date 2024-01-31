from copy import deepcopy
from enum import Enum
import os
import pickle
from typing import Annotated, Dict, Literal, Optional, TypedDict, Union

import yaml
from pyvista import CellType 
from torchmetrics import MeanAbsolutePercentageError, WeightedMeanAbsolutePercentageError, \
    SymmetricMeanAbsolutePercentageError, RelativeSquaredError # TODO: investigate in correlation coefficients etc...

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

        ##### external folders
        self.EXTERNAL_FOLDER = os.path.join("K:", "CFD-WT", "ML_AI", "2D_V01_database")
        self.EXTERNAL_FOLDER_MSH = os.path.join(self.EXTERNAL_FOLDER, "Mesh_ascii")
        self.EXTERNAL_FOLDER_CSV = os.path.join(self.EXTERNAL_FOLDER, "CSV_ascii")
        self.EXTERNAL_FOLDER_MESHCOMPLETE = os.path.join(self.EXTERNAL_FOLDER, "MeshCompleteObjs")
        self.EXTERNAL_FOLDER_MESHCOMPLETE_W_LABELS = os.path.join(self.EXTERNAL_FOLDER, "MeshCompleteObjsWithLabels_at300")
        self.EXTERNAL_FOLDER_GRAPHS = os.path.join(self.EXTERNAL_FOLDER, "Graphs")

        self.problematic_files = ["2dtc_002R074_001_s01"]

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

        self.dim = 2
        self.epsilon_for_point_matching = 1e-4
        self.mesh_to_features_scale_factor = 1e-3
        self.labels_for_which_element: Literal["points", "faces"] = "faces"

        self.features_to_remove = [
            'cellnumber',
        ]

        self.features_coordinates = [
            'x-coordinate',
            'y-coordinate',
        ]

        self.features_to_keep = [
            'x-velocity',
            'y-velocity',
            'pressure',
            'dx-velocity-dx',
            'dy-velocity-dx',
            'dp-dx',
            'dx-velocity-dy',
            'dy-velocity-dy',
            'dp-dy',
            'turb-diss-rate',
            'turb-kinetic-energy',
        ]

        self.labels_to_keep_for_training = [
            'x-velocity',
            'y-velocity',
            'pressure',
        ]

        self.csv_features = self.features_to_keep + self.features_coordinates + self.features_to_remove

        self.active_vectors_2d = ['x-velocity', 'y-velocity']

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

        self.pyvista_face_type_dict = {
            2: CellType.LINE,
        }

        # TODO: do not know how to specify cells in 3D for pyvista
        self.pyvista_cell_type_dict = {
            3: CellType.POLYGON,
            4: CellType.POLYGON,
            5: CellType.POLYGON,
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
            "dv_dt": 5,           # derivative along the tangent versor of v
            "dp_dt": 6,             # ...
            "dv_dn": 7,           # ...
            "dp_dn": 8,            # ...
            "component_id": 9,     # Maybe useful?
        }

        self.flag_directional_BC_velocity = True

        self.feature_normalization_mode = "Physical"

        self.label_normalization_mode = {
            "main": "Physical",                         # "None" or "Z-Normalization"
            "velocity_mode": "magnitude_wise",          # "component_wise" or "magnitude_wise" (only needed if Z-norm is chosen)
            "graph_wise": False,                        # True = dataset wise
            "no_shift": True,                           # True = only rescales, doesn't shift values
        }

        # TODO: compute and fill these values (maybe in a file?)
        with open(os.path.join(self.ROOT_DIR, "src", "config_pckg", "dataset_label_stats.pkl"), "rb") as f:
            dataset_label_stats = pickle.load(f)
        
        self.train_set_normalization_constants = {      # these are macro-averages
            "vx_mean": dataset_label_stats["mean"]["x-velocity"], # 0 is train split
            "vx_std": dataset_label_stats["std"]["x-velocity"],
            "vy_mean": dataset_label_stats["mean"]["y-velocity"],
            "vy_std": dataset_label_stats["std"]["x-velocity"],
            "p_mean": dataset_label_stats["mean"]["pressure"],
            "p_std": dataset_label_stats["std"]["pressure"],
            "v_mag_mean": dataset_label_stats["mean"]["v_mag"],
            "v_mag_std": dataset_label_stats["std"]["v_mag"],
        }

        with open(os.path.join(self.ROOT_DIR, "src", "config_pckg", "splits.pkl"), "rb") as f:
            self.split_idxs = pickle.load(f)

        self.metrics = {
            "MAPR": MeanAbsolutePercentageError(), 
            "wMAPR": WeightedMeanAbsolutePercentageError(), 
            "SMAPE": SymmetricMeanAbsolutePercentageError(), 
            "RSE": RelativeSquaredError()
        }

        self.metric_dict = {
            metric_name:{
                label_name : deepcopy(metric_obj) for label_name in self.labels_to_keep_for_training
            }
            for metric_name, metric_obj in self.metrics.items()}
        # TODO: implement global metrics


    def get_wandb_logging_info(self) -> Dict:
        '''
        Return a Dict with the information for wandb logging and the hyperparameters that are important to log
        '''
        return_dict = {
            "project": self.PROJECT_NAME,
            "group": None,
            "entity": None
        }

        logged_hyperparams = {}

        return_dict.update({"config":logged_hyperparams})

        return return_dict
    