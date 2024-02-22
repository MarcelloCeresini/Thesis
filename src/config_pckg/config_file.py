from copy import deepcopy
from enum import Enum
import os
import pickle
from typing import Annotated, Dict, Literal, Optional, TypedDict, Union
import numpy as np

import yaml
from pyvista import CellType 
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, PearsonCorrCoef, WeightedMeanAbsolutePercentageError, \
    SymmetricMeanAbsolutePercentageError, RelativeSquaredError # TODO: investigate in correlation coefficients etc...

from loss_pckg import metrics

class Config():

    def __init__(self, mode=None):
        '''
        Create a config class. To make it more readable, all fixed hyperparameters can be stored in a YAML file
        To change only some parameters between runs, we use a match case
        '''
        self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.DATA_DIR = os.path.join(self.ROOT_DIR, "data")
        self.PROJECT_NAME = os.path.basename(self.ROOT_DIR)
        self.device: Literal["cpu", "cuda"] = "cuda"

        ##### external folders
        self.EXTERNAL_FOLDER = os.path.join("K:", "CFD-WT", "ML_AI", "2D_V01_database")
        self.EXTERNAL_FOLDER_MSH = os.path.join(self.EXTERNAL_FOLDER, "Mesh_ascii")
        self.EXTERNAL_FOLDER_CSV = os.path.join(self.EXTERNAL_FOLDER, "CSV_ascii")
        self.EXTERNAL_FOLDER_MESHCOMPLETE = os.path.join(self.EXTERNAL_FOLDER, "MeshCompleteObjs")
        self.EXTERNAL_FOLDER_MESHCOMPLETE_W_LABELS = os.path.join(self.EXTERNAL_FOLDER, "MeshCompleteObjsWithLabels_at300")
        self.EXTERNAL_FOLDER_GRAPHS = os.path.join(self.EXTERNAL_FOLDER, "Graphs")
        
        self.standard_datalist_path = os.path.join(self.DATA_DIR, "datalists.pt")
        self.test_imgs_comparisons = os.path.join(self.DATA_DIR, "test_imgs_comparisons")
        self.test_htmls_comparisons = os.path.join(self.DATA_DIR, "test_htmls_comparisons")
        self.test_vtksz_comparisons = os.path.join(self.DATA_DIR, "test_vtksz_comparisons")
        # self.standard_dataloader_path = os.path.join(self.DATA_DIR, "dataloaders.pt")


        self.problematic_files = {"2dtc_002R074_001_s01"}

        with open(os.path.join(self.ROOT_DIR, "src", "config_pckg", "hyperparams.yaml"), "r") as stream:
            try:
                self.hyper_params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        with open(os.path.join(self.ROOT_DIR, "src", "config_pckg", "model_structure.yaml"), "r") as stream:
            try:
                self.model_structure = yaml.safe_load(stream)
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
        self.relative_atmosferic_pressure = 0
        self.n_theta_bins = 32
        self.quantile_values = 5
        self.distance_quantiles_angular_bin = np.linspace(0,1,self.quantile_values)
        self.default_radial_attribute_value = -1
        self.n_radial_attributes = self.n_theta_bins * self.quantile_values

        self.graph_node_feature_dict = {
            "tangent_versor_x": 0,  # x-component of versor tangent to the face
            "tangent_versor_y": 1,  # y......
            "face_area": 2,
            "v_t": 3,               # velocity along the tangent versor
            "v_n": 4,               # velocity along the perpendicular versor
            "p": 5,                 # pressure on the face
            "dv_dn": 6,             # ...
            "dp_dn": 7,             # ...
            # "dv_dt": 8,             # derivative along the tangent versor of v
            # "dp_dt": 9,             # ...
        }

        self.graph_node_feature_mask = {
            "v_t": 0,               # velocity along the tangent versor
            "v_n": 1,               # velocity along the perpendicular versor
            "p": 2,                 # pressure on the face
            "dv_dn": 3,             # ...
            "dp_dn": 4,             # ...
            "is_BC": 5
            # "dv_dt": 6,             # derivative along the tangent versor of v
            # "dp_dt": 7,             # ...
        }

        self.graph_node_features_not_for_training = {
            "component_id": 0,      # Maybe useful?
            "is_car": 1,
            "is_flap": 2,
            "is_tyre": 3,
        }

        self.graph_edge_attr_list = [
            "x_dist", "y_dist", "z_dist", "norm"
        ]

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
            "RSE": RelativeSquaredError(),
            "MSE": MeanSquaredError(), 
            "MAE": MeanAbsoluteError(),
            "Pearson": PearsonCorrCoef()
        }

        self.metric_aero = metrics.AeroMetric()

        self.metric_dict = {
            metric_name:{
                label_name : deepcopy(metric_obj) for label_name in self.labels_to_keep_for_training
            }
            for metric_name, metric_obj in self.metrics.items()}
        # TODO: implement global metrics

        # TODO: find another way to apply initial mask?
        self.input_dim = len(self.graph_node_feature_dict)+len(self.graph_node_feature_mask)+self.n_radial_attributes
        self.output_dim = len(self.labels_to_keep_for_training)

    def get_tensorboard_logging_info(self) -> Dict:
        '''
        Return a Dict with the information for wandb logging and the hyperparameters that are important to log
        '''
        logged_hyperparams = self.hyper_params
        logged_hyperparams.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "edge_feature_dim": len(self.graph_edge_attr_list),
            "label_dim": len(self.labels_to_keep_for_training)
        })

        return_dict = {
            "group": None,
            "hyperparams": logged_hyperparams,
            "model": self.model_structure,
        }

        return return_dict
    