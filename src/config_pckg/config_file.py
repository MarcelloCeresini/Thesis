from copy import deepcopy
from enum import Enum
import os
import sys
import pickle
from typing import Annotated, Dict, Literal, Optional, TypedDict, Union
import numpy as np
import torch

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
        self.PROJECT_NAME = os.path.basename(self.ROOT_DIR)
        self.WANDB_FLAG = True

        self.DATA_DIR = os.path.join(self.ROOT_DIR, "data")
        self.device: Literal["cpu", "cuda"] = "cuda"
        self.DEBUG_BYPASS_MODEL = False

        ##### external folders
        self.EXTERNAL_FOLDER = os.path.join("K:", "CFD-WT", "ML_AI", "2D_V01_database")
        self.EXTERNAL_FOLDER_MSH = os.path.join(self.EXTERNAL_FOLDER, "Mesh_ascii")
        self.EXTERNAL_FOLDER_CSV = os.path.join(self.EXTERNAL_FOLDER, "CSV_ascii")
        self.EXTERNAL_FOLDER_MESHCOMPLETE = os.path.join(self.EXTERNAL_FOLDER, "MeshCompleteObjs")
        self.EXTERNAL_FOLDER_MESHCOMPLETE_W_LABELS = os.path.join(self.EXTERNAL_FOLDER, "MeshCompleteObjsWithLabels_at300")
        self.EXTERNAL_FOLDER_GRAPHS = os.path.join(self.EXTERNAL_FOLDER, "Graphs")
        
        if "UNIBO" in sys.platform:
            raise NotImplementedError("Check this") # TODO: CHECK
            self.DATA_DIR = "/public.hpc/marcello.ceresini"
            self.device: Literal["cuda"] = "cuda"
        # else:
        #     raise NotImplementedError(f"{sys.platform} is not supported yet")

        self.standard_dataset_dir = os.path.join(self.DATA_DIR, "dataset_files")
        self.test_imgs_comparisons = os.path.join(self.DATA_DIR, "test_imgs_comparisons")
        self.test_htmls_comparisons = os.path.join(self.DATA_DIR, "test_htmls_comparisons")
        self.test_vtksz_comparisons = os.path.join(self.DATA_DIR, "test_vtksz_comparisons")

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

        with open(os.path.join(self.ROOT_DIR, "src", "config_pckg", "train_label_stats.pkl"), "rb") as f:
            dict_labels_train = pickle.load(f)
            self.dict_labels_train = {k:v.to_dict() for k,v in dict_labels_train.items()}

        self.physical_labels = [
            'x-velocity',
            'y-velocity',
            'pressure',
        ]

        self.labels_to_keep_for_training = [
            'x-velocity',
            'y-velocity',
            'pressure',
        ]

        self.labels_to_keep_for_training_turbulence = [
            'turb-kinetic-energy',
            'turb-diss-rate',
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

        # physical constants
        self.air_speed = 50 # m/s
        self.T = 288.16 # K
        self.relative_atmosferic_pressure = 0 # Pa
        self.atm = 101325 # Pa
        self.air_dynamic_viscosity = 1.7894e-5 # kg/(m*s)
        self.air_density = 1.225 # kg/m^3
        self.Q = (self.air_density * self.air_speed**2)/2
        self.air_kinematic_viscosity = self.air_dynamic_viscosity / self.air_density # m^2/s
        self.L = 1 # m
        self.reference_area = 1 
        self.Re = self.air_speed*self.L/self.air_kinematic_viscosity # adimensional
        self.standard_normalized_Re = self.L/self.air_kinematic_viscosity # air_speed=1
        self.dynamic_pressure = (0.5*self.reference_area*self.air_density*self.air_speed**2)
    
        self.C_lim_w = 7/8
        self.beta_star_w = 0.09

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
            # "tangent_versor_angle": 8,
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
            "ground": 4,
            "tyre": 5,
            "main_flap": 6,
            "second_flap": 7,
            "p_outlet": 8,
            "simmetry": 9,
            "v_inlet": 10,
        }

        self.car_parts_for_coefficients = ["main_flap", "second_flap", "tyre", "is_car"]

        # self.graph_edge_attr_list = ["x_dist", "y_dist", "norm"]
        self.graph_edge_attr_list = ["x_dist", "y_dist"]
        self.edge_feature_dim = len(self.graph_edge_attr_list)

        self.flag_directional_BC_velocity = True

        self.feature_normalization_mode = "Physical"

        self.label_normalization_mode = {
            "x-velocity": {
                "main": "physical", # physical / max-normalization / standardization
                "magnitude": True,
            },
            "y-velocity": {
                "main": "physical",
                "magnitude": True,
            },
            "pressure": {"main": "physical",},
            "turb-diss-rate": {"main": "max-normalization",},
            "turb-kinetic-energy": {"main": "max-normalization"},
        }

        self.w_min_for_clamp = 1e-6

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
            split_idxs = pickle.load(f)
            self.split_idxs = {k:list(v.values) for k,v in split_idxs.items()}

        self.metrics = {
            "MAPR": MeanAbsolutePercentageError, 
            "wMAPR": WeightedMeanAbsolutePercentageError, 
            "SMAPE": SymmetricMeanAbsolutePercentageError, 
            "RSE": RelativeSquaredError,
            "MSE": MeanSquaredError, 
            "MAE": MeanAbsoluteError,
            "Pearson": PearsonCorrCoef
        }

        self.metric_aero = metrics.AeroMetric

        self.bool_radial_attributes = True
        if self.bool_radial_attributes:
            self.input_dim = len(self.graph_node_feature_dict)+len(self.graph_node_feature_mask)+self.n_radial_attributes
        else:
            self.input_dim = len(self.graph_node_feature_dict)+len(self.graph_node_feature_mask)

        self.PINN_mode: Literal["supervised_only", "supervised_with_sampling", "continuity_only", "full_laminar", "turbulent_kw"] \
            = "turbulent_kw"
        
        self.output_turbulence: bool = True
        if self.PINN_mode == "turbulent_kw" and not self.output_turbulence:
            print("Cannot compute momentum residuals without turbolence, setting output_turbulence=True")
            self.output_turbulence = True
        
        if self.output_turbulence:
            self.labels_to_keep_for_training += self.labels_to_keep_for_training_turbulence
        self.output_dim = len(self.labels_to_keep_for_training)

        self.activation_for_max_normalized_features = True
        if self.activation_for_max_normalized_features:
            self.idx_to_apply_activation = torch.tensor([True \
                if self.label_normalization_mode.get(tmp, {}).get("main", "") == "max-normalization" else False\
                for tmp in self.labels_to_keep_for_training])

        self.metric_dict = {
            metric_name:{
                label_name : deepcopy(metric_obj) for label_name in self.labels_to_keep_for_training
            }
            for metric_name, metric_obj in self.metrics.items()}
        
        self.bool_bootstrap_bias = True
        self.bootstrap_bias = {k:self.dict_labels_train["mean"][k] for k in self.labels_to_keep_for_training}

        self.flag_BC_PINN: bool = True
        self.inference_mode_latent_sampled_points: Literal["squared_distance", "fourier_features", "baseline_positional_encoder", "new_edges"] \
            = "new_edges"
        self.graph_sampling_p_for_interpolation = 0.01
        self.fourier_feat_dim = 64

        domain_sampling_mode: Literal["all_domain", "percentage_of_domain", "uniformly_cells"] = \
                "uniformly_cells"
        self.domain_sampling = {"mode": domain_sampling_mode, 
                                    "percentage": 0.5}

        boundary_sampling_mode: Literal["all_boundary", "percentage_of_boundary"] = \
                "all_boundary"
        self.boundary_sampling = {"mode": boundary_sampling_mode, 
                                    "percentage": 3.,
                                    "shift_on_face":True}

        self.general_sampling = {"add_edges": True,
                                    "use_sampling_weights":True}
        
        self.n_sampled_new_edges = 3

        self.residual_loss: Literal["MAE", "MSE"] = "MSE"

        self.standard_weights = {
            "supervised": 1,
            "supervised_on_sampled": 1,
            "boundary": 15,
            "continuity": 10,
            "momentum_x": 3,
        }
        self.standard_weights["momentum_y"] = self.standard_weights["momentum_x"]

        self.physical_constraint_loss = True
        if self.physical_constraint_loss:
            self.standard_weights.update({
                "negative_k": 1,
                "negative_w": 1,
            })

        self.normalize_denormalized_loss_components = True

        self.dynamic_loss_weights = False
        # self.main_loss_component_dynamic = "supervised" #"supervised_on_sampled"
        self.main_loss_component_dynamic = "supervised"
        self.lambda_dynamic_weights = 0.3 # (0.1 in NSFnets arXiv:2003.06496v1)
        self.gamma_loss = 10

        self.gradient_clip_value = 1

        self.logging = {
            "model_log_mode": "all",
            "n_batches_freq": 1,
            "log_graph": True,
        }

    def get_logging_info(self) -> Dict:
        '''
        Return a Dict with the information for wandb logging and the hyperparameters that are important to log
        '''
        logged_hyperparams = self.hyper_params
        logged_hyperparams.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "edge_feature_dim": len(self.graph_edge_attr_list),
            "output_turbulence":self.output_turbulence,
            "label_dim": len(self.labels_to_keep_for_training),
            "labels_to_keep_for_training": self.labels_to_keep_for_training,
            "label_normalization_mode":self.label_normalization_mode,
            "activation_for_max_normalized_features":self.activation_for_max_normalized_features,
            "idx_to_apply_activation":self.idx_to_apply_activation,
            "v_inlet":self.air_speed,
            "dict_labels_train":self.dict_labels_train,
            "bool_bootstrap_bias": self.bool_bootstrap_bias,
            "bootstrap_bias": self.bootstrap_bias,
            "bool_radial_attributes": self.bool_radial_attributes,
            "PINN_mode": self.PINN_mode,
            "flag_BC_PINN": self.flag_BC_PINN,
            "feat_dict": self.graph_node_feature_dict,
            "mask_dict": self.graph_node_feature_mask,
            "domain_sampling": self.domain_sampling,
            "boundary_sampling": self.boundary_sampling,
            "general_sampling": self.general_sampling,
            "inference_mode_latent_sampled_points": self.inference_mode_latent_sampled_points,
            "graph_sampling_p_for_interpolation": self.graph_sampling_p_for_interpolation,
            "fourier_feat_dim": self.fourier_feat_dim,
            "n_sampled_new_edges": self.n_sampled_new_edges,
            "standard_weights": self.standard_weights,
        })

        return_dict = {
            "group": None,
            "hyperparams": logged_hyperparams,
            "model": self.model_structure,
        }

        return return_dict
    