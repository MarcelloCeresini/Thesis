import os

from config_pckg.config_file import Config
from utils import convert_msh_to_graph

conf = Config()

mesh_filename = os.path.join(conf.DATA_DIR, "msh", "2dtc_001R001_001_s01_ascii.msh") # 2D mesh, ASCII
labels_filename = os.path.join(conf.DATA_DIR, "csv", "2dtc_001R001_001_s01_cell_values.csv")
final_data_filename = os.path.join(conf.DATA_DIR, "raw", "2dtc_001R001_001_s01_ascii_W_LABELS.pt") 

data = convert_msh_to_graph(mesh_filename, conf,
                    filename_output_graph=final_data_filename,
                    labels_csv_filename=labels_filename)