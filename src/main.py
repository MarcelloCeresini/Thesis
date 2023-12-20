import os

from config_pckg.config_file import Config
from utils import convert_msh_to_graph

conf = Config()
mesh_filename = os.path.join(conf.DATA_DIR, "initial_exploration", "raw", "2dtc_001R001_001_s01_ascii.msh") # 2D mesh, ASCII
graph_filename = os.path.join(conf.DATA_DIR, "initial_exploration", "interim", "2dtc_001R001_001_s01_ascii.pt") # 2D mesh, ASCII
convert_msh_to_graph(mesh_filename, graph_filename, conf)
