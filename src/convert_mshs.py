import os

from config_pckg.config_file import Config
from utils import convert_msh_to_graph

conf = Config()

new_files = ["2dtc_001R001_001_s01_ascii.msh"]

for msh in new_files:
    mesh_filename = os.path.join(conf.DATA_DIR, "initial_exploration", "raw", msh) # 2D mesh, ASCII
    graph_filename = os.path.join(conf.DATA_DIR, "initial_exploration", "interim", msh.split("")[0]+"pt") # 2D mesh, ASCII
    convert_msh_to_graph(mesh_filename, graph_filename, conf)