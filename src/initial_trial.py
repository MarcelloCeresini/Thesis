import os

from config_pckg.config_file import Config
from utils import convert_msh_to_graph

conf = Config()

mesh_filename = os.path.join(conf.DATA_DIR, "msh", "2dtc_001R001_001_s01_ascii.msh") # 2D mesh, ASCII
labels_filename = os.path.join(conf.DATA_DIR, "csv", "2dtc_001R001_001_s01_cell_values.csv")
final_data_filename = os.path.join(conf.DATA_DIR, "raw", "2dtc_001R001_001_s01_ascii_W_LABELS_SIMPLE.pt") 

convert_msh_to_graph(mesh_filename, conf,
                           filename_output_graph=final_data_filename,
                           labels_csv_filename=labels_filename,
                           plot_mesh=False,
                           complex_graph=False)

'''
# TODO: implement this
plot_mesh_from_graph( # can also plot in a grid
    data_filename = final_data_filename,
    what_to_plot_list_of_tuples = [
        ("label", "pressure"),
        ("feature", "tangent_versor_x"),
    ]
        # for "label": conf.features_to_keep
        # for "feature": keys of conf.graph_node_feature_dict
    
)
'''