import glob
import os
import pickle
import utils

from config_pckg.config_file import Config


if __name__ == "__main__":
    conf = Config()

    with open(os.path.join(conf.EXTERNAL_FOLDER_MESHCOMPLETE_W_LABELS, "2dtc_002R002_001_s01.pkl"), "rb") as f:
        obj = pickle.load(f)

    # obj.plot_mesh([
    #     ("cell", "label", "pressure")
    # ])

    data = utils.convert_mesh_complete_info_obj_to_graph(conf, obj)
    print(data)
    # utils.get_face_BC_attributes(obj.mesh, obj.face_center_positions, obj.vertices_in_faces, conf)