import glob
import os
import pickle

from config_pckg.config_file import Config


if __name__ == "__main__":
    conf = Config()

    with open(os.path.join(conf.EXTERNAL_FOLDER_MESHCOMPLETE, "2dtc_002R002_001_s01.pkl"), "rb") as f:
        obj = pickle.load(f)


    last_csv = glob.glob(os.path.join(conf.EXTERNAL_FOLDER_CSV, "2dtc_002R002_001_s01"+"*_at300.csv"))
    if len(last_csv) == 1:
        path_csv = last_csv[0]
    obj.add_labels(path_csv)