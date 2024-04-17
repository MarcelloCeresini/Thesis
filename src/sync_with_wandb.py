import wandb
from wandb_osh import syncer
import os
import glob
from config_pckg.config_file import Config

# from wandb.proto.v4 import wandb_internal_pb2
# from wandb.sdk.internal import datastore

if __name__ == "__main__":
    
    conf = Config()
    dir_regex = os.path.join(conf.ROOT_DIR, "wandb", "offline-run-*")
    list_offline_dirs = glob.glob(dir_regex)

    print(list_offline_dirs)

    for offline_dir in list_offline_dirs:
        new_id = wandb.util.generate_id()
        syncer.sync_dir(
            dir = offline_dir,
            options=[f"--id {new_id}", "--include-synced", "--include-offline", "--sync-all"]
        )


# # https://github.com/wandb/wandb/issues/1768

# data_regex = os.path.join(
#     "H:\\","CFD-RD_SOLVER", "marcello", "Thesis-main", "wandb", "latest-run", "*.wandb"
# )

# # data_regex = os.path.join(
# #     conf.ROOT_DIR,
# #     "wandb",
# #     f"latest-run",
# #     f"run-*.wandb"
# # )

# data_path = glob.glob(data_regex)[0]

# assert os.path.isfile(data_path)
# ds = datastore.DataStore()
# ds.open_for_scan(data_path)

# for _ in range(1000):  
#     data = ds.scan_data()
#     pb = wandb_internal_pb2.Record()
#     pb.ParseFromString(data)  
#     # record = ds.scan_record()
#     # pb = wandb_internal_pb2.Record()
#     # pb.ParseFromString(record[1])

#     record_type = pb.WhichOneof("record_type")
#     if record_type == "history":
#         pass
#         for item in pb.history.item:
#             key = item.key
#             if key == "metric":
#                 print(item.value_json, pb.history.item[-1])
#             else:
#                 pass
#                 # print(key)
#         # and so on
#     elif record_type in ["output_raw", "header", "run", "files", "telemetry", "metric", "stats"]:
#         pass
#     elif record_type in ["summary"]:
#         pass
#     else:
#         pass
    