from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore
from config_pckg.config_file import Config
import os

conf = Config()

run_name = "eb5rgzif"

data_path = os.path.join(
    conf.ROOT_DIR,
    "wandb",
    f"offline-run-20240411_155730-{run_name}",
    f"run-{run_name}.wandb"
)

assert os.path.isfile(data_path)
ds = datastore.DataStore()
ds.open_for_scan(data_path)

for _ in range(1000):  
    data = ds.scan_data()
    pb = wandb_internal_pb2.Record()
    pb.ParseFromString(data)  
    record_type = pb.WhichOneof("record_type")
    if record_type == "history":
        tracked_value_0 = pb.history.item[0].value_json
        # and so on
    else:
        pass
    