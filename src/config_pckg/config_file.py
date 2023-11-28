import os
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


with open("hyperparams.yaml", "r") as stream:
    try:
        hyper_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
