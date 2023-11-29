from typing import List, Dict

import torch as T

from config_pckg.config_file import Config


def loss(pred, label, cfg: Config) -> Dict[str, T.float]:
    pass