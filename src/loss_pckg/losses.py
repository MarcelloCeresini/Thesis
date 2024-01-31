from typing import List, Dict, Optional

import torch
import torch.nn.functional as F

from config_pckg.config_file import Config


def MSE_loss(pred:torch.Tensor, label:torch.Tensor):
    return F.mse_loss(pred, label, reduction="mean")
