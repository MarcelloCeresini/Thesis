from typing import Literal
import unittest

import wandb


from utils import get_coefficients, init_wandb
from config_pckg.config_file import Config

# class GetCoefficients(unittest.TestCase):
#     def test_zero_pressure(self):
#         pred_coefficients = get_coefficients(conf, data, pred_sample_pressure)


if __name__ == "__main__":
    WANDB_MODE: Literal["offline"] = "offline"
    unittest.main()
    init_wandb(Config(), overwrite_WANDB_MODE=WANDB_MODE)
    conf = wandb.config
