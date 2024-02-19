import numpy as np
import torch


class AeroMetric():

    def __init__(self) -> None:
        self.n = None
        self.relative_error = None
        self.reset()

    def reset(self):
        self.n = 0
        self.relative_error = {"flap": torch.zeros((1)), 
                                "tyre": torch.zeros((1)), 
                                "car": torch.zeros((1))}

    def forward(self, pred, label):
        self.n += 1

        for key in self.relative_error:
            label_efficiency =  label[key][1].cpu() /label[key][0].cpu()
            pred_efficiency =   pred[key][1].cpu()  /pred[key][0].cpu()
            self.relative_error[key] += torch.abs(
                (label_efficiency-pred_efficiency)/label_efficiency)
            
    def compute(self):
        return_dict = {key: (val/self.n)[0] for key, val in self.relative_error.items()}
        self.reset()
        return return_dict