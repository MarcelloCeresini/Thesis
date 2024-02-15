import numpy as np


class AeroMetric():

    def __init__(self) -> None:
        self.n = 0
        self.relative_error = {"flap":0, "tyre":0, "car":0}

    def forward(self, pred, label):
        # TODO: move everything on cpu to be sure
        self.n += 1
        for key in self.relative_error:
            label_efficiency = label[key][1]/label[key][0]
            pred_efficiency = pred[key][1]/pred[key][0]
            self.relative_error[key] += np.abs(
                (label_efficiency-pred_efficiency)/label_efficiency)
            
    def compute(self):
        return_dict = {key: val/self.n for key, val in self.relative_error.items()}
        self.n = 0
        return return_dict