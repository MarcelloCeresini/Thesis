import numpy as np
import torch


class AeroMetric():

    def __init__(self) -> None:
        self.physical_quantities = ["drag", "lift", "efficiency"]
        self.components = ["flap", "tyre", "car"]
        self.reset()

    def reset(self):
        self.n = 0
        self.relative_error = {k: {
            k1: torch.zeros(1) for k1 in self.components
                } for k in self.physical_quantities}
        
    def forward(self, pred, label):
        self.n += 1

        for key in self.components: 
            self.relative_error["drag"][key] = torch.abs((label[key][0]-pred[key][0])/label[key][0])
        for key in self.components: 
            self.relative_error["lift"][key] = torch.abs((label[key][1]-pred[key][1])/label[key][1])
        
        for key in self.components:
            label_efficiency =  label[key][1].cpu() /label[key][0].cpu()
            pred_efficiency =   pred[key][1].cpu()  /pred[key][0].cpu()
            self.relative_error["efficiency"][key] += torch.abs(
                (label_efficiency-pred_efficiency)/label_efficiency
            )
            
    def compute(self):
        return_dict = {phy: {
            comp: v.item()/self.n for comp, v in self.relative_error[phy].items()
                } for phy in self.physical_quantities}

        self.reset()
        return return_dict