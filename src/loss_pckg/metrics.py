import numpy as np
import torch


class AeroMetric():

    def __init__(self) -> None:
        self.contributes = ["pressure", "shear", "total"]
        self.physical_quantities = ["horizontal", "vertical", "efficiency"]
        self.components = ["flap", "tyre", "car"]
        self.reset()

    def reset(self):
        self.n = 0
        self.relative_error = {contr: {k: { k1: torch.tensor(0.) 
            for k1 in self.components} 
                for k in self.physical_quantities}
                    for contr in self.contributes}

    def forward_one_contribute(self, contribute, pred, label):
        for key in self.components: 
            self.relative_error[contribute]["horizontal"][key] += \
                torch.abs((label[key][0].cpu()-pred[key][0].cpu())/label[key][0].cpu())
        for key in self.components: 
            self.relative_error[contribute]["vertical"][key] += \
                torch.abs((label[key][1].cpu()-pred[key][1].cpu())/label[key][1].cpu())
        
        for key in self.components:
            label_efficiency =  label[key][1].cpu() /label[key][0].cpu()
            pred_efficiency =   pred[key][1].cpu()  /pred[key][0].cpu()
            self.relative_error[contribute]["efficiency"][key] += torch.abs(
                (label_efficiency-pred_efficiency)/label_efficiency
            )

    def forward(self, pred, label):
        self.n += 1
        self.forward_one_contribute("pressure", pred[0], label[0])
        self.forward_one_contribute("shear", pred[1], label[1])
        total_forces_pred = {key: pred[0][key]+pred[1][key] 
                                for key in self.components}
        total_forces_lab = {key: label[0][key]+label[1][key] 
                                for key in self.components}
        self.forward_one_contribute("total", total_forces_pred, total_forces_lab)

    def compute_one_contribute(self, contribute):
        relative_error = self.relative_error[contribute]
        return {phy: {comp: v.item()/self.n 
                    for comp, v in relative_error[phy].items()} 
                        for phy in self.physical_quantities}

    def compute(self):
        return_dict = {contr: self.compute_one_contribute(contr) for contr in self.contributes}
        # return_dict = {phy: {
        #     comp: v.item()/self.n for comp, v in self.relative_error[phy].items()
        #         } for phy in self.physical_quantities}

        self.reset()
        return return_dict