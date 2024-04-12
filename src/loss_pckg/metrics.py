import numpy as np
import torch


class AeroMetric():

    def __init__(self, conf) -> None:
        self.conf = conf 
        self.contributes = ["pressure", "shear", "total"]
        self.physical_quantities = ["horizontal", "vertical", "efficiency"]
        self.components = self.conf.car_parts_for_coefficients
        self.reset()

    def reset(self):
        self.n = {k:0 for k in self.components}
        self.relative_error = {contr: {k: { k1: torch.tensor(0.) 
            for k1 in self.components} 
                for k in self.physical_quantities}
                    for contr in self.contributes}

    def forward_one_contribute(self, contribute, pred, label, skip_second_flap):
        if skip_second_flap:
            current_components = set(self.components).difference(["second_flap"])
        else:
            current_components = self.components

        for key in current_components: 
            self.relative_error[contribute]["horizontal"][key] += \
                torch.abs((label[key][0].cpu()-pred[key][0].cpu())/label[key][0].cpu())
        for key in current_components: 
            self.relative_error[contribute]["vertical"][key] += \
                torch.abs((label[key][1].cpu()-pred[key][1].cpu())/label[key][1].cpu())
        
        for key in current_components:
            label_efficiency =  label[key][1].cpu() /label[key][0].cpu()
            pred_efficiency =   pred[key][1].cpu()  /pred[key][0].cpu()
            self.relative_error[contribute]["efficiency"][key] += torch.abs(
                (label_efficiency-pred_efficiency)/label_efficiency
            )

    def forward(self, pred, label):
        self.n = {k:v+1 for k,v in self.n.items()}
        skip_second_flap = False
        if label["second_flap"][0].count_nonzero() == 0 and label["second_flap"][1].count_nonzero() == 0:
            self.n["second_flap"] -= 1
            skip_second_flap = True
        self.forward_one_contribute("pressure", 
            {k:v[0] for k,v in pred.items()}, 
            {k:v[0] for k,v in label.items()}, skip_second_flap=skip_second_flap)
        self.forward_one_contribute("shear", 
            {k:v[1] for k,v in pred.items()}, 
            {k:v[1] for k,v in label.items()}, skip_second_flap=skip_second_flap)
        total_forces_pred = {k:sum(v) for k,v in pred.items()}
        total_forces_lab = {k:sum(v) for k,v in label.items()}
        self.forward_one_contribute("total", total_forces_pred, total_forces_lab, skip_second_flap=skip_second_flap)


    def compute_one_contribute(self, contribute):
        relative_error = self.relative_error[contribute]
        return {phy: {comp: (v.item()/self.n[comp]) if self.n[comp] != 0 else 1000.
                    for comp, v in relative_error[phy].items()} 
                        for phy in self.physical_quantities}

    def compute(self):
        return_dict = {contr: self.compute_one_contribute(contr) for contr in self.contributes}
        # return_dict = {phy: {
        #     comp: v.item()/self.n for comp, v in self.relative_error[phy].items()
        #         } for phy in self.physical_quantities}

        self.reset()
        return return_dict