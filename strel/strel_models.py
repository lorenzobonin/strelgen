import os
import torch
import pytorch_lightning as pl

import strel.strel_properties as sp



#######################################################
# THIS FILE CONTAINS THE MAIN MODEL CLASS FOR STREL GUIDED GENERATION
# The class GenFromLatent takes as input a latent vector and outputs the robustness value
# of a given STREL property for the scenario generated from the latent vector.
# The property to evaluate can be chosen by setting the property_name argument in the constructor. 
# New properties can be added to strel_properties.py and then used here by adding a new case in the forward method.
#######################################################

class GenFromLatent(pl.LightningModule):
        def __init__(self, model, scen_id, full_types, pred_types, property_name="ped_unsafe", tmax=0.2, tglob=3):
            super().__init__()
            self.model = model
            self.scen_id = scen_id
            self.full_types = full_types
            self.pred_types = pred_types
            self.property_name = property_name
            self.tmax = tmax
            self.tglob = tglob
            self.valid_types = full_types


            

        def forward(self, z):

            out = self.model.latent_generator(
                z,
                self.scen_id,
                plot=False,
                enable_grads=True,
                return_pred_only=False,

            )
            full_world, pred_eval_local, mask_eval, eval_mask = out


            # Choose STREL property
            elif self.property_name == "pred_reach":
                robustness = sp.evaluate_reach_fast_slow(pred_eval_local, self.pred_types,d_zone=4)

            elif self.property_name == "surround_fast":
                robustness = sp.evaluate_speeding_surrounded_unmask(full_world, self.valid_types)

            elif self.property_name == "surround_pred":
                robustness = sp.evaluate_speeding_surrounded_unmask(pred_eval_local, self.pred_types, d_zone=1.8)

            elif self.property_name =="ped_pred":
                robustness = sp.evaluate_ped_somewhere_unmask(pred_eval_local, self.pred_types,d_zone=2.5)

            elif self.property_name == "ped_unsafe":
                robustness = sp.evaluate_ped_somewhere_unmask(full_world, self.valid_types, d_zone= 6.0)
            else:
                raise ValueError(f"Unknown property type '{self.property_name}'")

            return robustness