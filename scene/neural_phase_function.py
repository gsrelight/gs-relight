from collections import OrderedDict
import torch
import torch.nn as nn
import math
import tinycudann as tcnn

class Neural_phase(nn.Module):
    def __init__(self, hidden_feature_size=64, hidden_feature_layers=3, frequency=8, neural_material_size=8):
        super().__init__()
        self.neural_material_size = neural_material_size
        
        # configuration
        # frequency encoding
        encoding_config = {
                    "otype": "Frequency",
                    "n_frequencies": frequency
                }
        # shadow refine
        shadow_config = {
                    "otype": "FullyFusedMLP",
                    "activation": "LeakyReLU",
                    "output_activation": "Sigmoid", 
                    "n_neurons": hidden_feature_size,
                    "n_hidden_layers": hidden_feature_layers,
                }
        # other effects
        other_effects_config = {
                    "otype": "FullyFusedMLP",
                    "activation": "LeakyReLU",
                    "output_activation": "Sigmoid", 
                    "n_neurons": 128,
                    "n_hidden_layers": 3,
                }
        
        self.encoding = tcnn.Encoding(3, encoding_config)
        
        self.shadow_func = tcnn.Network(self.neural_material_size + self.encoding.n_output_dims * 2 + 1, 1, shadow_config)
           
        self.other_effects_func = tcnn.Network(self.neural_material_size + self.encoding.n_output_dims * 2, 3, other_effects_config)
        
        for n, m in self.shadow_func.named_children():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight.data, 0.0)
                
        for n, m in self.other_effects_func.named_children():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight.data, 0.0)

    def freeze(self):
        for name, param in self.shadow_func.named_parameters():
            param.requires_grad = False
        for name, param in self.other_effects_func.named_parameters():
            param.requires_grad = False

    def unfreeze(self):
        for name, param in self.shadow_func.named_parameters():
            param.requires_grad = True
        for name, param in self.other_effects_func.named_parameters():
            param.requires_grad = True

    def forward(self, wi, wo, pos, neural_material, hint):
        wi_enc = self.encoding(wi)
        wo_enc = self.encoding(wo)
        pos_enc = self.encoding(pos)
        other_effects = self.other_effects_func(torch.concat([wo_enc, pos_enc, neural_material], dim=-1)) * 2.0
        assert not torch.isnan(other_effects).any()
        decay = self.shadow_func(torch.concat([wi_enc, pos_enc, neural_material, hint], dim=-1))
        return torch.relu(decay - 1e-5), other_effects