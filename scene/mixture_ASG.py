from utils.general_utils import build_rotation
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _dot(x, y, keepdim=True):
    return torch.sum(x * y, -1, keepdim=keepdim)

class Mixture_of_ASG(nn.Module):
    def __init__(self, basis_asg_num=8):
        super().__init__()
        self.basis_asg_num = basis_asg_num
        self.const = math.sqrt(2) * math.pow(math.pi, 2 / 3)
        self.F0 = 0.04
        self.softplus = torch.nn.Softplus()

        # initialized as different scale for fast convergency
        self.asg_sigma_ratio = self.softplus(torch.linspace(-2., 0., steps=self.basis_asg_num, device="cuda"))
        asg_sigma = torch.zeros((self.basis_asg_num), dtype=torch.float, device="cuda")
        self.asg_sigma = nn.Parameter(asg_sigma.requires_grad_(True))
        
        # initlized as aniso
        asg_scales = torch.ones((self.basis_asg_num, 2), dtype=torch.float, device="cuda") * -2.1972
        asg_scales[:, 0] = asg_scales[:, 0] * 0.5
        self.asg_scales = nn.Parameter(asg_scales.requires_grad_(True))
        
        # ASG frames
        asg_rotation = torch.zeros((self.basis_asg_num, 4), dtype=torch.float, device="cuda")
        asg_rotation[:, 0] = 1
        self.asg_rotation = nn.Parameter(asg_rotation.requires_grad_(True))

    @property
    def get_asg_lam_miu(self):
        return torch.sigmoid(self.asg_scales) * 10. # (basis_asg_num, 2)
    
    @property
    def get_asg_sigma(self):
        return torch.sigmoid(self.asg_sigma) * self.asg_sigma_ratio # (basis_asg_num)
    
    @property
    def get_asg_axis(self):
        return build_rotation(self.asg_rotation).reshape(-1, 3, 3) # (basis_asg_num, 3, 3)
    
    @property
    def get_asg_normal(self):
        return self.get_asg_axis[:, :, 2] # (basis_asg_num, 3)
    

    def forward(self, wi, wo, alpha, asg_scales, asg_axises):
        """
        wi, wo: (K, 3)
        alpha: (K, basis_asg_num)
        """
        half = F.normalize(wo + wi, p=2, dim=-1)                    # (K, 3)
        Fresnel = self.F0 + (1 - self.F0) * \
            torch.clamp(1.0 - _dot(wi, half), 0.0, 1.0).pow(5)      # (K, 1)
        
        half = half.unsqueeze(1).expand(-1, self.basis_asg_num, -1) # (K, basis_asg_num, 3)
        alpha = alpha                                               # (K, basis_asg_num)
        
        # axis of ASG frame
        asg_x = asg_axises[:, :, 0].unsqueeze(0)                    # (1, basis_asg_num, 3)
        asg_y = asg_axises[:, :, 1].unsqueeze(0)                    # (1, basis_asg_num, 3)
        asg_z = asg_axises[:, :, 2].unsqueeze(0)                    # (1, basis_asg_num, 3)

        # variance
        lam = asg_scales[:, 0].unsqueeze(0)                         # (1, basis_asg_num)
        miu = asg_scales[:, 1].unsqueeze(0)                         # (1, basis_asg_num)
        sigma = self.get_asg_sigma.unsqueeze(0)                     # (1, basis_asg_num)
        
        s = F.normalize(half - _dot(half, asg_z) * asg_z, p=2, dim=-1)
        aniso_ratio = torch.sqrt((_dot(s, asg_x, keepdim=False) / lam).pow(2) \
            + (_dot(s, asg_y, keepdim=False) / miu).pow(2))         # (K, basis_asg_num)
        
        cos_theta = _dot(half, asg_z, keepdim=False)                # (K, basis_asg_num)
        cos_theta = torch.clamp(cos_theta, -1+1e-6, 1-1e-6)
        asg_res = torch.exp(- 0.5 * (torch.arccos(cos_theta) * aniso_ratio / sigma)**2)
        asg_res = asg_res / (self.const * sigma)
        mm_asg_res = torch.sum(alpha * asg_res, dim=-1, keepdim=True) # (K, 1)
        
        return mm_asg_res * Fresnel                                  # (K, 1)
