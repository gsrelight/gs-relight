#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
import torch.nn.functional as F
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.neural_phase_function import Neural_phase
from scene.mixture_ASG import Mixture_of_ASG

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
        self.kd_activation = torch.nn.Softplus()
        self.mm_asg_aplha_activation = torch.nn.Softplus()


    def __init__(self, 
                 sh_degree : int, 
                 use_MBRDF=False, 
                 basis_asg_num=8, 
                 hidden_feature_size=32, 
                 hidden_feature_layer=3, 
                 phase_frequency=4, 
                 neural_material_size=6, 
                 maximum_gs=1_000_000):
        
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)

        self.use_MBRDF = use_MBRDF
        self.kd = torch.empty(0)
        self.ks = torch.empty(0)
        
        # ASG params
        self.basis_asg_num = basis_asg_num
        self.alpha_asg = torch.empty(0)
        self.asg_func = torch.empty(0)
        
        # local frame
        self.local_q = torch.empty(0)
        
        # latent params
        self.neural_material = torch.empty(0)
        self.neural_material_size = neural_material_size
        self.hidden_feature_size = hidden_feature_size
        self.hidden_feature_layer = hidden_feature_layer
        self.phase_frequency = phase_frequency
        self.neural_phasefunc = torch.empty(0)

        if self.use_MBRDF:
            print("Use our shading function!")
        
        # maximum Gaussian number
        self.maximum_gs = maximum_gs
        
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.out_weights_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self.kd,
            self.ks,
            self.basis_asg_num,
            self.alpha_asg,
            self.asg_func.asg_sigma,
            self.asg_func.asg_rotation,
            self.asg_func.asg_scales,
            self.local_q,
            self.neural_material,
            self.neural_material_size,
            self.neural_phasefunc.state_dict() if self.use_MBRDF else self.neural_phasefunc,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.out_weights_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.maximum_gs
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self.kd,
        self.ks,
        self.basis_asg_num,
        self.alpha_asg,
        self.asg_func.asg_sigma,
        self.asg_func.asg_rotation,
        self.asg_func.asg_scales,
        self.local_q,
        self.neural_material,
        self.neural_material_size,
        neural_phasefunc_param,
        self._scaling,
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        out_weights_accum,
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.maximum_gs) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.out_weights_accum = out_weights_accum
        self.denom = denom
        if self.use_MBRDF:
            self.neural_phasefunc.load_state_dict(neural_phasefunc_param)
        else:
            self.neural_phasefunc = neural_phasefunc_param
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_kd(self):
        return self.kd_activation(self.kd)
    
    @property
    def get_ks(self):
        return self.kd_activation(self.ks)

    @property
    def get_alpha_asg(self):
        return self.mm_asg_aplha_activation(self.alpha_asg)

    @property
    def get_local_axis(self):
        return build_rotation(self.local_q) # (K, 3, 3)
    
    @property
    def get_local_z(self):
        return self.get_local_axis[:, :, 2] # (K, 3)

    @property
    def get_neural_material(self):
        return self.neural_material
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        if self.use_MBRDF:
            kd = torch.ones((features.shape[0], 3), dtype=torch.float, device="cuda")*0.5
            self.kd = nn.Parameter(kd.requires_grad_(True))
            ks = torch.ones((features.shape[0], 3), dtype=torch.float, device="cuda")*0.5
            self.ks = nn.Parameter(ks.requires_grad_(True))
            
            alpha_asg = torch.zeros((features.shape[0], self.basis_asg_num), dtype=torch.float, device="cuda")
            self.alpha_asg = nn.Parameter(alpha_asg.requires_grad_(True))
            self.asg_func = Mixture_of_ASG(self.basis_asg_num)
            
            local_rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            local_rots[:, 0] = 1
            self.local_q = nn.Parameter(local_rots.requires_grad_(True))
            
            neural_materials = torch.ones((features.shape[0], self.neural_material_size), dtype=torch.float, device="cuda")
            self.neural_material = nn.Parameter(neural_materials.requires_grad_(True))
            self.neural_phasefunc = Neural_phase(hidden_feature_size=self.hidden_feature_size, \
                                                hidden_feature_layers=self.hidden_feature_layer, \
                                                frequency=self.phase_frequency, \
                                                neural_material_size=self.neural_material_size).to(device="cuda")
           
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.out_weights_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.use_MBRDF:
            l.append({'params': [self.kd], 'lr': training_args.kd_lr, "name": "kd"})
            l.append({'params': [self.ks], 'lr': training_args.ks_lr, "name": "ks"})
            l.append({'params': [self.alpha_asg], 'lr': training_args.asg_lr_init, "name": "alpha_asg"})
            l.append({'params': [self.asg_func.asg_sigma], 'lr': training_args.asg_lr_init, "name": "asg_sigma"})
            l.append({'params': [self.asg_func.asg_scales], 'lr': training_args.asg_lr_init, "name": "asg_scales"})
            l.append({'params': [self.asg_func.asg_rotation], 'lr': training_args.asg_lr_init, "name": "asg_rotation"})
            l.append({'params': [self.local_q], 'lr': training_args.local_q_lr_init, "name": "local_q"})
            l.append({'params': [self.neural_material], 'lr': training_args.neural_phasefunc_lr_init, "name": "neural_material"})
            l.append({'params': self.neural_phasefunc.parameters(), 'lr': training_args.neural_phasefunc_lr_init, "name": "neural_phasefunc"})


        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.asg_scheduler_args = get_expon_lr_func(lr_init=training_args.asg_lr_init,
                                                    lr_final=training_args.asg_lr_final,
                                                    lr_delay_mult=training_args.asg_lr_delay_mult,
                                                    max_steps=training_args.asg_lr_max_steps)
        
        self.local_q_scheduler_args = get_expon_lr_func(lr_init=training_args.local_q_lr_init,
                                                    lr_final=training_args.local_q_lr_final,
                                                    lr_delay_mult=training_args.local_q_lr_delay_mult,
                                                    max_steps=training_args.local_q_lr_max_steps)
        
        self.neural_phasefunc_scheduler_args = get_expon_lr_func(lr_init=training_args.neural_phasefunc_lr_init,
                                                    lr_final=training_args.neural_phasefunc_lr_final,
                                                    lr_delay_mult=training_args.neural_phasefunc_lr_delay_mult,
                                                    max_steps=training_args.neural_phasefunc_lr_max_steps)


    def update_learning_rate(self, iteration, asg_freeze_step=0, local_q_freeze_step=0, freeze_phasefunc_steps=0):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] in ["alpha_asg", "asg_sigma", "asg_rotation", "asg_scales"] :
                lr = self.asg_scheduler_args(max(0, iteration - asg_freeze_step))
                param_group['lr'] = lr
            if param_group["name"] == "local_q":
                lr = self.local_q_scheduler_args(max(0, iteration - local_q_freeze_step))
                param_group['lr'] = lr
            if param_group["name"] == ["neural_phasefunc", "neural_material"]:
                lr = self.neural_phasefunc_scheduler_args(max(0, iteration - freeze_phasefunc_steps))
                param_group['lr'] = lr


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.use_MBRDF:
            for i in range(self.kd.shape[1]):
                l.append('kd_{}'.format(i))
            for i in range(self.ks.shape[1]):
                l.append('ks_{}'.format(i))
            for i in range(self.alpha_asg.shape[1]):
                l.append('alpha_asg_{}'.format(i))
            for i in range(self.local_q.shape[1]):
                l.append('local_q_{}'.format(i))
            for i in range(self.neural_material.shape[1]):
                l.append('neural_material_{}'.format(i))

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        if self.use_MBRDF:
            kd = self.kd.detach().cpu().numpy()
            ks = self.ks.detach().cpu().numpy()
            alpha_asg = self.alpha_asg.detach().cpu().numpy()
            local_q = self.local_q.detach().cpu().numpy()
            neural_material = self.neural_material.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.use_MBRDF:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, \
                                        kd, ks, alpha_asg, local_q, neural_material), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        
        if self.use_MBRDF:
            kd_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("kd_")]
            kd_names = sorted(kd_names, key = lambda x: int(x.split('_')[-1]))
            kd = np.zeros((xyz.shape[0], len(kd_names)))
            for idx, attr_name in enumerate(kd_names):
                kd[:, idx] = np.asarray(plydata.elements[0][attr_name])
                
            ks_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ks_")]
            ks_names = sorted(ks_names, key = lambda x: int(x.split('_')[-1]))
            ks = np.zeros((xyz.shape[0], len(ks_names)))
            for idx, attr_name in enumerate(ks_names):
                ks[:, idx] = np.asarray(plydata.elements[0][attr_name])
                
            alpha_asg_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("alpha_asg_")]
            alpha_asg_names = sorted(alpha_asg_names, key = lambda x: int(x.split('_')[-1]))
            assert len(alpha_asg_names) == self.basis_asg_num
            alpha_asg = np.zeros((xyz.shape[0], len(alpha_asg_names)))
            for idx, attr_name in enumerate(alpha_asg_names):
                alpha_asg[:, idx] = np.asarray(plydata.elements[0][attr_name])
                
            local_q_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("local_q_")]
            local_q_names = sorted(local_q_names, key = lambda x: int(x.split('_')[-1]))
            local_q = np.zeros((xyz.shape[0], len(local_q_names)))
            for idx, attr_name in enumerate(local_q_names):
                local_q[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
            neural_material_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("neural_material_")]
            neural_material_names = sorted(neural_material_names, key = lambda x: int(x.split('_')[-1]))
            neural_materials = np.zeros((xyz.shape[0], len(neural_material_names)))
            for idx, attr_name in enumerate(neural_material_names):
                neural_materials[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        if self.use_MBRDF:
            self.kd = nn.Parameter(torch.tensor(kd, dtype=torch.float, device="cuda").requires_grad_(True))
            self.ks = nn.Parameter(torch.tensor(ks, dtype=torch.float, device="cuda").requires_grad_(True))
            self.alpha_asg = nn.Parameter(torch.tensor(alpha_asg, dtype=torch.float, device="cuda").requires_grad_(True))
            self.local_q = nn.Parameter(torch.tensor(local_q, dtype=torch.float, device="cuda").requires_grad_(True))
            self.neural_material = nn.Parameter(torch.tensor(neural_materials, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1 or "asg" == group["name"][:3]:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1 or "asg" == group["name"][:3]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.use_MBRDF:
            self.kd = optimizable_tensors["kd"]
            self.ks = optimizable_tensors["ks"]
            self.alpha_asg = optimizable_tensors["alpha_asg"]
            self.local_q = optimizable_tensors["local_q"]
            self.neural_material = optimizable_tensors["neural_material"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.out_weights_accum = self.out_weights_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1 or "asg" == group["name"][:3]:
                continue
            # assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_kd, new_ks, new_alpha_asg, local_q, new_neural_material):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        if self.use_MBRDF:
            d.update({
                "kd": new_kd,
                "ks": new_ks,
                "alpha_asg": new_alpha_asg,
                "local_q": local_q,
                "neural_material": new_neural_material
            })
            
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.use_MBRDF:
            self.kd = optimizable_tensors["kd"]
            self.ks = optimizable_tensors["ks"]
            self.alpha_asg = optimizable_tensors["alpha_asg"]
            self.local_q = optimizable_tensors["local_q"]
            self.neural_material = optimizable_tensors["neural_material"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.out_weights_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_kd = None if not self.use_MBRDF else self.kd[selected_pts_mask].repeat(N,1)
        new_ks = None if not self.use_MBRDF else self.ks[selected_pts_mask].repeat(N,1)
        new_alpha_asg = None if not self.use_MBRDF else self.alpha_asg[selected_pts_mask].repeat(N,1)
        new_local_q = None if not self.use_MBRDF else self.local_q[selected_pts_mask].repeat(N,1)
        new_neural_material = None if not self.use_MBRDF else self.get_neural_material[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, \
            new_kd, new_ks, new_alpha_asg, new_local_q, new_neural_material)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        
        return prune_filter

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_kd = None if not self.use_MBRDF else self.kd[selected_pts_mask]
        new_ks = None if not self.use_MBRDF else self.ks[selected_pts_mask]
        new_alpha_asg = None if not self.use_MBRDF else self.alpha_asg[selected_pts_mask]
        new_local_q = None if not self.use_MBRDF else self.local_q[selected_pts_mask]
        new_neural_material = None if not self.use_MBRDF else self.get_neural_material[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, \
            new_kd, new_ks, new_alpha_asg, new_local_q, new_neural_material)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        out_weights_acc = self.out_weights_accum

        self.densify_and_clone(grads, max_grad, extent)
        _prune_filter = self.densify_and_split(grads, max_grad, extent)
        
        out_weights_acc = out_weights_acc[~_prune_filter[:out_weights_acc.shape[0]]]
        padded_out_weights = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        padded_out_weights[:out_weights_acc.shape[0]] = out_weights_acc.squeeze()
        padded_out_weights[out_weights_acc.shape[0]:] = torch.max(out_weights_acc)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            
        out_weights_mask = self.prune_visibility_mask(padded_out_weights)
        prune_mask = torch.logical_or(prune_mask, out_weights_mask)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, width, height, out_weights):
        grad = viewspace_point_tensor.grad.squeeze(0) # [N, 2]
        # Normalize the gradient to [-1, 1] screen size
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.out_weights_accum += out_weights
        
    def prune_visibility_mask(self, out_weights_acc):
        n_before = self.get_xyz.shape[0]
        n_after = self.maximum_gs
        n_prune = n_before - n_after
        prune_mask = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
        if n_prune > 0:
            # Find the mask of top n_prune smallest `self.out_weights_accum`
            _, indices = torch.topk(out_weights_acc, n_prune, largest=False)
            prune_mask[indices] = True
        return prune_mask
