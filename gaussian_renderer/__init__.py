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
import torch.nn.functional as F
import numpy as np
import math

from gsplat import rasterization
from utils.graphics_utils import fov2focal
from diff_gaussian_rasterization_light import GaussianRasterizationSettings as GaussianRasterizationSettings_light
from diff_gaussian_rasterization_light import  GaussianRasterizer as GaussianRasterizer_light
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import getProjectionMatrix, look_at

def render(viewpoint_camera, 
           pc : GaussianModel, 
           light_stream, 
           calc_stream, 
           local_axises, 
           asg_scales, 
           asg_axises, 
           pipe, 
           bg_color : torch.Tensor, 
           scaling_modifier = 1.0, 
           override_color = None, 
           fix_labert = False, 
           inten_scale = 1.0,
           is_train = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # calculate the fov and projmatrix of light
    fx_origin = viewpoint_camera.image_width / (2. * tanfovx)
    fy_origin = viewpoint_camera.image_height / (2. * tanfovy)

    # calculate the fov for shadow splatting
    light_position = viewpoint_camera.pl_pos[0].detach().cpu().numpy()
    camera_position = viewpoint_camera.camera_center.detach().cpu().numpy()
    f_scale_ratio = np.sqrt(np.sum(light_position * light_position) / np.sum(camera_position * camera_position))
    
    fx_far = fx_origin * f_scale_ratio
    fy_far = fy_origin * f_scale_ratio
    
    tanfovx_far = 0.5 * viewpoint_camera.image_width / fx_far
    tanfovy_far = 0.5 * viewpoint_camera.image_height / fy_far

    fovx_far = 2 * math.atan(tanfovx_far)
    fovy_far = 2 * math.atan(tanfovy_far)

    # calculate the project matrix of shadow splatting
    object_center=pc.get_xyz.mean(dim=0).detach()
    world_view_transform_light=look_at(light_position,
                                       object_center.detach().cpu().numpy(),
                                       up_dir=np.array([0, 0, 1]))
    world_view_transform_light=torch.tensor(world_view_transform_light,
                                            device=viewpoint_camera.world_view_transform.device,
                                            dtype=viewpoint_camera.world_view_transform.dtype)
    light_prjection_matric = getProjectionMatrix(znear=viewpoint_camera.znear, zfar=viewpoint_camera.zfar, fovX=fovx_far, fovY=fovy_far).transpose(0,1).cuda()
    full_proj_transform_light = (world_view_transform_light.unsqueeze(0).bmm(light_prjection_matric.unsqueeze(0))).squeeze(0)
    
    raster_settings_light = GaussianRasterizationSettings_light(
        image_height = int(viewpoint_camera.image_height),
        image_width = int(viewpoint_camera.image_width),
        tanfovx = tanfovx_far,
        tanfovy = tanfovy_far,
        bg = bg_color[:3],
        scale_modifier = scaling_modifier,
        viewmatrix = world_view_transform_light,
        projmatrix = full_proj_transform_light,
        sh_degree = pc.active_sh_degree,
        campos = viewpoint_camera.pl_pos[0],
        prefiltered = False,
        debug = pipe.debug,
    )
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # shadow splatting
    light_stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad():
        with torch.cuda.stream(light_stream):
            rasterizer_light = GaussianRasterizer_light(raster_settings=raster_settings_light)
            opcacity_light = torch.zeros(scales.shape[0], dtype=torch.float32, device=scales.device)
            _, out_weight, _, shadow = rasterizer_light(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = torch.zeros((2, 3), dtype=torch.float32, device=scales.device),
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp,
                non_trans = opcacity_light,
                offset = 0.015,
                thres = 4,
                is_train = is_train)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pc.use_MBRDF:
            with torch.cuda.stream(calc_stream):
                assert viewpoint_camera.pl_pos.shape[0] == 1
                # calculate view and light dirs
                pl_pos_expand = viewpoint_camera.pl_pos.expand(pc.get_xyz.shape[0], -1) # (K, 3)
                wi_ray = pl_pos_expand - pc.get_xyz # (K, 3)
                dist_2_inv = 1.0 / torch.sum(wi_ray**2, dim=-1, keepdim=True)
                wi = wi_ray * torch.sqrt(dist_2_inv) # (K, 3)
                wo = _safe_normalize(viewpoint_camera.camera_center - pc.get_xyz) # (K, 3)

                local_z = local_axises[:, :, 2] # (K, 3)
                # transfer to local axis
                wi_local = torch.einsum('Ki,Kij->Kj', wi, local_axises) # (K, 3)
                wo_local = torch.einsum('Ki,Kij->Kj', wo, local_axises) # (K, 3)
                # shading function
                cosTheta = _NdotWi(local_z, wi, torch.nn.ELU(alpha=0.01), 0.01)
                diffuse = pc.get_kd / math.pi
                specular = pc.get_ks * pc.asg_func(wi_local, wo_local, pc.get_alpha_asg, asg_scales, asg_axises) # (K, 3)
            
                if fix_labert:
                    colors_precomp = diffuse
                else:
                    colors_precomp = diffuse + specular 
                # intensity decays with distance
                colors_precomp = colors_precomp * cosTheta * dist_2_inv

            # calc_stream.wait_stream(light_stream)
            torch.cuda.current_stream().wait_stream(light_stream)
            torch.cuda.current_stream().wait_stream(calc_stream)
            
            # shaodow splat values
            opcacity_light = torch.clamp_min(opcacity_light, 1e-6)
            shadow = shadow / opcacity_light # (K,)
            assert not torch.isnan(shadow).any()
            
            # neural components
            decay, other_effects = pc.neural_phasefunc(wi, wo, pc.get_xyz, pc.get_neural_material, shadow.unsqueeze(-1)) # (K, 1), (K, 3)
            
            colors_precomp = torch.concat([colors_precomp * inten_scale, decay, other_effects * dist_2_inv * inten_scale], dim=-1) # (K, 7)
        elif pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    torch.cuda.synchronize()
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 

    focalx = fov2focal(viewpoint_camera.FoVx, viewpoint_camera.image_width)
    focaly = fov2focal(viewpoint_camera.FoVy, viewpoint_camera.image_height)
    K = torch.tensor([[focalx, 0, viewpoint_camera.cx], [0, focaly, viewpoint_camera.cy], [0., 0., 1.]], device="cuda")
    rendered_image, alphas, meta = rasterization(
        means = means3D, # [N, 3]
        quats = rotations, # [N, 4]
        scales = scales, # [N, 3]
        opacities = opacity.squeeze(-1), # [N]
        colors = colors_precomp, # [N, 7]
        viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None, ...], # [1, 4, 4]
        Ks = K[None, ...], # [1, 3, 3]
        width = int(viewpoint_camera.image_width),
        height = int(viewpoint_camera.image_height),
        near_plane = viewpoint_camera.znear,
        far_plane = viewpoint_camera.zfar,
        eps2d = 0.3,
        sh_degree = None,
        packed = False,
        backgrounds = bg_color[None, ...]
    )

    # The intermediate results from fully_fused_projection
    rendered_image = rendered_image[0].permute(2, 0, 1)
    radii = meta['radii'].squeeze(0)
    try:
        meta["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    torch.cuda.synchronize()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return {"render": rendered_image[0:3, :, :],
            "shadow": rendered_image[3:4, :, :],
            "other_effects": rendered_image[4:7, :, :],
            "viewspace_points": meta["means2d"],
            "visibility_filter" : radii > 0,
            "radii": radii,
            "out_weight": out_weight}

def _dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)

def _safe_normalize(x):
    return torch.nn.functional.normalize(x, dim = -1, eps=1e-8)

def _NdotWi(nrm, wi, elu, a):
    """
    nrm: (K, 3)
    wi: (K, 3)
    return (K, 1)
    """
    tmp  = a * (1. - 1 / math.e)
    return (elu(_dot(nrm, wi)) + tmp) / (1. + tmp)
