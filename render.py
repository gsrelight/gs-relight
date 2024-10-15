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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.neural_phase_function import Neural_phase
from scene.mixture_ASG import Mixture_of_ASG

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, gamma):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    local_axises = gaussians.get_local_axis         # (K, 3, 3)
    asg_scales = gaussians.asg_func.get_asg_lam_miu # (basis_asg_num, 2)
    asg_axises = gaussians.asg_func.get_asg_axis    # (basis_asg_num, 3, 3)
        
    light_stream = torch.cuda.Stream()
    calc_stream = torch.cuda.Stream()
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, pipeline, background)
        rendering = render_pkg["render"] * render_pkg["shadow"] + render_pkg["other_effects"]
        gt = view.original_image[0:3, :, :]

        if gamma:
            gt = torch.pow(gt, 1/2.2)
            rendering = torch.pow(rendering, 1/2.2)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, 
                iteration : int, 
                pipeline : PipelineParams, 
                skip_train : bool, 
                skip_test : bool, 
                opt_pose: bool, 
                gamma: bool,
                valid: bool):
    dataset.data_device = "cpu"
    if opt_pose:
        dataset.source_path = os.path.join(dataset.model_path, f'point_cloud/iteration_{iteration}')

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.use_nerual_phasefunc, basis_asg_num=dataset.basis_asg_num)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, valid=valid)
        
        if dataset.use_nerual_phasefunc:
            _model_path = os.path.join(dataset.model_path, f"chkpnt{iteration}.pth")
            if os.path.exists(_model_path):
                (model_params, first_iter) = torch.load(_model_path)
                # load ASG parameters
                gaussians.asg_func = Mixture_of_ASG(dataset.basis_asg_num)
                gaussians.asg_func.asg_sigma = model_params[8]
                gaussians.asg_func.asg_rotation = model_params[9]
                gaussians.asg_func.asg_scales = model_params[10]
                # load MLP parameters
                gaussians.neural_phasefunc = Neural_phase(hidden_feature_size=dataset.phasefunc_hidden_size, \
                                        hidden_feature_layers=dataset.phasefunc_hidden_layers, \
                                        frequency=dataset.phasefunc_frequency, \
                                        neural_material_size=dataset.neural_material_size).to(device="cuda")
                gaussians.neural_phasefunc.load_state_dict(model_params[14])
                gaussians.neural_phasefunc.eval()
            else:
                raise Exception(f"Could not find : {_model_path}")

        bg_color = [1, 1, 1, 1, 0, 0, 0] if dataset.white_background else [0, 0, 0, 0, 0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, gamma)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, gamma)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gamma", action="store_true", default=False)
    parser.add_argument("--opt_pose", action="store_true", default=False)
    parser.add_argument("--valid", action="store_true", default=False)
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), \
                args.skip_train, args.skip_test, args.opt_pose, args.gamma, args.valid)