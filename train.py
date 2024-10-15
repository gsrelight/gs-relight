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

import os
import torch
import math
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, unfreeze_iterations, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, use_MBRDF=dataset.use_nerual_phasefunc, basis_asg_num=dataset.basis_asg_num, \
                            hidden_feature_size=dataset.phasefunc_hidden_size, hidden_feature_layer=dataset.phasefunc_hidden_layers, \
                            phase_frequency=dataset.phasefunc_frequency, neural_material_size=dataset.neural_material_size,
                            maximum_gs=dataset.maximum_gs)
    scene = Scene(dataset, gaussians, opt=opt, shuffle=True)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1, 1, 0, 0, 0] if dataset.white_background else [0, 0, 0, 0, 0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    prune_visibility = False
    viewpoint_stack = None
    opt_test = False
    opt_test_ready = False
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # 
    phase_func_freezed = False
    asg_freezed = True
    if first_iter < unfreeze_iterations:
        gaussians.neural_phasefunc.freeze()
        phase_func_freezed = True
        
    # initialize parallel GPU stream 
    light_stream = torch.cuda.Stream()
    calc_stream = torch.cuda.Stream()
    
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # update lr of asg
        gaussians.update_learning_rate(iteration, \
                                        asg_freeze_step=opt.asg_lr_freeze_step, \
                                        local_q_freeze_step=opt.local_q_lr_freeze_step, \
                                        freeze_phasefunc_steps=opt.freeze_phasefunc_steps)
        # opt camera or point light
        if scene.optimizing:
            scene.update_lr(iteration, \
                            freez_train_cam=opt.train_cam_freeze_step, \
                            freez_train_pl=opt.train_pl_freeze_step, \
                            cam_opt=dataset.cam_opt, \
                            pl_opt=dataset.pl_opt)
            
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            if not dataset.use_nerual_phasefunc:
                gaussians.oneupSHdegree()
                
        if iteration <= opt.asg_freeze_step:
            gaussians.asg_func.asg_scales.requires_grad_(False)
            gaussians.asg_func.asg_rotation.requires_grad_(False)
        elif asg_freezed:
            asg_freezed = False
            gaussians.asg_func.asg_scales.requires_grad_(True)
            gaussians.asg_func.asg_rotation.requires_grad_(True)
            print("set ansio param requires_grad: ", gaussians.asg_func.asg_scales.requires_grad)
        
        # Pick a random Camera
        if not viewpoint_stack:
            # only do pose opt for test sets
            if opt_test_ready and scene.optimizing:
                opt_test = True
                viewpoint_stack = scene.getTestCameras().copy()
                opt_test_ready = False
            else:
                opt_test = False
                viewpoint_stack = scene.getTrainCameras().copy()
                opt_test_ready = True
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((7), device="cuda") if opt.random_background else background
        
        # precompute shading frames and ASG frames
        local_axises = gaussians.get_local_axis # (K, 3, 3)
        asg_scales = gaussians.asg_func.get_asg_lam_miu # (basis_asg_num, 2)
        asg_axises = gaussians.asg_func.get_asg_axis    # (basis_asg_num, 3, 3)

        # only opt with diffuse term at the beginning for a stable training process
        if iteration < opt.spcular_freeze_step + opt.fit_linear_step:
            render_pkg = render(viewpoint_cam, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, pipe, bg, fix_labert=True, is_train=prune_visibility)
        else:
            render_pkg = render(viewpoint_cam, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, pipe, bg, is_train=prune_visibility)
        
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image, shadow, other_effects = render_pkg["render"], render_pkg["shadow"], render_pkg["other_effects"]
        
        if iteration <= unfreeze_iterations:
            image = image
        else:
            image = image * shadow + other_effects

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        if dataset.hdr:
            if iteration <= opt.spcular_freeze_step:
                gt_image = torch.pow(gt_image, 1./2.2)
            elif iteration < opt.spcular_freeze_step + opt.fit_linear_step//2:
                gamma = 1.1 * float(opt.spcular_freeze_step + opt.fit_linear_step - iteration + 1) / float(opt.fit_linear_step // 2 + 1)
                gt_image = torch.pow(gt_image, 1./gamma)
        else:
            image = torch.clip(image, 0.0, 1.0)

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()
            
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), \
                testing_iterations, scene, render, (pipe, background), gamma=2.2 if dataset.hdr else 1.0)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if opt_test and scene.optimizing:
                if iteration < opt.iterations:
                    scene.optimizer.step()
                    scene.optimizer.zero_grad(set_to_none = True)
                    # do not optimize the scene
                    gaussians.optimizer.zero_grad(set_to_none = True)
            else:
                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1], render_pkg["out_weight"])

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        if gaussians.get_xyz.shape[0] > gaussians.maximum_gs * 0.95:
                            prune_visibility = True
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
                else:
                    prune_visibility = False

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    # opt the camera pose
                    if scene.optimizing:
                        scene.optimizer.step()
                        scene.optimizer.zero_grad(set_to_none = True)
                    gaussians.optimizer.zero_grad(set_to_none = True)
                
            if phase_func_freezed and iteration >= unfreeze_iterations:
                gaussians.neural_phasefunc.unfreeze()
                phase_func_freezed = False

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if iteration == opt.spcular_freeze_step:
                gaussians.neural_phasefunc.freeze()
                gaussians.neural_material.requires_grad_(False)
            
            if iteration == opt.spcular_freeze_step + opt.fit_linear_step:
                gaussians.neural_phasefunc.unfreeze()
                gaussians.neural_material.requires_grad_(True)

        # update cam and light
        if scene.optimizing:
            viewpoint_cam.update("SO3xR3")
    

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, gamma=1.0):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        light_stream = torch.cuda.Stream()
        calc_stream = torch.cuda.Stream()
        local_axises = scene.gaussians.get_local_axis # (K, 3, 3)
        asg_scales = scene.gaussians.asg_func.get_asg_lam_miu # (basis_asg_num, sg_num, 2)
        asg_axises = scene.gaussians.asg_func.get_asg_axis    # (basis_asg_num, sg_num, 3, 3)
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, *renderArgs)
                    mimage, shadow, other_effects = render_pkg["render"], render_pkg["shadow"], render_pkg["other_effects"]
                    image = torch.clamp(mimage * shadow + other_effects, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None].pow(1./gamma), global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None].pow(1./gamma), global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--unfreeze_iterations", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.unfreeze_iterations, args.debug_from)

    # All done
    print("\nTraining complete.")
