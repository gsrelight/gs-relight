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
import random
import json
from utils.system_utils import searchForMaxIteration
from utils.general_utils import  get_expon_lr_func
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import fov2focal

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, 
                 gaussians : GaussianModel, 
                 opt=None, 
                 load_iteration=None, 
                 shuffle=True, 
                 resolution_scales=[1.0], 
                 valid=False):
        """b
        :param path: Path to Blender scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.view_num, \
                                                           valid=valid, extension=".exr" if args.hdr else ".png")
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            
        if args.cam_opt or args.pl_opt:
            self.optimizing = True
        else:
            self.optimizing = False

        self.save_scale = 1.0
        if opt is not None:
            # optimizer for camera and light
            self.cam_scheduler_args = get_expon_lr_func(lr_init=opt.opt_cam_lr_init,
                                                    lr_final=opt.opt_cam_lr_final,
                                                    lr_delay_steps=opt.opt_cam_lr_delay_step,
                                                    lr_delay_mult=opt.opt_cam_lr_delay_mult,
                                                    max_steps=opt.opt_cam_lr_max_steps)
            
            self.pl_scheduler_args = get_expon_lr_func(lr_init=opt.opt_pl_lr_init,
                                                    lr_final=opt.opt_pl_lr_final,
                                                    lr_delay_steps=opt.opt_pl_lr_delay_step,
                                                    lr_delay_mult=opt.opt_pl_lr_delay_mult,
                                                    max_steps=opt.opt_pl_lr_max_steps)
            
            cam_params = []
            pl_params = []
            self.save_scale = resolution_scales[0]
            for scale in resolution_scales:
                cam_params.extend([self.train_cameras[scale][i].cam_pose_adj for i in range(len(self.train_cameras[scale]))])
                cam_params.extend([self.test_cameras[scale][i].cam_pose_adj for i in range(len(self.test_cameras[scale]))])
                pl_params.extend([self.train_cameras[scale][i].pl_adj for i in range(len(self.train_cameras[scale]))])
                pl_params.extend([self.test_cameras[scale][i].pl_adj for i in range(len(self.test_cameras[scale]))])

            if self.optimizing:
                self.optimizer = torch.optim.Adam(
                    [
                        {"params": cam_params, "lr" : 0.0, "name": "cam_adj"},
                        {"params": pl_params, "lr" : 0.0, "name": "pl_adj"}
                    ],
                    lr = 0,
                    eps=1e-15
                )
            else:
                self.optimizer = None

    # update lr rate 
    def update_lr(self, iteration, freez_train_cam, freez_train_pl, cam_opt, pl_opt):
        if self.optimizing:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "cam_adj":
                    if not cam_opt or iteration < freez_train_cam:
                        param_group['lr'] = 0
                    else:
                        lr = self.cam_scheduler_args(iteration)
                        param_group['lr'] = lr
                elif param_group["name"] == "pl_adj":
                    if not pl_opt or iteration < freez_train_pl:
                        param_group['lr'] = 0
                    else:
                        lr = self.pl_scheduler_args(iteration)
                        param_group['lr'] = lr

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        if self.optimizing:
            cx = self.train_cameras[self.save_scale][0].cx
            cy = self.train_cameras[self.save_scale][0].cy
            fx = fov2focal(self.train_cameras[self.save_scale][0].FoVx, self.train_cameras[self.save_scale][0].image_width)
            fy = fov2focal(self.train_cameras[self.save_scale][0].FoVy, self.train_cameras[self.save_scale][0].image_height)
            intrinsics = [cx, cy, fx, fy]
            # camera and light in train set
            cam_new = {'camera_intrinsics': intrinsics, "frames": []}
            for i in range(len(self.train_cameras[self.save_scale])):
                camnow = self.train_cameras[self.save_scale][i]
                R, T, pl_pos = camnow.get("SO3xR3")
                focalx = fov2focal(camnow.FoVx, camnow.image_width)
                focaly = fov2focal(camnow.FoVy, camnow.image_height)
                cam_new["frames"].append(
                {
                    "file_path": camnow.image_name,
                    "img_path": camnow.image_path,
                    "R_opt": R.tolist(),
                    "T_opt": T.tolist(),
                    "pl_pos": pl_pos[0].tolist(),
                    "camera_intrinsics": [camnow.cx, camnow.cy, focalx, focaly],
                })
            with open(os.path.join(point_cloud_path, "transforms_train.json"), "w") as outfile:
                json.dump(cam_new, outfile)
        
            # camera and light in test set
            cam_new = {'camera_intrinsics': intrinsics, "frames": []}
            for i in range(len(self.test_cameras[self.save_scale])):
                camnow = self.test_cameras[self.save_scale][i]
                R, T, pl_pos = camnow.get("SO3xR3")
                focalx = fov2focal(camnow.FoVx, camnow.image_width)
                focaly = fov2focal(camnow.FoVy, camnow.image_height)
                cam_new["frames"].append(
                {
                    "file_path": camnow.image_name,
                    "img_path": camnow.image_path,
                    "R_opt": R.tolist(),
                    "T_opt": T.tolist(),
                    "pl_pos": pl_pos[0].tolist(),
                    "camera_intrinsics": [camnow.cx, camnow.cy, focalx, focaly],
                })
            with open(os.path.join(point_cloud_path, "transforms_test.json"), "w") as outfile:
                json.dump(cam_new, outfile)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]