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
os.environ['OPENCV_IO_ENABLE_OPENEXR']="1"
import sys
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import cv2 as cv
from tqdm import tqdm


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    cx: float
    cy: float
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    pl_intensity: np.array
    pl_pos: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readCamerasFromTransforms(path, transformsfile, white_background, view_num, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        cx, cy = None, None
        if "camera_angle_x" in contents.keys():
            fovx = contents["camera_angle_x"]
        else:
            # intrinsics of real capture
            intrinsics = contents['camera_intrinsics']
            cx = intrinsics[0]
            cy = intrinsics[1]
            fx = intrinsics[2]
            fy = intrinsics[3]


        frames = contents["frames"]
        for idx, frame in tqdm(list(enumerate(frames))):

            if view_num > 0 and idx >= view_num:
                break

            if "img_path" in frame.keys():
                cam_name = frame["img_path"]
            else:
                cam_name = os.path.join(path, frame["file_path"] + extension)

            # load per cam intrinsic (for lightstage)
            if "camera_intrinsics" in frame.keys():
                intrinsics = frame['camera_intrinsics']
                cx = intrinsics[0]
                cy = intrinsics[1]
                fx = intrinsics[2]
                fy = intrinsics[3]

            if "R_opt" in frame.keys():
                # load opt pose
                R = np.asarray(frame["R_opt"])
                T = np.asarray(frame["T_opt"])
            else:
                # NeRF 'transform_matrix' is a camera-to-world transform
                c2w = np.array(frame["transform_matrix"])
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            if extension == ".png":
                image = Image.open(image_path)
                im_data = np.array(image.convert("RGBA"))
                norm_data = im_data / 255.0
            elif extension == ".exr":
                image = cv.imread(image_path, cv.IMREAD_UNCHANGED | cv.IMREAD_ANYDEPTH | cv.IMREAD_ANYCOLOR)
                norm_data = cv.cvtColor(image, cv.COLOR_BGRA2RGBA)
            else:
                raise Exception(f"Could not support : {extension}")
            
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            
            if extension == ".png":
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                if "camera_angle_x" in contents.keys():
                    fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                    FovY = fovy 
                    FovX = fovx
                else:
                    FovY = focal2fov(fy, image.size[1])
                    FovX = focal2fov(fx, image.size[0])

                if cx == None:
                    cx = image.size[0] / 2.
                if cy == None:
                    cy = image.size[1] / 2.

                cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, cx=cx, cy=cy, image=image,
                                image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                                pl_intensity=None if "pl_intensity" not in frame.keys() else frame["pl_intensity"],
                                pl_pos=None if frame["pl_pos"] is None else frame["pl_pos"]))
            else:
                image = arr.astype(np.float32)
                if "camera_angle_x" in contents.keys():
                    fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[0])
                    FovY = fovy 
                    FovX = fovx
                else:
                    FovY = focal2fov(fy, image.shape[0])
                    FovX = focal2fov(fx, image.shape[1])

                if cx == None:
                    cx = image.shape[1] / 2.
                if cy == None:
                    cy = image.shape[0] / 2.

                cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, cx=cx, cy=cy, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[1], height=image.shape[0],
                                pl_intensity=None if "pl_intensity" not in frame.keys() else frame["pl_intensity"],
                                pl_pos=None if frame["pl_pos"] is None else frame["pl_pos"]))

            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, view_num, valid=False, extension=".png"):
    print("json path: ", path)
    if valid:
        # Only used for visualization, we use 400 frames for visualization
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, 400, extension)
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_valid.json", white_background, 400, extension)
    else:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, view_num, extension)
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, view_num, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 1.0 - 0.5
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Blender" : readNerfSyntheticInfo
}