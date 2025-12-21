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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud



class CameraInfo(NamedTuple):#这是一个命名元组 (NamedTuple)，用于存储单个相机的信息。
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int #
    height: int #

class SceneInfo(NamedTuple):#命名元组，用于存储场景信息
    point_cloud: BasicPointCloud
    train_cameras: list#训练集中的相机列表
    test_cameras: list
    nerf_normalization: dict#用于 NeRF 归一化的字典？
    ply_path: str#存储点云的 PLY 文件路径

def getNerfppNorm(cam_info):#什么意思呢？#此函数用于计算相机的中心和对角线长度，以便进行归一化处理。
#内部函数 get_center_and_diag 计算所有相机中心的均值和距离的最大值（对角线长度）。
#输入参数 cam_info 是相机的信息列表，返回值是包含 translate 和 radius 的字典。
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

    center, diagonal = get_center_and_diag(cam_centers)#得到相机的中心和对角线长度
    radius = diagonal * 1.1#半径等于对角线×1.1?

    translate = -center#-的相机中心？

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
#从 COLMAP 输出读取相机信息。
#处理相机的外部和内部参数，生成 CameraInfo 对象并返回相机列表。
#根据不同的相机模型（简单针孔或针孔）计算视场角。
    cam_infos = []#返回的值
    for idx, key in enumerate(cam_extrinsics):
#在 COLMAP 等 3D 重建工具中，cam_extrinsics 通常是一个字典或其他数据结构，
# 键值对中包含每个相机的 ID 以及对应的外部参数（旋转和位移）。
# 这些参数用于将三维场景中的点从世界坐标系转换到相机坐标系，以便进行图像渲染或其他后续处理。
        sys.stdout.write('\r')#清除行首：在控制台输出中清除当前行的内容，以便更新进度信息
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))#输出进度：打印当前读取的相机索引和总相机数量。
        sys.stdout.flush()#刷新输出：确保进度信息立即显示在控制台。

        extr = cam_extrinsics[key]#获取当前相机的外部参数：根据相机的唯一标识符 key 从 cam_extrinsics 中获取相应的外部参数。
        intr = cam_intrinsics[extr.camera_id]#获取当前相机的外部参数：根据相机的唯一标识符 key 从 cam_extrinsics 中获取相应的外部参数。
        height = intr.height#提取图像尺寸：从内部参数中提取图像的高度和宽度
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))#获取旋转矩阵：将四元数 (extr.qvec) 转换为旋转矩阵，并转置。
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model in ["SIMPLE_RADIAL", "RADIAL"]:
            # SIMPLE_RADIAL: f, cx, cy, k1
            # RADIAL: f, cx, cy, k1, k2
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV_FISHEYE":
            # OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "FULL_OPENCV":
            # FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "SIMPLE_RADIAL_FISHEYE":
            # SIMPLE_RADIAL_FISHEYE: f, cx, cy, k
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "RADIAL_FISHEYE":
            # RADIAL_FISHEYE: f, cx, cy, k1, k2
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "THIN_PRISM_FISHEYE":
            # THIN_PRISM_FISHEYE: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            # 对于未知模型，尝试使用第一个参数作为焦距
            print(f"Warning: Unknown camera model '{intr.model}', attempting to use first param as focal length")
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        #处理未支持的相机模型：如果相机模型不在支持的范围内，触发断言并输出错误信息。

        image_path = os.path.join(images_folder, os.path.basename(extr.name))#构建图像路径：根据图像文件夹和外部参数中的图像名称构建完整路径。
        # #Join函数是Python中字符串类（str）的一个方法，用于将一个字符串列表（或其他可迭代对象）中的所有字符串连接起来。
        image_name = os.path.basename(image_path).split(".")[0]#提取图像名称：从图像路径中提取文件名（不带扩展名）。
        image = Image.open(image_path)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
        #将创建的 CameraInfo 对象添加到 cam_infos 列表中。
    sys.stdout.write('\n')#在进度输出后添加换行符，以便于格式化输出。
    return cam_infos

def fetchPly(path):#fetchPly 从 PLY 文件中读取点云数据，返回 BasicPointCloud 对象。
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)#返回基础点云

def storePly(path, xyz, rgb):#storePly 将三维点云数据和颜色信息存储为 PLY 文件。
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

def readColmapSceneInfo(path, images, eval, llffhold=8):#一个用于选择训练/测试相机的参数 (llffhold，默认值为 8)。
#读取 COLMAP 场景信息，包括相机外部和内部参数。
#生成训练和测试相机的信息列表，并计算 NeRF 归一化参数。
#如果 PLY 文件不存在，则从二进制文件转换生成点云。
    try:#尝试构建 COLMAP 输出中相机外部参数和内部参数的二进制文件路径。
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:#如果读取二进制文件失败，构建文本文件的路径作为备选方案。
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images#设置读取目录：如果未提供图像文件夹路径，则使用默认的 images 目录。
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    #读取相机信息：调用 readColmapCameras 函数，传入相机的外部和内部参数以及图像文件夹路径，获取未排序的相机信息列表。

    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    #排序相机信息：根据图像名称对相机信息列表进行排序，生成一个新的列表 cam_infos。为啥要排序？
    if eval:#判断是否评估模式：如果 eval 为 True，则从 cam_infos 中筛选出训练相机信息，选择索引不是 llffhold 倍数的相机。
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]#虽然 llffhold 用于划分训练和测试集，但这并不是 k 折交叉验证，而是一种简单的按索引选择的策略。
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]#筛选测试相机信息：选择索引是 llffhold 倍数的相机作为测试集。
    else:#如果 eval 为 False，则将所有相机信息视为训练集，测试集为空。
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)#调用 getNerfppNorm 函数，传入训练相机信息，获取归一化参数。

    ply_path = os.path.join(path, "sparse/0/points3D.ply")#构建 PLY、二进制和文本格式的点云文件路径。
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):#如果 PLY 文件不存在，则进行转换。
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")#提示用户正在将点云从二进制格式转换为 PLY 格式。
        try:#从二进制文件中读取三维点云的坐标和颜色信息。
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:#如果读取二进制文件失败，则尝试从文本文件中读取点云数据。
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)#将读取到的三维坐标和颜色信息存储为 PLY 文件。
    try:
        pcd = fetchPly(ply_path)#读取刚存储的 PLY 文件，获取点云数据。
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info#将点云、训练和测试相机信息、归一化参数以及 PLY 文件路径封装到 SceneInfo 对象中。

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):#从 JSON 文件读取 NeRF 转换信息，生成相机信息。
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):#读取 NeRF 合成数据的相机信息，并生成随机点云（如果没有 COLMAP 数据）。
    #接受场景路径(path)、是否使用白色背景(white_background)、评估标志(eval)，以及图像文件扩展名（默认为.png）作为输入。
    print("Reading Training Transforms")#提示用户正在读取训练集的转换信息。
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)#调用 readCamerasFromTransforms 函数，从指定路径读取训练集相机的变换信息，返回相机信息列表 train_cam_infos。
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    #读取测试相机信息：调用 readCamerasFromTransforms 函数，从指定路径读取测试集相机的变换信息，返回相机信息列表 test_cam_infos。
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)#调用 getNerfppNorm 函数，传入训练相机信息，获取归一化参数。

    ply_path = os.path.join(path, "points3d.ply")#在指定路径下构建点云文件的 PLY 文件路径。
    if not os.path.exists(ply_path):#检查 PLY 文件是否存在：如果 PLY 文件不存在，则生成点云。
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000#设置点云点数：定义生成的随机点的数量，这里为 100,000。
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3#生成随机点坐标：在 [-1.3, 1.3] 范围内生成随机的三维点坐标。
        shs = np.random.random((num_pts, 3)) / 255.0#生成随机颜色：生成随机的颜色值，范围在 [0, 1] 之间（假设原始值在 [0, 255] 之间）。
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))#创建基本点云对象：使用生成的随机点坐标和颜色创建一个 BasicPointCloud 对象，法线设置为零向量。

        storePly(ply_path, xyz, SH2RGB(shs) * 255)#存储 PLY 文件：将生成的三维坐标和颜色（转换回 [0, 255] 范围）存储为 PLY 文件
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
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
#"Colmap"：这个键对应的值是 readColmapSceneInfo 函数，表示当需要加载 COLMAP 格式的场景信息时，调用此函数。
#：这个键对应的值是 readNerfSyntheticInfo 函数，表示当需要加载 Blender 格式或 NeRF 合成数据时，调用此函数