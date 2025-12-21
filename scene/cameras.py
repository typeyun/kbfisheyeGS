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

import math
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
        # ===== 新增：鱼眼 / KB 模型参数（可选，默认关闭） =====
        fisheye: bool = False,
        kb_k1: float = 0.0,
        kb_k2: float = 0.0,
        kb_k3: float = 0.0,
        kb_k4: float = 0.0,
        kb_max_theta: float = math.pi / 2.0,  # 常见取值：90°
        cx: float = 0.0,
        cy: float = 0.0,
        fx: float = 0.0,
        fy: float = 0.0,
        # =================================================
    ):
        super(Camera, self).__init__()

        self.uid = uid  # 区分同一位姿的不同照片
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        # ---- 设备选择 ----
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        # ---- 影像与 Alpha ----
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        # ---- 近远裁剪面 ----
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # ---- 经典 pinhole 相关矩阵（即便 fisheye=True 也保留，用于通用裁剪/可视化等）----
        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale)
        ).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # ===== 鱼眼 / KB 参数（存到相机对象，渲染层会透传到 CUDA）=====
        self.fisheye = bool(fisheye)
        self.kb_k1 = float(kb_k1)
        self.kb_k2 = float(kb_k2)
        self.kb_k3 = float(kb_k3)
        self.kb_k4 = float(kb_k4)
        self.kb_max_theta = float(kb_max_theta)  # 弧度
        self.cx = float(cx)
        self.cy = float(cy)
        self.fx = float(fx)
        self.fy = float(fy)
        # ===========================================================


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
        # ===== 与 Camera 保持一致的可选鱼眼 / KB 参数 =====
        fisheye: bool = False,
        kb_k1: float = 0.0,
        kb_k2: float = 0.0,
        kb_k3: float = 0.0,
        kb_k4: float = 0.0,
        kb_max_theta: float = math.pi / 2.0,
        cx: float = 0.0,
        cy: float = 0.0,
        fx: float = 0.0,
        fy: float = 0.0,
        # =========================================================
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

        # 保持与 Camera 一致的鱼眼/KB字段，便于下游统一读取
        self.fisheye = bool(fisheye)
        self.kb_k1 = float(kb_k1)
        self.kb_k2 = float(kb_k2)
        self.kb_k3 = float(kb_k3)
        self.kb_k4 = float(kb_k4)
        self.kb_max_theta = float(kb_max_theta)
        self.cx = float(cx)
        self.cy = float(cy)
        self.fx = float(fx)
        self.fy = float(fy)

