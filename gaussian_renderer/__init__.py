# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import math
import torch
from typing import Optional, Dict, Any

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


@torch.no_grad()
def _to_bg_color(background: Optional[torch.Tensor], device: torch.device):
    """
    规范背景色到 [3] 的 float 张量（0~1），默认黑色。
    """
    if background is None:
        return torch.zeros(3, device=device, dtype=torch.float32)
    if isinstance(background, (tuple, list)):
        return torch.tensor(background, device=device, dtype=torch.float32)
    if isinstance(background, torch.Tensor):
        bg = background.to(device=device, dtype=torch.float32).flatten()
        if bg.numel() == 1:
            return bg.repeat(3)
        return bg[:3]
    raise TypeError("Unsupported background type for renderer.")


def _get_cam_or_pipe(attr: str, cam: Any, pipe: Any, default=None):
    """
    相机优先，管道兜底，最后使用 default。
    """
    if hasattr(cam, attr):
        v = getattr(cam, attr)
        if v is not None:
            return v
    if hasattr(pipe, attr):
        v = getattr(pipe, attr)
        if v is not None:
            return v
    return default


def render(
    viewpoint_camera,                # Camera 或 MiniCam
    pc: GaussianModel,               # 场景中的高斯模型
    pipe,                            # PipelineParams（含 fisheye/KB 兜底参数）
    bg_color: Optional[torch.Tensor] = None,
    scaling_modifier: float = 1.0,
    override_color: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    渲染单视角。
    返回：
        {
          "render": [H,W,3] 0~1,
          "viewspace_points": means2D (屏幕空间点；由 CUDA 内核写回),
          "visibility_filter": radii > 0,
          "radii": [N] 每个高斯的屏幕半径
        }
    """
    device = pc.get_xyz.device

    # ---- 背景色 ----
    bg_color = _to_bg_color(bg_color, device)

    # ---- FOV & 视角几何（pinhole 路径用；fisheye 时 CUDA 将走 KB）----
    # 注意：FoV 单位为弧度，tanfov = tan(FoV/2)
    tanfovx = math.tan(float(getattr(viewpoint_camera, "FoVx")))
    tanfovy = math.tan(float(getattr(viewpoint_camera, "FoVy")))

    # ---- 相机优先，管道兜底：fish/KB 参数 ----
    fisheye = bool(_get_cam_or_pipe("fisheye", viewpoint_camera, pipe, False))
    kb_k1 = float(_get_cam_or_pipe("kb_k1", viewpoint_camera, pipe, 0.0))
    kb_k2 = float(_get_cam_or_pipe("kb_k2", viewpoint_camera, pipe, 0.0))
    kb_k3 = float(_get_cam_or_pipe("kb_k3", viewpoint_camera, pipe, 0.0))
    kb_k4 = float(_get_cam_or_pipe("kb_k4", viewpoint_camera, pipe, 0.0))
    kb_max_theta = float(_get_cam_or_pipe("kb_max_theta", viewpoint_camera, pipe, math.pi / 2.0))
    cx = float(_get_cam_or_pipe("cx", viewpoint_camera, pipe, 0.0))
    cy = float(_get_cam_or_pipe("cy", viewpoint_camera, pipe, 0.0))
    fx = float(_get_cam_or_pipe("fx", viewpoint_camera, pipe, 0.0))
    fy = float(_get_cam_or_pipe("fy", viewpoint_camera, pipe, 0.0))

    # 简单的运行时检查（避免 fisheye=True 但 fx/fy = 0）
    if fisheye and (fx == 0.0 or fy == 0.0):
        raise ValueError(
            "fisheye=True 但 fx/fy 为 0。请在相机或 Pipeline 参数中设置有效的像素焦距 fx、fy。"
        )

    # ---- Rasterization Settings（含 fisheye/KB）----
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=float(scaling_modifier),
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=bool(getattr(pipe, "debug", False)),

        # === 新增：鱼眼 / KB 畸变参数（相机优先，管道兜底）===
        fisheye=fisheye,
        kb_params=torch.tensor([kb_k1, kb_k2, kb_k3, kb_k4], device=device, dtype=torch.float32),
        max_theta=float(kb_max_theta),
        cx=float(cx), cy=float(cy),
        fx=float(fx), fy=float(fy),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # ---- 取高斯参数 ----
    means3D = pc.get_xyz                          # [N,3]
    opacity = pc.get_opacity                      # [N,1]
    cov3D_precomp = None
    scales = None
    rotations = None

    # 若选择 Python 端预计算 3D 协方差，则不传 scales/rotations（由 cov3D_precomp 直用）
    if getattr(pipe, "compute_cov3D_python", False):
        cov3D_precomp = pc.get_covariance(float(scaling_modifier))
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # ---- 颜色：可选 override / Python SH 转换 / 交由 CUDA ----
    shs = None
    colors_precomp = None
    if override_color is not None:
        colors_precomp = override_color.to(device=device, dtype=torch.float32)
    else:
        if getattr(pipe, "convert_SHs_python", False):
            # [N,3, (deg+1)^2] 视向量取 camera_center 到 xyz
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            # 方向：从点指向相机（或相反，视你仓库定义；保持与原实现一致）
            dir_to_cam = (means3D - viewpoint_camera.camera_center).detach()
            dir_to_cam = dir_to_cam / (dir_to_cam.norm(dim=1, keepdim=True) + 1e-8)
            colors_precomp = torch.clamp(
                eval_sh(pc.active_sh_degree, shs_view, dir_to_cam), min=0.0, max=1.0
            )
        else:
            shs = pc.get_features

    # ---- means2D 作为“输出缓冲”传入（CUDA 内核会写回屏幕坐标）----
    # 这里创建一个占位 tensor；内核将用屏幕坐标覆盖它。
    means2D = torch.zeros((means3D.shape[0], 2), device=device, dtype=means3D.dtype)

    # ---- 调用 CUDA 光栅化器 ----
    # 返回的 color 即渲染图像；radii>0 的作为可见性过滤器（具体行为依赖你的 CUDA 实现）
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # ---- 打包输出 ----
    screenspace_points = means2D                   # CUDA 写回后的屏幕空间点
    visibility = radii > 0

    return {
        "render": rendered_image,                  # [H,W,3]
        "viewspace_points": screenspace_points,    # [N,2]
        "visibility_filter": visibility,           # [N] bool
        "radii": radii,                            # [N]
    }
