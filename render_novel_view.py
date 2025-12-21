"""
渲染任意新视角的脚本
支持自定义相机位置、朝向、视角参数
"""

import torch
import numpy as np
import os
from argparse import ArgumentParser
from scene import Scene
from gaussian_renderer import GaussianModel, render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams
from scene.cameras import Camera
import torchvision
from PIL import Image
import math


def create_camera_from_pose(
    position,           # 相机位置 [x, y, z]
    look_at,            # 相机朝向点 [x, y, z]
    up_vector,          # 上方向 [x, y, z]
    fov_x,              # 水平视场角（度）
    fov_y,              # 垂直视场角（度）
    width,              # 图像宽度
    height,             # 图像高度
    device="cuda",
    fisheye=False,      # 是否使用鱼眼模型
    kb_params=None,     # KB模型参数字典
):
    """
    从相机位置和朝向创建Camera对象
    
    参数:
        position: 相机在世界坐标系中的位置 [x, y, z]
        look_at: 相机朝向的目标点 [x, y, z]
        up_vector: 相机的上方向向量 [x, y, z]
        fov_x: 水平视场角（度）
        fov_y: 垂直视场角（度）
        width: 渲染图像宽度
        height: 渲染图像高度
        device: 计算设备
        fisheye: 是否使用鱼眼相机模型
        kb_params: KB模型参数 {'k1', 'k2', 'k3', 'k4', 'max_theta', 'cx', 'cy', 'fx', 'fy'}
    """
    # 转换为numpy数组
    position = np.array(position, dtype=np.float32)
    look_at = np.array(look_at, dtype=np.float32)
    up_vector = np.array(up_vector, dtype=np.float32)
    
    # 计算相机坐标系
    # Z轴：从相机指向目标点的反方向（OpenGL约定）
    z_axis = position - look_at
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # X轴：上方向与Z轴的叉积
    x_axis = np.cross(up_vector, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Y轴：Z轴与X轴的叉积
    y_axis = np.cross(z_axis, x_axis)
    
    # 构建旋转矩阵 R (世界坐标系到相机坐标系)
    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    
    # 平移向量 T
    T = position
    
    # 创建虚拟图像（全黑）
    dummy_image = torch.zeros((3, height, width), dtype=torch.float32)
    
    # 转换FOV为弧度
    fov_x_rad = fov_x * math.pi / 180.0
    fov_y_rad = fov_y * math.pi / 180.0
    
    # 准备KB参数
    kb_dict = {
        'fisheye': fisheye,
        'kb_k1': 0.0,
        'kb_k2': 0.0,
        'kb_k3': 0.0,
        'kb_k4': 0.0,
        'kb_max_theta': math.pi / 2.0,
        'cx': width / 2.0,
        'cy': height / 2.0,
        'fx': width / (2.0 * math.tan(fov_x_rad / 2.0)),
        'fy': height / (2.0 * math.tan(fov_y_rad / 2.0)),
    }
    
    if kb_params is not None:
        kb_dict.update(kb_params)
    
    # 创建Camera对象
    camera = Camera(
        colmap_id=0,
        R=R,
        T=T,
        FoVx=fov_x_rad,
        FoVy=fov_y_rad,
        image=dummy_image,
        gt_alpha_mask=None,
        image_name="novel_view",
        uid=0,
        data_device=device,
        **kb_dict
    )
    
    return camera


def render_novel_view(
    model_path,
    iteration,
    position,
    look_at,
    up_vector=[0, 1, 0],
    fov_x=60.0,
    fov_y=45.0,
    width=800,
    height=600,
    output_path="novel_view.png",
    white_background=False,
    fisheye=False,
    kb_params=None,
):
    """
    渲染一个新视角
    
    参数:
        model_path: 训练好的模型路径
        iteration: 使用的迭代次数（-1表示最新）
        position: 相机位置 [x, y, z]
        look_at: 相机朝向点 [x, y, z]
        up_vector: 上方向向量 [x, y, z]
        fov_x: 水平视场角（度）
        fov_y: 垂直视场角（度）
        width: 图像宽度
        height: 图像高度
        output_path: 输出图像路径
        white_background: 是否使用白色背景
        fisheye: 是否使用鱼眼模型
        kb_params: KB模型参数
    """
    print(f"加载模型: {model_path}")
    print(f"迭代次数: {iteration}")
    print(f"相机位置: {position}")
    print(f"朝向点: {look_at}")
    print(f"图像尺寸: {width}x{height}")
    print(f"视场角: FOV_X={fov_x}°, FOV_Y={fov_y}°")
    if fisheye:
        print(f"使用鱼眼相机模型")
        if kb_params:
            print(f"KB参数: {kb_params}")
    
    # 初始化高斯模型
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    
    # 设置模型路径
    args_list = ['--model_path', model_path]
    args = parser.parse_args(args_list)
    
    with torch.no_grad():
        # 加载高斯模型
        gaussians = GaussianModel(model_params.extract(args).sh_degree)
        scene = Scene(model_params.extract(args), gaussians, load_iteration=iteration, shuffle=False)
        
        # 设置背景颜色
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 创建相机
        camera = create_camera_from_pose(
            position=position,
            look_at=look_at,
            up_vector=up_vector,
            fov_x=fov_x,
            fov_y=fov_y,
            width=width,
            height=height,
            fisheye=fisheye,
            kb_params=kb_params,
        )
        
        # 渲染
        print("开始渲染...")
        rendering = render(camera, gaussians, pipeline_params.extract(args), background)["render"]
        
        # 保存图像
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        torchvision.utils.save_image(rendering, output_path)
        print(f"渲染完成！图像已保存到: {output_path}")
        
        return rendering


def render_orbit_views(
    model_path,
    iteration,
    center_point,
    radius,
    num_views=36,
    height_offset=0.0,
    fov_x=60.0,
    fov_y=45.0,
    width=800,
    height=600,
    output_dir="orbit_views",
    white_background=False,
    fisheye=False,
    kb_params=None,
):
    """
    渲染环绕视角（相机围绕中心点旋转）
    
    参数:
        model_path: 训练好的模型路径
        iteration: 使用的迭代次数
        center_point: 环绕中心点 [x, y, z]
        radius: 环绕半径
        num_views: 生成的视角数量
        height_offset: 相机高度偏移
        其他参数同 render_novel_view
    """
    print(f"生成 {num_views} 个环绕视角")
    print(f"中心点: {center_point}, 半径: {radius}, 高度偏移: {height_offset}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    center = np.array(center_point)
    
    for i in range(num_views):
        angle = 2 * math.pi * i / num_views
        
        # 计算相机位置（在XZ平面上环绕）
        x = center[0] + radius * math.cos(angle)
        z = center[2] + radius * math.sin(angle)
        y = center[1] + height_offset
        
        position = [x, y, z]
        output_path = os.path.join(output_dir, f"view_{i:04d}.png")
        
        print(f"\n渲染视角 {i+1}/{num_views}")
        render_novel_view(
            model_path=model_path,
            iteration=iteration,
            position=position,
            look_at=center_point,
            up_vector=[0, 1, 0],
            fov_x=fov_x,
            fov_y=fov_y,
            width=width,
            height=height,
            output_path=output_path,
            white_background=white_background,
            fisheye=fisheye,
            kb_params=kb_params,
        )
    
    print(f"\n所有视角渲染完成！保存在: {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="渲染任意新视角")
    
    # 基本参数
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--iteration", type=int, default=-1, help="使用的迭代次数（-1表示最新）")
    parser.add_argument("--output", type=str, default="novel_view.png", help="输出图像路径")
    
    # 相机参数
    parser.add_argument("--position", type=float, nargs=3, default=[0, 0, 5], help="相机位置 [x y z]")
    parser.add_argument("--look_at", type=float, nargs=3, default=[0, 0, 0], help="相机朝向点 [x y z]")
    parser.add_argument("--up", type=float, nargs=3, default=[0, 1, 0], help="上方向向量 [x y z]")
    
    # 视角参数
    parser.add_argument("--fov_x", type=float, default=60.0, help="水平视场角（度）")
    parser.add_argument("--fov_y", type=float, default=45.0, help="垂直视场角（度）")
    parser.add_argument("--width", type=int, default=800, help="图像宽度")
    parser.add_argument("--height", type=int, default=600, help="图像高度")
    
    # 渲染参数
    parser.add_argument("--white_background", action="store_true", help="使用白色背景")
    parser.add_argument("--fisheye", action="store_true", help="使用鱼眼相机模型")
    
    # 环绕模式
    parser.add_argument("--orbit", action="store_true", help="生成环绕视角")
    parser.add_argument("--orbit_center", type=float, nargs=3, default=[0, 0, 0], help="环绕中心点 [x y z]")
    parser.add_argument("--orbit_radius", type=float, default=5.0, help="环绕半径")
    parser.add_argument("--orbit_views", type=int, default=36, help="环绕视角数量")
    parser.add_argument("--orbit_height", type=float, default=0.0, help="相机高度偏移")
    parser.add_argument("--orbit_dir", type=str, default="orbit_views", help="环绕视角输出目录")
    
    # KB模型参数（可选）
    parser.add_argument("--kb_k1", type=float, default=0.0, help="KB模型参数k1")
    parser.add_argument("--kb_k2", type=float, default=0.0, help="KB模型参数k2")
    parser.add_argument("--kb_k3", type=float, default=0.0, help="KB模型参数k3")
    parser.add_argument("--kb_k4", type=float, default=0.0, help="KB模型参数k4")
    
    args = parser.parse_args()
    
    # 准备KB参数
    kb_params = None
    if args.fisheye:
        kb_params = {
            'kb_k1': args.kb_k1,
            'kb_k2': args.kb_k2,
            'kb_k3': args.kb_k3,
            'kb_k4': args.kb_k4,
        }
    
    # 初始化系统状态
    safe_state(False)
    
    if args.orbit:
        # 环绕模式
        render_orbit_views(
            model_path=args.model_path,
            iteration=args.iteration,
            center_point=args.orbit_center,
            radius=args.orbit_radius,
            num_views=args.orbit_views,
            height_offset=args.orbit_height,
            fov_x=args.fov_x,
            fov_y=args.fov_y,
            width=args.width,
            height=args.height,
            output_dir=args.orbit_dir,
            white_background=args.white_background,
            fisheye=args.fisheye,
            kb_params=kb_params,
        )
    else:
        # 单个视角模式
        render_novel_view(
            model_path=args.model_path,
            iteration=args.iteration,
            position=args.position,
            look_at=args.look_at,
            up_vector=args.up,
            fov_x=args.fov_x,
            fov_y=args.fov_y,
            width=args.width,
            height=args.height,
            output_path=args.output,
            white_background=args.white_background,
            fisheye=args.fisheye,
            kb_params=kb_params,
        )
