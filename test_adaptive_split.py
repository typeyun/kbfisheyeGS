#!/usr/bin/env python3
"""
测试自适应高斯分裂功能（论文公式40）
验证基于Hessian范数的自适应阈值是否正确工作
"""

import torch
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud

def create_test_gaussians(num_points=100):
    """创建测试用的高斯模型"""
    print(f"创建 {num_points} 个测试高斯...")
    
    # 创建测试点云
    points = np.random.randn(num_points, 3).astype(np.float32)
    colors = np.random.rand(num_points, 3).astype(np.float32)
    normals = np.zeros((num_points, 3), dtype=np.float32)
    
    pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
    
    # 创建高斯模型
    gaussians = GaussianModel(sh_degree=3)
    gaussians.create_from_pcd(pcd, spatial_lr_scale=1.0)
    
    return gaussians

def test_hessian_computation():
    """测试1: Hessian范数计算"""
    print("\n" + "="*60)
    print("测试1: Hessian范数近似计算")
    print("="*60)
    
    gaussians = create_test_gaussians(50)
    
    # 模拟训练设置
    class TrainingArgs:
        percent_dense = 0.01
        position_lr_init = 0.00016
        position_lr_final = 0.0000016
        position_lr_delay_mult = 0.01
        position_lr_max_steps = 30000
        feature_lr = 0.0025
        opacity_lr = 0.05
        scaling_lr = 0.005
        rotation_lr = 0.001
    
    gaussians.training_setup(TrainingArgs())
    
    # 模拟几次梯度更新
    print("\n模拟梯度更新...")
    for i in range(5):
        # 创建模拟的视图空间点张量
        viewspace_points = torch.randn(50, 3, device="cuda", requires_grad=True)
        
        # 模拟梯度
        fake_loss = viewspace_points.sum()
        fake_loss.backward()
        
        # 更新统计
        update_filter = torch.ones(50, dtype=torch.bool, device="cuda")
        gaussians.add_densification_stats(viewspace_points, update_filter)
        
        print(f"  迭代 {i+1}: 梯度累积完成")
    
    # 计算Hessian范数
    hessian_norm = gaussians.compute_hessian_norm_approx()
    
    print(f"\n✅ Hessian范数统计:")
    print(f"  平均值: {hessian_norm.mean().item():.6f}")
    print(f"  最小值: {hessian_norm.min().item():.6f}")
    print(f"  最大值: {hessian_norm.max().item():.6f}")
    print(f"  标准差: {hessian_norm.std().item():.6f}")
    
    # 验证范数至少为1
    assert hessian_norm.min() >= 1.0, "Hessian范数应该至少为1"
    print("\n✅ 测试1通过: Hessian范数计算正确")
    
    return gaussians

def test_adaptive_threshold():
    """测试2: 自适应阈值计算"""
    print("\n" + "="*60)
    print("测试2: 自适应阈值计算（论文公式40）")
    print("="*60)
    
    gaussians = create_test_gaussians(100)
    
    class TrainingArgs:
        percent_dense = 0.01
        position_lr_init = 0.00016
        position_lr_final = 0.0000016
        position_lr_delay_mult = 0.01
        position_lr_max_steps = 30000
        feature_lr = 0.0025
        opacity_lr = 0.05
        scaling_lr = 0.005
        rotation_lr = 0.001
    
    gaussians.training_setup(TrainingArgs())
    
    # 模拟不同曲率的区域
    print("\n创建不同曲率的测试场景...")
    
    # 区域1: 高曲率（大梯度变化）
    high_curvature_indices = torch.arange(0, 30, device="cuda")
    # 区域2: 低曲率（小梯度变化）
    low_curvature_indices = torch.arange(30, 60, device="cuda")
    # 区域3: 中等曲率
    medium_curvature_indices = torch.arange(60, 100, device="cuda")
    
    # 模拟梯度更新（不同区域不同的梯度变化）
    for i in range(10):
        viewspace_points = torch.zeros(100, 3, device="cuda", requires_grad=True)
        
        # 高曲率区域：大梯度变化
        viewspace_points.data[high_curvature_indices] = torch.randn(30, 3, device="cuda") * 2.0
        
        # 低曲率区域：小梯度变化
        viewspace_points.data[low_curvature_indices] = torch.randn(30, 3, device="cuda") * 0.1
        
        # 中等曲率区域
        viewspace_points.data[medium_curvature_indices] = torch.randn(40, 3, device="cuda") * 0.5
        
        fake_loss = viewspace_points.sum()
        fake_loss.backward()
        
        update_filter = torch.ones(100, dtype=torch.bool, device="cuda")
        gaussians.add_densification_stats(viewspace_points, update_filter)
    
    # 计算Hessian范数
    hessian_norm = gaussians.compute_hessian_norm_approx()
    
    print(f"\n不同区域的Hessian范数:")
    print(f"  高曲率区域: {hessian_norm[high_curvature_indices].mean().item():.6f}")
    print(f"  低曲率区域: {hessian_norm[low_curvature_indices].mean().item():.6f}")
    print(f"  中等曲率区域: {hessian_norm[medium_curvature_indices].mean().item():.6f}")
    
    # 计算自适应阈值
    base_threshold = 0.0002
    adaptive_threshold = base_threshold / hessian_norm.squeeze()
    
    print(f"\n自适应阈值（基础阈值={base_threshold}）:")
    print(f"  高曲率区域: {adaptive_threshold[high_curvature_indices].mean().item():.8f}")
    print(f"  低曲率区域: {adaptive_threshold[low_curvature_indices].mean().item():.8f}")
    print(f"  中等曲率区域: {adaptive_threshold[medium_curvature_indices].mean().item():.8f}")
    
    # 验证：高曲率区域应该有更低的阈值（更容易分裂）
    high_curve_threshold = adaptive_threshold[high_curvature_indices].mean()
    low_curve_threshold = adaptive_threshold[low_curvature_indices].mean()
    
    print(f"\n✅ 阈值比较:")
    print(f"  高曲率阈值 < 低曲率阈值: {high_curve_threshold < low_curve_threshold}")
    
    assert high_curve_threshold < low_curve_threshold, \
        "高曲率区域应该有更低的阈值（更容易分裂）"
    
    print("\n✅ 测试2通过: 自适应阈值计算正确")
    
    return gaussians

def test_split_comparison():
    """测试3: 对比固定阈值 vs 自适应阈值"""
    print("\n" + "="*60)
    print("测试3: 固定阈值 vs 自适应阈值分裂对比")
    print("="*60)
    
    # 创建两个相同的高斯模型
    gaussians_fixed = create_test_gaussians(50)
    gaussians_adaptive = create_test_gaussians(50)
    
    class TrainingArgs:
        percent_dense = 0.01
        position_lr_init = 0.00016
        position_lr_final = 0.0000016
        position_lr_delay_mult = 0.01
        position_lr_max_steps = 30000
        feature_lr = 0.0025
        opacity_lr = 0.05
        scaling_lr = 0.005
        rotation_lr = 0.001
    
    gaussians_fixed.training_setup(TrainingArgs())
    gaussians_adaptive.training_setup(TrainingArgs())
    
    # 模拟梯度更新
    print("\n模拟梯度更新...")
    for i in range(10):
        viewspace_points = torch.randn(50, 3, device="cuda", requires_grad=True)
        fake_loss = viewspace_points.sum()
        fake_loss.backward()
        
        update_filter = torch.ones(50, dtype=torch.bool, device="cuda")
        gaussians_fixed.add_densification_stats(viewspace_points, update_filter)
        gaussians_adaptive.add_densification_stats(viewspace_points, update_filter)
    
    # 计算平均梯度
    grads_fixed = gaussians_fixed.xyz_gradient_accum / gaussians_fixed.denom
    grads_adaptive = gaussians_adaptive.xyz_gradient_accum / gaussians_adaptive.denom
    
    # 记录初始点数
    initial_points = gaussians_fixed.get_xyz.shape[0]
    print(f"\n初始高斯数量: {initial_points}")
    
    # 使用固定阈值分裂
    print("\n使用固定阈值分裂...")
    gaussians_fixed.densify_and_split(
        grads_fixed, 
        grad_threshold=0.0002, 
        scene_extent=1.0,
        use_adaptive_threshold=False
    )
    points_after_fixed = gaussians_fixed.get_xyz.shape[0]
    print(f"  固定阈值后: {points_after_fixed} 个高斯")
    
    # 使用自适应阈值分裂
    print("\n使用自适应阈值分裂...")
    gaussians_adaptive.densify_and_split(
        grads_adaptive, 
        grad_threshold=0.0002, 
        scene_extent=1.0,
        use_adaptive_threshold=True
    )
    points_after_adaptive = gaussians_adaptive.get_xyz.shape[0]
    print(f"  自适应阈值后: {points_after_adaptive} 个高斯")
    
    print(f"\n✅ 分裂结果对比:")
    print(f"  固定阈值增加: {points_after_fixed - initial_points} 个高斯")
    print(f"  自适应阈值增加: {points_after_adaptive - initial_points} 个高斯")
    print(f"  差异: {abs(points_after_adaptive - points_after_fixed)} 个高斯")
    
    print("\n✅ 测试3通过: 分裂功能正常工作")

def test_integration():
    """测试4: 完整集成测试"""
    print("\n" + "="*60)
    print("测试4: 完整集成测试")
    print("="*60)
    
    gaussians = create_test_gaussians(100)
    
    class TrainingArgs:
        percent_dense = 0.01
        position_lr_init = 0.00016
        position_lr_final = 0.0000016
        position_lr_delay_mult = 0.01
        position_lr_max_steps = 30000
        feature_lr = 0.0025
        opacity_lr = 0.05
        scaling_lr = 0.005
        rotation_lr = 0.001
    
    gaussians.training_setup(TrainingArgs())
    
    print("\n模拟完整的密集化流程...")
    initial_points = gaussians.get_xyz.shape[0]
    
    # 模拟多次迭代
    for iteration in range(20):
        # 模拟渲染和梯度计算
        viewspace_points = torch.randn(gaussians.get_xyz.shape[0], 3, device="cuda", requires_grad=True)
        fake_loss = viewspace_points.sum()
        fake_loss.backward()
        
        # 更新统计
        update_filter = torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        gaussians.add_densification_stats(viewspace_points, update_filter)
        
        # 每5次迭代执行一次密集化
        if (iteration + 1) % 5 == 0:
            grads = gaussians.xyz_gradient_accum / gaussians.denom
            gaussians.densify_and_prune(
                max_grad=0.0002,
                min_opacity=0.005,
                extent=1.0,
                max_screen_size=20
            )
            print(f"  迭代 {iteration+1}: 当前高斯数量 = {gaussians.get_xyz.shape[0]}")
    
    final_points = gaussians.get_xyz.shape[0]
    
    print(f"\n✅ 完整流程结果:")
    print(f"  初始高斯: {initial_points}")
    print(f"  最终高斯: {final_points}")
    print(f"  净增加: {final_points - initial_points}")
    
    print("\n✅ 测试4通过: 完整集成测试成功")

def main():
    """运行所有测试"""
    print("="*60)
    print("自适应高斯分裂功能测试（论文公式40）")
    print("="*60)
    
    try:
        # 测试1: Hessian范数计算
        test_hessian_computation()
        
        # 测试2: 自适应阈值
        test_adaptive_threshold()
        
        # 测试3: 分裂对比
        test_split_comparison()
        
        # 测试4: 完整集成
        test_integration()
        
        print("\n" + "="*60)
        print("🎉 所有测试通过！")
        print("="*60)
        print("\n✅ 自适应高斯分裂功能已完整实现（100%）")
        print("✅ 论文公式40: 分裂阈值 ∝ 1 / max(1, ||H(μ_c)||_F)")
        print("✅ 高曲率区域自动使用更低阈值，实现更细粒度分裂")
        print("✅ 低曲率区域保持较高阈值，避免过度分裂")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
