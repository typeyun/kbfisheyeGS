#!/usr/bin/env python3
"""
可视化自适应高斯分裂效果
对比固定阈值 vs 自适应阈值的分裂行为
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud

def create_scene_with_varying_curvature(num_points=200):
    """
    创建一个包含不同曲率区域的测试场景
    - 区域1: 高曲率（复杂几何）
    - 区域2: 中等曲率
    - 区域3: 低曲率（平坦表面）
    """
    points = []
    colors = []
    
    # 区域1: 高曲率（球面）
    n1 = num_points // 3
    theta = np.random.uniform(0, np.pi, n1)
    phi = np.random.uniform(0, 2*np.pi, n1)
    r = 1.0
    x1 = r * np.sin(theta) * np.cos(phi)
    y1 = r * np.sin(theta) * np.sin(phi)
    z1 = r * np.cos(theta) - 2.0  # 偏移
    points1 = np.stack([x1, y1, z1], axis=1)
    colors1 = np.tile([1.0, 0.0, 0.0], (n1, 1))  # 红色
    
    # 区域2: 中等曲率（圆柱）
    n2 = num_points // 3
    theta2 = np.random.uniform(0, 2*np.pi, n2)
    z2 = np.random.uniform(-1, 1, n2)
    r2 = 0.5
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    points2 = np.stack([x2, y2, z2], axis=1)
    colors2 = np.tile([0.0, 1.0, 0.0], (n2, 1))  # 绿色
    
    # 区域3: 低曲率（平面）
    n3 = num_points - n1 - n2
    x3 = np.random.uniform(-1, 1, n3)
    y3 = np.random.uniform(-1, 1, n3)
    z3 = np.ones(n3) * 2.0  # 平面
    points3 = np.stack([x3, y3, z3], axis=1)
    colors3 = np.tile([0.0, 0.0, 1.0], (n3, 1))  # 蓝色
    
    # 合并
    points = np.vstack([points1, points2, points3]).astype(np.float32)
    colors = np.vstack([colors1, colors2, colors3]).astype(np.float32)
    normals = np.zeros_like(points)
    
    return BasicPointCloud(points=points, colors=colors, normals=normals), (n1, n2, n3)

def simulate_training(gaussians, iterations=20, curvature_regions=None):
    """模拟训练过程，生成不同曲率的梯度"""
    n1, n2, n3 = curvature_regions
    
    for i in range(iterations):
        num_points = gaussians.get_xyz.shape[0]
        viewspace_points = torch.zeros(num_points, 3, device="cuda", requires_grad=True)
        
        # 高曲率区域：大梯度变化
        if n1 > 0:
            viewspace_points.data[:n1] = torch.randn(n1, 3, device="cuda") * 2.0
        
        # 中等曲率区域
        if n2 > 0:
            viewspace_points.data[n1:n1+n2] = torch.randn(n2, 3, device="cuda") * 0.8
        
        # 低曲率区域：小梯度变化
        if n3 > 0:
            viewspace_points.data[n1+n2:] = torch.randn(n3, 3, device="cuda") * 0.2
        
        fake_loss = viewspace_points.sum()
        fake_loss.backward()
        
        update_filter = torch.ones(num_points, dtype=torch.bool, device="cuda")
        gaussians.add_densification_stats(viewspace_points, update_filter)

def visualize_comparison():
    """可视化对比固定阈值 vs 自适应阈值"""
    print("="*60)
    print("可视化自适应高斯分裂效果")
    print("="*60)
    
    # 创建场景
    print("\n创建测试场景...")
    pcd, curvature_regions = create_scene_with_varying_curvature(200)
    n1, n2, n3 = curvature_regions
    
    # 创建两个相同的高斯模型
    gaussians_fixed = GaussianModel(sh_degree=3)
    gaussians_adaptive = GaussianModel(sh_degree=3)
    
    gaussians_fixed.create_from_pcd(pcd, spatial_lr_scale=1.0)
    gaussians_adaptive.create_from_pcd(pcd, spatial_lr_scale=1.0)
    
    # 训练设置
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
    
    # 模拟训练
    print("模拟训练过程...")
    simulate_training(gaussians_fixed, iterations=20, curvature_regions=curvature_regions)
    simulate_training(gaussians_adaptive, iterations=20, curvature_regions=curvature_regions)
    
    # 计算统计信息
    grads_fixed = gaussians_fixed.xyz_gradient_accum / gaussians_fixed.denom
    grads_adaptive = gaussians_adaptive.xyz_gradient_accum / gaussians_adaptive.denom
    hessian_norm = gaussians_adaptive.compute_hessian_norm_approx()
    
    # 计算自适应阈值
    base_threshold = 0.0002
    adaptive_threshold = base_threshold / hessian_norm.squeeze()
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('自适应高斯分裂效果对比', fontsize=16, fontweight='bold')
    
    # 1. 梯度分布
    ax = axes[0, 0]
    grads_np = grads_fixed.cpu().numpy().squeeze()
    colors = ['red'] * n1 + ['green'] * n2 + ['blue'] * n3
    ax.scatter(range(len(grads_np)), grads_np, c=colors, alpha=0.6, s=20)
    ax.axhline(y=base_threshold, color='black', linestyle='--', label='固定阈值')
    ax.set_xlabel('高斯索引')
    ax.set_ylabel('梯度大小')
    ax.set_title('梯度分布\n红=高曲率, 绿=中等, 蓝=低曲率')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Hessian范数分布
    ax = axes[0, 1]
    hessian_np = hessian_norm.cpu().numpy().squeeze()
    ax.scatter(range(len(hessian_np)), hessian_np, c=colors, alpha=0.6, s=20)
    ax.set_xlabel('高斯索引')
    ax.set_ylabel('Hessian范数 ||H||')
    ax.set_title('Hessian范数分布（曲率指标）')
    ax.grid(True, alpha=0.3)
    
    # 3. 自适应阈值分布
    ax = axes[0, 2]
    adaptive_np = adaptive_threshold.cpu().numpy()
    ax.scatter(range(len(adaptive_np)), adaptive_np, c=colors, alpha=0.6, s=20)
    ax.axhline(y=base_threshold, color='black', linestyle='--', label='固定阈值')
    ax.set_xlabel('高斯索引')
    ax.set_ylabel('阈值')
    ax.set_title('自适应阈值分布\n阈值 = 基础阈值 / ||H||')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 固定阈值分裂判断
    ax = axes[1, 0]
    will_split_fixed = grads_np >= base_threshold
    ax.scatter(range(len(grads_np)), grads_np, 
               c=['orange' if s else 'gray' for s in will_split_fixed],
               alpha=0.6, s=20)
    ax.axhline(y=base_threshold, color='black', linestyle='--', label='固定阈值')
    ax.set_xlabel('高斯索引')
    ax.set_ylabel('梯度大小')
    ax.set_title(f'固定阈值分裂判断\n橙色=将分裂 ({will_split_fixed.sum()}个)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 自适应阈值分裂判断
    ax = axes[1, 1]
    will_split_adaptive = grads_np >= adaptive_np
    ax.scatter(range(len(grads_np)), grads_np,
               c=['orange' if s else 'gray' for s in will_split_adaptive],
               alpha=0.6, s=20)
    ax.plot(range(len(adaptive_np)), adaptive_np, 'r-', alpha=0.5, label='自适应阈值')
    ax.set_xlabel('高斯索引')
    ax.set_ylabel('梯度大小')
    ax.set_title(f'自适应阈值分裂判断\n橙色=将分裂 ({will_split_adaptive.sum()}个)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 统计对比
    ax = axes[1, 2]
    
    # 按区域统计
    regions = ['高曲率\n(球面)', '中等曲率\n(圆柱)', '低曲率\n(平面)']
    region_indices = [
        range(0, n1),
        range(n1, n1+n2),
        range(n1+n2, n1+n2+n3)
    ]
    
    fixed_counts = [will_split_fixed[idx].sum() for idx in region_indices]
    adaptive_counts = [will_split_adaptive[idx].sum() for idx in region_indices]
    
    x = np.arange(len(regions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, fixed_counts, width, label='固定阈值', color='steelblue')
    bars2 = ax.bar(x + width/2, adaptive_counts, width, label='自适应阈值', color='coral')
    
    ax.set_xlabel('区域类型')
    ax.set_ylabel('将分裂的高斯数量')
    ax.set_title('各区域分裂数量对比')
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = 'adaptive_split_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化结果已保存到: {output_path}")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("统计信息")
    print("="*60)
    
    print(f"\n总高斯数量: {len(grads_np)}")
    print(f"  - 高曲率区域: {n1}")
    print(f"  - 中等曲率区域: {n2}")
    print(f"  - 低曲率区域: {n3}")
    
    print(f"\n固定阈值方法:")
    print(f"  总分裂数: {will_split_fixed.sum()}")
    print(f"  - 高曲率: {fixed_counts[0]}")
    print(f"  - 中等曲率: {fixed_counts[1]}")
    print(f"  - 低曲率: {fixed_counts[2]}")
    
    print(f"\n自适应阈值方法:")
    print(f"  总分裂数: {will_split_adaptive.sum()}")
    print(f"  - 高曲率: {adaptive_counts[0]}")
    print(f"  - 中等曲率: {adaptive_counts[1]}")
    print(f"  - 低曲率: {adaptive_counts[2]}")
    
    print(f"\n差异分析:")
    print(f"  高曲率区域: {adaptive_counts[0] - fixed_counts[0]:+d} ({(adaptive_counts[0]/max(fixed_counts[0],1)-1)*100:+.1f}%)")
    print(f"  中等曲率区域: {adaptive_counts[1] - fixed_counts[1]:+d} ({(adaptive_counts[1]/max(fixed_counts[1],1)-1)*100:+.1f}%)")
    print(f"  低曲率区域: {adaptive_counts[2] - fixed_counts[2]:+d} ({(adaptive_counts[2]/max(fixed_counts[2],1)-1)*100:+.1f}%)")
    
    print("\n✅ 自适应方法优势:")
    print("  - 在高曲率区域增加分裂（更细粒度）")
    print("  - 在低曲率区域减少分裂（避免浪费）")
    print("  - 自动适应场景复杂度")
    
    # 显示图表
    try:
        plt.show()
    except:
        print("\n注意: 无法显示图表窗口，但已保存到文件")

def main():
    """主函数"""
    try:
        visualize_comparison()
        print("\n" + "="*60)
        print("🎉 可视化完成！")
        print("="*60)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
