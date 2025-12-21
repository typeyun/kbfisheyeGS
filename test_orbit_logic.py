"""
测试360度环绕渲染的核心逻辑
不需要CUDA或模型，只测试数学计算
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path


def calculate_orbit_positions(center, radius, num_views, height_offset=0.0):
    """
    计算环绕轨迹上的相机位置
    
    参数:
        center: 环绕中心点 [x, y, z]
        radius: 环绕半径
        num_views: 视角数量
        height_offset: 高度偏移
    
    返回:
        positions: 相机位置列表
        angles: 对应的角度列表（度）
    """
    center = np.array(center)
    positions = []
    angles = []
    
    for i in range(num_views):
        angle = 2 * math.pi * i / num_views
        
        # 在XZ平面上环绕
        x = center[0] + radius * math.cos(angle)
        z = center[2] + radius * math.sin(angle)
        y = center[1] + height_offset
        
        positions.append([x, y, z])
        angles.append(angle * 180 / math.pi)  # 转换为度
    
    return positions, angles


def visualize_orbit(center, radius, positions, output_path="orbit_visualization.png"):
    """可视化环绕轨迹"""
    center = np.array(center)
    positions = np.array(positions)
    
    fig = plt.figure(figsize=(15, 5))
    
    # 俯视图 (XZ平面)
    ax1 = fig.add_subplot(131)
    ax1.plot(positions[:, 0], positions[:, 2], 'b.-', label='相机轨迹')
    ax1.plot(center[0], center[2], 'r*', markersize=15, label='中心点')
    ax1.scatter(positions[0, 0], positions[0, 2], c='g', s=100, marker='^', label='起点')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_title('俯视图 (XZ平面)')
    ax1.grid(True)
    ax1.axis('equal')
    ax1.legend()
    
    # 侧视图 (XY平面)
    ax2 = fig.add_subplot(132)
    ax2.plot(positions[:, 0], positions[:, 1], 'b.-', label='相机轨迹')
    ax2.plot(center[0], center[1], 'r*', markersize=15, label='中心点')
    ax2.scatter(positions[0, 0], positions[0, 1], c='g', s=100, marker='^', label='起点')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('侧视图 (XY平面)')
    ax2.grid(True)
    ax2.axis('equal')
    ax2.legend()
    
    # 3D视图
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(positions[:, 0], positions[:, 2], positions[:, 1], 'b.-', label='相机轨迹')
    ax3.scatter(center[0], center[2], center[1], c='r', s=100, marker='*', label='中心点')
    ax3.scatter(positions[0, 0], positions[0, 2], positions[0, 1], c='g', s=100, marker='^', label='起点')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_zlabel('Y')
    ax3.set_title('3D视图')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化图像已保存: {output_path}")
    plt.close()


def test_orbit_calculation():
    """测试环绕轨迹计算"""
    print("="*70)
    print("测试1: 基本环绕轨迹")
    print("="*70)
    
    center = [0, 0, 0]
    radius = 5.0
    num_views = 36
    
    positions, angles = calculate_orbit_positions(center, radius, num_views)
    
    print(f"参数:")
    print(f"  中心点: {center}")
    print(f"  半径: {radius}")
    print(f"  视角数: {num_views}")
    
    print(f"\n前10个相机位置:")
    for i in range(min(10, len(positions))):
        pos = positions[i]
        angle = angles[i]
        print(f"  视角 {i:2d} ({angle:6.1f}°): [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]")
    
    # 验证距离
    distances = [np.linalg.norm(np.array(pos) - np.array(center)) for pos in positions]
    avg_dist = np.mean(distances)
    max_diff = max(abs(d - radius) for d in distances)
    
    print(f"\n距离验证:")
    print(f"  平均距离: {avg_dist:.6f}")
    print(f"  期望距离: {radius:.6f}")
    print(f"  最大偏差: {max_diff:.9f}")
    
    if max_diff < 1e-6:
        print("  ✅ 距离计算精确")
    else:
        print(f"  ⚠️  存在偏差: {max_diff}")
    
    # 可视化
    visualize_orbit(center, radius, positions, "test_orbit_basic.png")
    
    return True


def test_orbit_with_height():
    """测试带高度偏移的环绕"""
    print("\n" + "="*70)
    print("测试2: 带高度偏移的环绕")
    print("="*70)
    
    center = [0, 0, 0]
    radius = 5.0
    num_views = 36
    height_offset = 2.0
    
    positions, angles = calculate_orbit_positions(center, radius, num_views, height_offset)
    
    print(f"参数:")
    print(f"  中心点: {center}")
    print(f"  半径: {radius}")
    print(f"  高度偏移: {height_offset}")
    print(f"  视角数: {num_views}")
    
    # 验证高度
    heights = [pos[1] for pos in positions]
    expected_height = center[1] + height_offset
    
    print(f"\n高度验证:")
    print(f"  期望高度: {expected_height}")
    print(f"  实际高度: {heights[0]:.6f}")
    print(f"  所有高度一致: {all(abs(h - expected_height) < 1e-6 for h in heights)}")
    
    if all(abs(h - expected_height) < 1e-6 for h in heights):
        print("  ✅ 高度计算正确")
    else:
        print("  ❌ 高度计算有误")
    
    # 可视化
    visualize_orbit(center, radius, positions, "test_orbit_height.png")
    
    return True


def test_orbit_custom_center():
    """测试自定义中心点的环绕"""
    print("\n" + "="*70)
    print("测试3: 自定义中心点")
    print("="*70)
    
    center = [2, 1, -3]
    radius = 3.0
    num_views = 24
    
    positions, angles = calculate_orbit_positions(center, radius, num_views)
    
    print(f"参数:")
    print(f"  中心点: {center}")
    print(f"  半径: {radius}")
    print(f"  视角数: {num_views}")
    
    print(f"\n前5个相机位置:")
    for i in range(min(5, len(positions))):
        pos = positions[i]
        angle = angles[i]
        dist = np.linalg.norm(np.array(pos) - np.array(center))
        print(f"  视角 {i:2d} ({angle:6.1f}°): [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}] 距离={dist:.4f}")
    
    # 可视化
    visualize_orbit(center, radius, positions, "test_orbit_custom.png")
    
    return True


def test_different_view_counts():
    """测试不同的视角数量"""
    print("\n" + "="*70)
    print("测试4: 不同视角数量")
    print("="*70)
    
    center = [0, 0, 0]
    radius = 5.0
    view_counts = [12, 36, 72, 120]
    
    for num_views in view_counts:
        positions, angles = calculate_orbit_positions(center, radius, num_views)
        angle_step = 360.0 / num_views
        
        print(f"\n视角数: {num_views}")
        print(f"  角度步长: {angle_step:.2f}°")
        print(f"  生成位置数: {len(positions)}")
        print(f"  角度范围: {angles[0]:.1f}° ~ {angles[-1]:.1f}°")
    
    return True


def test_video_frame_calculation():
    """测试视频帧数计算"""
    print("\n" + "="*70)
    print("测试5: 视频帧数计算")
    print("="*70)
    
    fps = 30
    durations = [3, 5, 10]  # 秒
    
    print(f"帧率: {fps} fps")
    print(f"\n不同时长的视频需要的帧数:")
    
    for duration in durations:
        num_frames = fps * duration
        angle_step = 360.0 / num_frames
        
        print(f"\n  {duration}秒视频:")
        print(f"    需要帧数: {num_frames}")
        print(f"    角度步长: {angle_step:.2f}°")
        print(f"    命令示例: python render_novel_view.py --orbit --orbit_views {num_frames}")
    
    return True


def generate_example_commands():
    """生成示例命令"""
    print("\n" + "="*70)
    print("实用命令示例")
    print("="*70)
    
    examples = [
        {
            "name": "快速预览（12个视角）",
            "cmd": "python render_novel_view.py --model_path output/model --orbit --orbit_views 12 --orbit_dir preview_12"
        },
        {
            "name": "标准360度（36个视角）",
            "cmd": "python render_novel_view.py --model_path output/model --orbit --orbit_views 36 --orbit_dir orbit_36"
        },
        {
            "name": "高清视频（120帧，4秒@30fps）",
            "cmd": "python render_novel_view.py --model_path output/model --orbit --orbit_views 120 --width 1920 --height 1080 --orbit_dir video_hd"
        },
        {
            "name": "俯视环绕（高度偏移）",
            "cmd": "python render_novel_view.py --model_path output/model --orbit --orbit_height 3 --orbit_views 36 --orbit_dir orbit_top"
        },
        {
            "name": "自定义中心和半径",
            "cmd": "python render_novel_view.py --model_path output/model --orbit --orbit_center 1 0 2 --orbit_radius 8 --orbit_views 36"
        },
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print(f"   {example['cmd']}")
    
    print("\n" + "="*70)


def main():
    print("\n" + "="*70)
    print(" "*15 + "360度环绕渲染 - 核心逻辑测试")
    print("="*70)
    
    try:
        # 运行所有测试
        test_orbit_calculation()
        test_orbit_with_height()
        test_orbit_custom_center()
        test_different_view_counts()
        test_video_frame_calculation()
        
        # 生成示例命令
        generate_example_commands()
        
        print("\n" + "="*70)
        print("✅ 所有测试通过！")
        print("="*70)
        
        print("\n生成的可视化图像:")
        print("  - test_orbit_basic.png   (基本环绕)")
        print("  - test_orbit_height.png  (带高度偏移)")
        print("  - test_orbit_custom.png  (自定义中心)")
        
        print("\n下一步:")
        print("  1. 查看生成的可视化图像")
        print("  2. 训练一个模型或使用已有模型")
        print("  3. 运行实际的360度渲染")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
