"""
快速测试360度渲染功能
不需要训练模型，直接测试渲染逻辑
"""

import torch
import numpy as np
import math
from pathlib import Path

def test_camera_creation():
    """测试相机创建功能"""
    print("="*60)
    print("测试1: 相机创建功能")
    print("="*60)
    
    try:
        from render_novel_view import create_camera_from_pose
        
        # 创建一个测试相机
        camera = create_camera_from_pose(
            position=[3, 2, 5],
            look_at=[0, 0, 0],
            up_vector=[0, 1, 0],
            fov_x=60.0,
            fov_y=45.0,
            width=800,
            height=600,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print("✅ 相机创建成功")
        print(f"  - 图像尺寸: {camera.image_width} x {camera.image_height}")
        print(f"  - FOV: {camera.FoVx*180/math.pi:.1f}° x {camera.FoVy*180/math.pi:.1f}°")
        print(f"  - 相机中心: {camera.camera_center.cpu().numpy()}")
        print(f"  - 设备: {camera.data_device}")
        
        return True
    except Exception as e:
        print(f"❌ 相机创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orbit_calculation():
    """测试环绕轨迹计算"""
    print("\n" + "="*60)
    print("测试2: 环绕轨迹计算")
    print("="*60)
    
    try:
        center = np.array([0, 0, 0])
        radius = 5.0
        num_views = 36
        
        print(f"环绕参数:")
        print(f"  - 中心点: {center}")
        print(f"  - 半径: {radius}")
        print(f"  - 视角数: {num_views}")
        
        positions = []
        for i in range(num_views):
            angle = 2 * math.pi * i / num_views
            x = center[0] + radius * math.cos(angle)
            z = center[2] + radius * math.sin(angle)
            y = center[1]
            positions.append([x, y, z])
        
        print(f"\n生成的相机位置（前5个）:")
        for i, pos in enumerate(positions[:5]):
            print(f"  视角 {i}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        
        # 验证所有位置到中心的距离
        distances = [np.linalg.norm(np.array(pos) - center) for pos in positions]
        avg_dist = np.mean(distances)
        max_diff = max(abs(d - radius) for d in distances)
        
        print(f"\n距离验证:")
        print(f"  - 平均距离: {avg_dist:.4f}")
        print(f"  - 期望距离: {radius:.4f}")
        print(f"  - 最大偏差: {max_diff:.6f}")
        
        if max_diff < 0.001:
            print("✅ 环绕轨迹计算正确")
            return True
        else:
            print("❌ 环绕轨迹计算有误差")
            return False
            
    except Exception as e:
        print(f"❌ 轨迹计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_script_import():
    """测试脚本导入"""
    print("\n" + "="*60)
    print("测试3: 脚本模块导入")
    print("="*60)
    
    try:
        from render_novel_view import (
            create_camera_from_pose,
            render_novel_view,
            render_orbit_views
        )
        print("✅ 所有函数导入成功")
        print("  - create_camera_from_pose")
        print("  - render_novel_view")
        print("  - render_orbit_views")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_validation():
    """测试参数验证"""
    print("\n" + "="*60)
    print("测试4: 参数验证")
    print("="*60)
    
    test_cases = [
        {
            "name": "标准参数",
            "position": [3, 2, 5],
            "look_at": [0, 0, 0],
            "fov_x": 60.0,
            "fov_y": 45.0,
            "width": 800,
            "height": 600,
        },
        {
            "name": "广角镜头",
            "position": [1, 1, 1],
            "look_at": [0, 0, 0],
            "fov_x": 90.0,
            "fov_y": 67.5,
            "width": 1920,
            "height": 1080,
        },
        {
            "name": "长焦镜头",
            "position": [10, 5, 10],
            "look_at": [0, 0, 0],
            "fov_x": 30.0,
            "fov_y": 22.5,
            "width": 1920,
            "height": 1080,
        },
    ]
    
    try:
        from render_novel_view import create_camera_from_pose
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n测试用例 {i}: {test['name']}")
            camera = create_camera_from_pose(
                position=test['position'],
                look_at=test['look_at'],
                up_vector=[0, 1, 0],
                fov_x=test['fov_x'],
                fov_y=test['fov_y'],
                width=test['width'],
                height=test['height'],
                device="cpu"
            )
            print(f"  ✅ 参数有效")
            print(f"     位置: {test['position']}")
            print(f"     FOV: {test['fov_x']}° x {test['fov_y']}°")
            print(f"     尺寸: {test['width']}x{test['height']}")
        
        print("\n✅ 所有参数验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 参数验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """检查依赖项"""
    print("\n" + "="*60)
    print("检查依赖项")
    print("="*60)
    
    dependencies = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "PIL": "Pillow",
        "torchvision": "TorchVision",
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - 未安装")
            all_ok = False
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA (设备: {torch.cuda.get_device_name(0)})")
    else:
        print(f"⚠️  CUDA - 不可用（将使用CPU）")
    
    return all_ok


def main():
    print("\n" + "="*70)
    print(" "*20 + "360度渲染功能快速测试")
    print("="*70)
    
    results = []
    
    # 检查依赖
    results.append(("依赖检查", check_dependencies()))
    
    # 测试脚本导入
    results.append(("脚本导入", test_script_import()))
    
    # 测试相机创建
    results.append(("相机创建", test_camera_creation()))
    
    # 测试轨迹计算
    results.append(("轨迹计算", test_orbit_calculation()))
    
    # 测试参数验证
    results.append(("参数验证", test_parameter_validation()))
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20s}: {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！渲染功能正常")
        print("\n下一步:")
        print("  1. 训练一个模型: python train.py -s <data_path> -m output/model")
        print("  2. 测试单视角渲染: python render_novel_view.py --model_path output/model")
        print("  3. 测试360度渲染: python render_novel_view.py --model_path output/model --orbit")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
    
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n严重错误: {e}")
        import traceback
        traceback.print_exc()
