"""
测试360度环绕渲染功能
使用GT数据集进行快速训练和渲染测试
"""

import os
import sys
import subprocess
from pathlib import Path

def check_gt_data():
    """检查GT数据是否存在"""
    gt_path = Path("../GT")
    if not gt_path.exists():
        print("❌ GT文件夹不存在")
        return False
    
    images = list(gt_path.glob("*.jpg")) + list(gt_path.glob("*.png"))
    print(f"✅ 找到 {len(images)} 张图片")
    return len(images) > 0


def prepare_colmap_data():
    """准备COLMAP数据结构"""
    print("\n=== 准备数据结构 ===")
    
    # 创建标准COLMAP目录结构
    data_dir = Path("data/GT_test")
    input_dir = data_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制图片到input目录
    gt_path = Path("../GT")
    images = sorted(gt_path.glob("*.jpg"))
    
    print(f"复制 {len(images)} 张图片到 {input_dir}")
    for img in images[:20]:  # 只用前20张进行快速测试
        import shutil
        shutil.copy(img, input_dir / img.name)
    
    print(f"✅ 数据准备完成: {data_dir}")
    return str(data_dir)


def run_colmap(data_dir):
    """运行COLMAP进行相机标定和稀疏重建"""
    print("\n=== 运行COLMAP ===")
    print("注意: 这需要安装COLMAP")
    
    # 检查是否有convert.py脚本
    if not Path("convert.py").exists():
        print("❌ convert.py不存在，无法转换数据")
        return False
    
    # 这里应该运行COLMAP，但为了测试，我们先跳过
    print("⚠️  跳过COLMAP步骤（需要手动运行）")
    print("如果要完整测试，请先运行:")
    print(f"  colmap automatic_reconstructor --workspace_path {data_dir} --image_path {data_dir}/input")
    
    return True


def quick_train(data_dir, iterations=1000):
    """快速训练模型"""
    print(f"\n=== 快速训练 ({iterations}次迭代) ===")
    
    output_dir = f"output/GT_test_{iterations}"
    
    cmd = [
        "python", "train.py",
        "-s", data_dir,
        "-m", output_dir,
        "--iterations", str(iterations),
        "--test_iterations", str(iterations),
        "--save_iterations", str(iterations),
    ]
    
    print(f"命令: {' '.join(cmd)}")
    print("开始训练...")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 训练完成")
        return output_dir
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        print(f"错误输出: {e.stderr}")
        return None
    except FileNotFoundError:
        print("❌ 找不到train.py或Python环境配置有问题")
        return None


def test_single_view(model_path):
    """测试单个视角渲染"""
    print("\n=== 测试单个视角渲染 ===")
    
    cmd = [
        "python", "render_novel_view.py",
        "--model_path", model_path,
        "--position", "3", "2", "5",
        "--look_at", "0", "0", "0",
        "--width", "800",
        "--height", "600",
        "--output", "test_single_view.png"
    ]
    
    print(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 单视角渲染成功")
        print(f"输出: test_single_view.png")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 渲染失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False


def test_orbit_render(model_path, num_views=36):
    """测试360度环绕渲染"""
    print(f"\n=== 测试360度环绕渲染 ({num_views}个视角) ===")
    
    cmd = [
        "python", "render_novel_view.py",
        "--model_path", model_path,
        "--orbit",
        "--orbit_center", "0", "0", "0",
        "--orbit_radius", "5",
        "--orbit_views", str(num_views),
        "--width", "800",
        "--height", "600",
        "--orbit_dir", "test_orbit_360"
    ]
    
    print(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ 环绕渲染成功")
        print(f"输出目录: test_orbit_360/")
        print(f"生成了 {num_views} 个视角")
        
        # 检查生成的文件
        orbit_dir = Path("test_orbit_360")
        if orbit_dir.exists():
            files = list(orbit_dir.glob("*.png"))
            print(f"实际生成文件数: {len(files)}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 渲染失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False


def test_with_existing_model():
    """使用已有模型进行测试"""
    print("\n=== 查找已有模型 ===")
    
    output_dir = Path("output")
    if not output_dir.exists():
        print("❌ output文件夹不存在")
        return None
    
    # 查找所有模型
    models = list(output_dir.glob("*/"))
    if not models:
        print("❌ 没有找到训练好的模型")
        return None
    
    print(f"找到 {len(models)} 个模型:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model.name}")
    
    # 使用第一个模型
    model_path = str(models[0])
    print(f"\n使用模型: {model_path}")
    
    return model_path


def main():
    print("="*60)
    print("360度环绕渲染功能测试")
    print("="*60)
    
    # 检查GT数据
    if not check_gt_data():
        print("\n请确保GT文件夹存在并包含图片")
        return
    
    # 选择测试模式
    print("\n请选择测试模式:")
    print("1. 使用已有模型测试（快速）")
    print("2. 完整测试（包括训练，需要较长时间）")
    print("3. 仅测试渲染脚本（使用示例参数）")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 使用已有模型
        model_path = test_with_existing_model()
        if model_path:
            test_single_view(model_path)
            test_orbit_render(model_path, num_views=12)  # 少量视角快速测试
    
    elif choice == "2":
        # 完整测试流程
        print("\n⚠️  完整测试需要:")
        print("  1. 安装COLMAP")
        print("  2. 运行相机标定")
        print("  3. 训练模型（可能需要几分钟到几小时）")
        
        confirm = input("\n确认继续? (y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消")
            return
        
        # 准备数据
        data_dir = prepare_colmap_data()
        
        # 运行COLMAP（需要手动）
        run_colmap(data_dir)
        
        print("\n请先手动运行COLMAP进行相机标定，然后:")
        print(f"  python train.py -s {data_dir} -m output/GT_test")
        print("\n训练完成后，再次运行此脚本选择模式1进行测试")
    
    elif choice == "3":
        # 仅测试脚本
        print("\n=== 测试渲染脚本（不需要模型）===")
        print("\n这将测试脚本的参数解析和基本功能")
        print("注意: 由于没有实际模型，渲染会失败，但可以验证脚本逻辑")
        
        # 测试命令行参数
        print("\n测试1: 显示帮助信息")
        os.system("python render_novel_view.py --help")
        
        print("\n✅ 脚本测试完成")
        print("\n要进行实际渲染测试，请:")
        print("  1. 先训练一个模型")
        print("  2. 然后选择模式1进行测试")
    
    else:
        print("无效的选择")
        return
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
