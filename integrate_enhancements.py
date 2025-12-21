#!/usr/bin/env python3
"""
快速集成脚本：二阶修正和自适应雅可比

使用方法:
    python integrate_enhancements.py --check     # 检查环境
    python integrate_enhancements.py --backup    # 备份原文件
    python integrate_enhancements.py --integrate # 集成新功能
    python integrate_enhancements.py --compile   # 重新编译
    python integrate_enhancements.py --test      # 运行测试
    python integrate_enhancements.py --all       # 执行所有步骤
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

class EnhancementIntegrator:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.cuda_dir = self.project_root / "submodules" / "diff-gaussian-rasterization" / "cuda_rasterizer"
        self.backup_dir = self.project_root / "backups"
        
    def check_environment(self):
        """检查环境和文件"""
        print("🔍 检查环境...")
        
        checks = {
            "项目根目录": self.project_root.exists(),
            "CUDA目录": self.cuda_dir.exists(),
            "forward.cu": (self.cuda_dir / "forward.cu").exists(),
            "backward.cu": (self.cuda_dir / "backward.cu").exists(),
            "forward_enhanced.cu": (self.cuda_dir / "forward_enhanced.cu").exists(),
            "backward_enhanced.cu": (self.cuda_dir / "backward_enhanced.cu").exists(),
        }
        
        all_ok = True
        for name, status in checks.items():
            icon = "✅" if status else "❌"
            print(f"  {icon} {name}")
            if not status:
                all_ok = False
        
        if not all_ok:
            print("\n⚠️  环境检查失败，请确保所有文件都存在")
            return False
        
        print("\n✅ 环境检查通过")
        return True
    
    def backup_files(self):
        """备份原始文件"""
        print("\n💾 备份原始文件...")
        
        self.backup_dir.mkdir(exist_ok=True)
        
        files_to_backup = [
            "forward.cu",
            "backward.cu",
            "forward.h",
        ]
        
        for filename in files_to_backup:
            src = self.cuda_dir / filename
            if src.exists():
                dst = self.backup_dir / f"{filename}.backup"
                shutil.copy2(src, dst)
                print(f"  ✅ 已备份: {filename} -> {dst}")
            else:
                print(f"  ⚠️  文件不存在: {filename}")
        
        print("\n✅ 备份完成")
    
    def integrate_enhancements(self):
        """集成增强功能"""
        print("\n🔧 集成增强功能...")
        
        print("\n📝 请手动完成以下步骤:")
        print("\n1. 打开 forward.cu")
        print("   - 在第18行附近，更新 FisheyeCameraParams 结构")
        print("   - 添加新的阈值参数和开关")
        print()
        print("2. 添加辅助函数")
        print("   - 在 computeCov3D 后添加 trace() 函数")
        print()
        print("3. 从 forward_enhanced.cu 复制以下函数:")
        print("   - compute_hessian_u()")
        print("   - compute_hessian_v()")
        print("   - compute_second_order_correction()")
        print("   - compute_numerical_jacobian()")
        print("   - compute_analytical_jacobian()")
        print("   - compute_adaptive_jacobian()")
        print()
        print("4. 修改 preprocessCUDA 函数")
        print("   - 找到第247-310行的雅可比计算部分")
        print("   - 替换为自适应雅可比调用")
        print("   - 添加二阶修正计算")
        print()
        print("5. 更新 backward.cu")
        print("   - 从 backward_enhanced.cu 复制相关函数")
        print("   - 更新 computeCov2DCUDA 函数")
        print()
        
        response = input("\n是否已完成手动集成? (y/n): ")
        if response.lower() != 'y':
            print("⚠️  请完成手动集成后再继续")
            return False
        
        print("✅ 集成完成")
        return True
    
    def compile_cuda(self):
        """重新编译CUDA扩展"""
        print("\n🔨 重新编译CUDA扩展...")
        
        rasterization_dir = self.project_root / "submodules" / "diff-gaussian-rasterization"
        
        # 清理旧的编译文件
        print("  清理旧的编译文件...")
        build_dir = rasterization_dir / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
            print("  ✅ 已删除 build/")
        
        # 编译
        print("\n  开始编译...")
        try:
            os.chdir(rasterization_dir)
            result = subprocess.run(
                [sys.executable, "setup.py", "install"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("  ✅ 编译成功")
                return True
            else:
                print("  ❌ 编译失败")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"  ❌ 编译出错: {e}")
            return False
        finally:
            os.chdir(self.project_root)
    
    def run_tests(self):
        """运行测试"""
        print("\n🧪 运行测试...")
        
        print("\n测试1: 导入测试")
        try:
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
            print("  ✅ 导入成功")
        except Exception as e:
            print(f"  ❌ 导入失败: {e}")
            return False
        
        print("\n测试2: 参数测试")
        try:
            import torch
            settings = GaussianRasterizationSettings(
                image_height=512,
                image_width=512,
                tanfovx=0.5,
                tanfovy=0.5,
                bg=torch.zeros(3),
                scale_modifier=1.0,
                viewmatrix=torch.eye(4),
                projmatrix=torch.eye(4),
                sh_degree=3,
                campos=torch.zeros(3),
                prefiltered=False,
                debug=False,
                # 新增参数
                fisheye=True,
                kb_params=torch.tensor([0.1, 0.01, 0.001, 0.0001]),
                max_theta=1.57,
                cx=256.0,
                cy=256.0,
                fx=500.0,
                fy=500.0
            )
            print("  ✅ 参数创建成功")
            print(f"     fisheye: {settings.fisheye}")
            print(f"     kb_params: {settings.kb_params}")
        except Exception as e:
            print(f"  ❌ 参数测试失败: {e}")
            return False
        
        print("\n✅ 所有测试通过")
        return True
    
    def run_all(self):
        """执行所有步骤"""
        print("=" * 60)
        print("🚀 开始完整集成流程")
        print("=" * 60)
        
        steps = [
            ("检查环境", self.check_environment),
            ("备份文件", self.backup_files),
            ("集成功能", self.integrate_enhancements),
            ("编译CUDA", self.compile_cuda),
            ("运行测试", self.run_tests),
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'=' * 60}")
            print(f"步骤: {step_name}")
            print(f"{'=' * 60}")
            
            if not step_func():
                print(f"\n❌ 步骤失败: {step_name}")
                return False
        
        print("\n" + "=" * 60)
        print("🎉 所有步骤完成！")
        print("=" * 60)
        print("\n下一步:")
        print("1. 运行训练测试: python train.py --source_path <data> --fisheye")
        print("2. 查看实现指南: 二阶修正和自适应雅可比实现指南.md")
        print("3. 调优参数以获得最佳效果")
        return True

def main():
    parser = argparse.ArgumentParser(description="集成二阶修正和自适应雅可比")
    parser.add_argument("--check", action="store_true", help="检查环境")
    parser.add_argument("--backup", action="store_true", help="备份原文件")
    parser.add_argument("--integrate", action="store_true", help="集成新功能")
    parser.add_argument("--compile", action="store_true", help="重新编译")
    parser.add_argument("--test", action="store_true", help="运行测试")
    parser.add_argument("--all", action="store_true", help="执行所有步骤")
    parser.add_argument("--project-root", default=".", help="项目根目录")
    
    args = parser.parse_args()
    
    integrator = EnhancementIntegrator(args.project_root)
    
    if args.all:
        integrator.run_all()
    elif args.check:
        integrator.check_environment()
    elif args.backup:
        integrator.backup_files()
    elif args.integrate:
        integrator.integrate_enhancements()
    elif args.compile:
        integrator.compile_cuda()
    elif args.test:
        integrator.run_tests()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
