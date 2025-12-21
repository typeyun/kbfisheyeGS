#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick environment check and test script
"""

import os
import sys
from pathlib import Path

def check_environment():
    """Check environment and files"""
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    cuda_dir = project_root / "submodules" / "diff-gaussian-rasterization" / "cuda_rasterizer"
    
    checks = {
        "Project root": project_root.exists(),
        "CUDA directory": cuda_dir.exists(),
        "forward.cu": (cuda_dir / "forward.cu").exists(),
        "backward.cu": (cuda_dir / "backward.cu").exists(),
        "forward_enhanced.cu": (cuda_dir / "forward_enhanced.cu").exists(),
        "backward_enhanced.cu": (cuda_dir / "backward_enhanced.cu").exists(),
        "train.py": (project_root / "train.py").exists(),
    }
    
    all_ok = True
    for name, status in checks.items():
        icon = "[OK]" if status else "[FAIL]"
        print(f"  {icon} {name}")
        if not status:
            all_ok = False
    
    print()
    if all_ok:
        print("[SUCCESS] All files found!")
    else:
        print("[WARNING] Some files are missing")
    
    return all_ok

def test_import():
    """Test if diff_gaussian_rasterization can be imported"""
    print("=" * 60)
    print("Import Test")
    print("=" * 60)
    
    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        print("  [OK] diff_gaussian_rasterization imported successfully")
        return True
    except ImportError as e:
        print(f"  [FAIL] Cannot import: {e}")
        print("  [INFO] You may need to compile the CUDA extension first")
        return False

def test_pytorch():
    """Test PyTorch and CUDA"""
    print("=" * 60)
    print("PyTorch and CUDA Test")
    print("=" * 60)
    
    try:
        import torch
        print(f"  [OK] PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  [OK] CUDA version: {torch.version.cuda}")
            return True
        else:
            print("  [WARNING] CUDA not available")
            return False
    except ImportError:
        print("  [FAIL] PyTorch not installed")
        return False

def test_parameters():
    """Test creating rasterization settings with fisheye parameters"""
    print("=" * 60)
    print("Parameter Test")
    print("=" * 60)
    
    try:
        import torch
        from diff_gaussian_rasterization import GaussianRasterizationSettings
        
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
            fisheye=True,
            kb_params=torch.tensor([0.1, 0.01, 0.001, 0.0001]),
            max_theta=1.57,
            cx=256.0,
            cy=256.0,
            fx=500.0,
            fy=500.0
        )
        
        print("  [OK] Settings created successfully")
        print(f"       fisheye: {settings.fisheye}")
        print(f"       kb_params: {settings.kb_params}")
        print(f"       max_theta: {settings.max_theta}")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Cannot create settings: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("Quick Check Script")
    print("=" * 60 + "\n")
    
    results = []
    
    # Check 1: Environment
    results.append(("Environment", check_environment()))
    print()
    
    # Check 2: PyTorch
    results.append(("PyTorch", test_pytorch()))
    print()
    
    # Check 3: Import
    results.append(("Import", test_import()))
    print()
    
    # Check 4: Parameters (only if import succeeded)
    if results[-1][1]:
        results.append(("Parameters", test_parameters()))
        print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, status in results:
        icon = "[PASS]" if status else "[FAIL]"
        print(f"  {icon} {name}")
    
    all_passed = all(status for _, status in results)
    
    print()
    if all_passed:
        print("[SUCCESS] All checks passed!")
        print("\nNext steps:")
        print("  1. You can start training with: python train.py --source_path <data> --fisheye")
        print("  2. Or integrate enhancements following the guide")
    else:
        print("[INFO] Some checks failed")
        print("\nNext steps:")
        if not results[2][1]:  # Import failed
            print("  1. Compile CUDA extension:")
            print("     cd submodules/diff-gaussian-rasterization")
            print("     python setup.py install")
        print("  2. Check the detailed error messages above")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
