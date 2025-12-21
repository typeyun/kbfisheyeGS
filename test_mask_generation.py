#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掩码生成功能测试脚本
测试所有掩码生成功能是否正常工作
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil

# 设置UTF-8编码输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def test_simple_mask_generation():
    """测试简易掩码生成"""
    print("\n" + "="*60)
    print("测试1: 简易掩码生成 (fisheye_3dgs_mask_simple.py)")
    print("="*60)
    
    # 创建临时测试图像
    temp_dir = tempfile.mkdtemp()
    image_dir = Path(temp_dir) / "images"
    mask_dir = Path(temp_dir) / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 创建测试图像（模拟鱼眼图像）
        h, w = 512, 512
        test_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 画一个圆形区域（模拟鱼眼有效区域）
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 20
        cv2.circle(test_image, center, radius, (255, 255, 255), -1)
        
        # 保存测试图像
        test_image_path = image_dir / "test_fisheye.png"
        cv2.imwrite(str(test_image_path), test_image)
        print(f"✅ 创建测试图像: {test_image_path}")
        
        # 导入并运行掩码生成
        sys.path.insert(0, str(Path(__file__).parent))
        from fisheye_3dgs_mask_simple import generate_circle_mask, apply_morphology
        
        # 生成圆形掩码
        mask = generate_circle_mask(h, w, radius_scale=0.98)
        print(f"✅ 生成圆形掩码: shape={mask.shape}, dtype={mask.dtype}")
        
        # 检查掩码值
        unique_values = np.unique(mask)
        print(f"✅ 掩码唯一值: {unique_values}")
        assert set(unique_values).issubset({0, 255}), "掩码应该只包含0和255"
        
        # 应用形态学处理
        mask_eroded = apply_morphology(mask, erode_pixels=2)
        print(f"✅ 形态学处理完成: shape={mask_eroded.shape}")
        
        # 保存掩码
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / "test_fisheye_mask.png"
        cv2.imwrite(str(mask_path), mask_eroded)
        print(f"✅ 保存掩码: {mask_path}")
        
        # 验证掩码文件
        loaded_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert loaded_mask is not None, "无法加载保存的掩码"
        print(f"✅ 验证掩码文件: shape={loaded_mask.shape}")
        
        # 统计有效像素
        valid_pixels = np.sum(mask_eroded == 255)
        total_pixels = h * w
        valid_ratio = valid_pixels / total_pixels * 100
        print(f"✅ 有效像素比例: {valid_ratio:.2f}%")
        
        print("\n✅ 测试1通过: 简易掩码生成功能正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_pipeline_mask_generation():
    """测试完整管道掩码生成"""
    print("\n" + "="*60)
    print("测试2: 完整管道掩码生成 (fisheye_mask_pipeline.py)")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    image_dir = Path(temp_dir) / "images"
    mask_dir = Path(temp_dir) / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 创建测试图像
        h, w = 512, 512
        test_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 创建渐变圆形（模拟真实鱼眼）
        center = (w // 2, h // 2)
        for r in range(min(w, h) // 2, 0, -1):
            intensity = int(255 * r / (min(w, h) // 2))
            cv2.circle(test_image, center, r, (intensity, intensity, intensity), 1)
        
        test_image_path = image_dir / "test_fisheye.png"
        cv2.imwrite(str(test_image_path), test_image)
        print(f"✅ 创建测试图像: {test_image_path}")
        
        # 导入管道函数
        from fisheye_mask_pipeline import (
            generate_circle_fov,
            binarize_from_image,
            morphology_refine,
            extra_erode
        )
        
        # 读取图像
        img = cv2.imread(str(test_image_path), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"✅ 读取图像: shape={img.shape}")
        
        # 1. 二值化
        binary = binarize_from_image(gray, thresh=-1)  # OTSU
        print(f"✅ 二值化完成: unique values={np.unique(binary)}")
        
        # 2. 形态学处理
        morph = morphology_refine(binary, ksize=5, iterations=1)
        print(f"✅ 形态学处理完成")
        
        # 3. 圆形FOV
        fov_mask = generate_circle_fov(h, w, radius_scale=0.98)
        print(f"✅ 生成FOV掩码")
        
        # 4. 最终掩码
        final_mask = cv2.bitwise_and(morph, fov_mask)
        final_mask = extra_erode(final_mask, erode_pixels=2)
        print(f"✅ 生成最终掩码")
        
        # 保存所有中间结果
        mask_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_dir / "test_bin.png"), binary)
        cv2.imwrite(str(mask_dir / "test_morph.png"), morph)
        cv2.imwrite(str(mask_dir / "test_fov.png"), fov_mask)
        cv2.imwrite(str(mask_dir / "test_mask.png"), final_mask)
        print(f"✅ 保存所有掩码文件")
        
        # 验证文件
        for filename in ["test_bin.png", "test_morph.png", "test_fov.png", "test_mask.png"]:
            filepath = mask_dir / filename
            assert filepath.exists(), f"文件不存在: {filepath}"
            mask = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
            assert mask is not None, f"无法读取: {filepath}"
            print(f"  ✓ {filename}: shape={mask.shape}")
        
        print("\n✅ 测试2通过: 完整管道掩码生成功能正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_postprocess_masks():
    """测试掩码后处理"""
    print("\n" + "="*60)
    print("测试3: 掩码后处理 (postprocess_masks.py)")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    mask_dir = Path(temp_dir) / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 创建测试掩码（带噪声）
        h, w = 512, 512
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 主要区域
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 20
        cv2.circle(mask, center, radius, 255, -1)
        
        # 添加噪声
        noise_points = np.random.randint(0, min(h, w), (100, 2))
        for pt in noise_points:
            cv2.circle(mask, tuple(pt), 2, 255, -1)
        
        # 添加小洞
        hole_points = np.random.randint(0, min(h, w), (50, 2))
        for pt in hole_points:
            if mask[pt[1], pt[0]] == 255:
                cv2.circle(mask, tuple(pt), 3, 0, -1)
        
        mask_path = mask_dir / "test_mask.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"✅ 创建带噪声的测试掩码: {mask_path}")
        
        # 导入后处理函数
        from postprocess_masks import binarize_mask, morphology_refine
        
        # 二值化
        binary = binarize_mask(mask, thresh=128)
        print(f"✅ 二值化完成: unique values={np.unique(binary)}")
        
        # 形态学处理
        refined = morphology_refine(binary, ksize=5, iterations=1)
        print(f"✅ 形态学处理完成")
        
        # 比较处理前后
        noise_before = np.sum((mask == 255) & (refined == 0))
        holes_filled = np.sum((mask == 0) & (refined == 255))
        print(f"✅ 去除噪声像素: {noise_before}")
        print(f"✅ 填补空洞像素: {holes_filled}")
        
        # 保存结果
        out_dir = Path(temp_dir) / "masks_post"
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "test_bin.png"), binary)
        cv2.imwrite(str(out_dir / "test_morph.png"), refined)
        print(f"✅ 保存后处理结果")
        
        print("\n✅ 测试3通过: 掩码后处理功能正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_mask_quality():
    """测试掩码质量指标"""
    print("\n" + "="*60)
    print("测试4: 掩码质量评估")
    print("="*60)
    
    try:
        # 创建理想掩码
        h, w = 512, 512
        ideal_mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 20
        cv2.circle(ideal_mask, center, radius, 255, -1)
        
        # 生成测试掩码
        from fisheye_3dgs_mask_simple import generate_circle_mask, apply_morphology
        test_mask = generate_circle_mask(h, w, radius_scale=0.96)
        test_mask = apply_morphology(test_mask, erode_pixels=2)
        
        # 计算质量指标
        intersection = np.sum((ideal_mask == 255) & (test_mask == 255))
        union = np.sum((ideal_mask == 255) | (test_mask == 255))
        iou = intersection / union if union > 0 else 0
        
        valid_pixels = np.sum(test_mask == 255)
        total_pixels = h * w
        coverage = valid_pixels / total_pixels * 100
        
        print(f"✅ IoU (与理想掩码): {iou:.4f}")
        print(f"✅ 覆盖率: {coverage:.2f}%")
        print(f"✅ 有效像素数: {valid_pixels}")
        
        # 检查掩码连通性
        num_labels, labels = cv2.connectedComponents(test_mask)
        print(f"✅ 连通区域数: {num_labels - 1}")  # 减1是因为背景也算一个
        
        # 质量检查
        assert iou > 0.85, f"IoU过低: {iou}"
        assert coverage > 70, f"覆盖率过低: {coverage}%"
        assert num_labels <= 2, f"连通区域过多: {num_labels - 1}"
        
        print("\n✅ 测试4通过: 掩码质量符合要求")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试4失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("鱼眼掩码生成功能完整测试")
    print("="*60)
    
    results = []
    
    # 测试1: 简易掩码生成
    results.append(("简易掩码生成", test_simple_mask_generation()))
    
    # 测试2: 完整管道
    results.append(("完整管道掩码生成", test_pipeline_mask_generation()))
    
    # 测试3: 后处理
    results.append(("掩码后处理", test_postprocess_masks()))
    
    # 测试4: 质量评估
    results.append(("掩码质量评估", test_mask_quality()))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("🎉 所有测试通过！掩码功能100%完成")
        print("="*60)
        print("\n功能清单:")
        print("  ✅ 简易圆形掩码生成")
        print("  ✅ 基于亮度的掩码生成")
        print("  ✅ 形态学处理（开运算+闭运算）")
        print("  ✅ 掩码腐蚀（边缘收缩）")
        print("  ✅ 完整的掩码生成管道")
        print("  ✅ 掩码后处理")
        print("  ✅ 质量评估和验证")
        
        print("\n使用方法:")
        print("  1. 简易生成:")
        print("     python fisheye_3dgs_mask_simple.py --image_dir <images> --out_dir masks")
        print("\n  2. 完整管道:")
        print("     python fisheye_mask_pipeline.py --image_dir <images> --out_dir masks")
        print("\n  3. 后处理:")
        print("     python postprocess_masks.py")
        
        return 0
    else:
        print("\n" + "="*60)
        print("❌ 部分测试失败")
        print("="*60)
        return 1


if __name__ == "__main__":
    exit(main())
