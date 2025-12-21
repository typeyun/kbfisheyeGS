#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对已有掩码图像做二值化 + 形态学处理

输入：masks 目录下的 *_mask.png
输出：
    *_bin.png   二值化后的掩码
    *_morph.png 形态学处理（开运算 + 闭运算）后的掩码
"""

from pathlib import Path
import cv2
import numpy as np


def binarize_mask(mask, thresh=128):
    """
    把掩码二值化为 0 / 255
    """
    # 如果是 3 通道图，先转灰度
    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask

    # 固定阈值二值化（也可以换成 OTSU）
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return binary


def morphology_refine(binary, ksize=5, iterations=1):
    """
    对二值掩码做形态学处理：
    先开运算去噪，再闭运算填补小洞
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    # 开运算：先腐蚀再膨胀，去小噪点
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)

    # 闭运算：先膨胀再腐蚀，填小洞
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return closed


def main():
    mask_dir = Path("masks")  # 你的掩码文件夹
    out_dir = Path("masks_post")  # 输出文件夹
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(
        [p for p in mask_dir.iterdir()
         if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]]
    )

    print(f"[INFO] Found {len(mask_paths)} masks in {mask_dir}")

    for p in mask_paths:
        print(f"[INFO] Processing: {p.name}")
        mask = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"[WARN] Failed to read: {p}")
            continue

        # 1. 二值化
        binary = binarize_mask(mask, thresh=128)

        # 2. 形态学处理（你可以根据需要调 ksize / iterations）
        morph = morphology_refine(binary, ksize=5, iterations=1)

        # 保存结果
        bin_out = out_dir / f"{p.stem}_bin.png"
        morph_out = out_dir / f"{p.stem}_morph.png"

        cv2.imwrite(str(bin_out), binary)
        cv2.imwrite(str(morph_out), morph)

        print(f"[OK] Saved: {bin_out.name}, {morph_out.name}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
