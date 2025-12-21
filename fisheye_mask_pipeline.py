#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从原始鱼眼图像生成：
1) 二值图 (xxx_bin.png)
2) 形态学处理结果 (xxx_morph.png)
3) 最终掩码 (xxx_mask.png)

不需要标定文件，默认用图像中心 + 圆形 FOV 裁掉黑边。
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fisheye mask pipeline: binarization + morphology + final mask."
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="输入原始鱼眼图像文件夹"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="输出文件夹（会自动创建）"
    )
    parser.add_argument(
        "--radius_scale", type=float, default=0.98,
        help="圆形 FOV 半径缩放系数，默认 0.98（越小圆越小）"
    )
    parser.add_argument(
        "--binary_thresh", type=int, default=-1,
        help="二值化阈值；<0 使用 OTSU 自动阈值，>=0 使用固定阈值"
    )
    parser.add_argument(
        "--morph_ksize", type=int, default=5,
        help="形态学核大小（奇数），默认 5"
    )
    parser.add_argument(
        "--morph_iter", type=int, default=1,
        help="形态学迭代次数，默认 1"
    )
    parser.add_argument(
        "--erode_final", type=int, default=0,
        help="对最终掩码额外腐蚀的像素数（>0 会再缩一点边缘）"
    )
    return parser.parse_args()


def generate_circle_fov(h, w, radius_scale=0.98):
    """
    以图像中心为圆心，min(h,w)/2 * radius_scale 为半径，生成圆形 FOV 掩码（0/255）。
    """
    cx = w / 2.0
    cy = h / 2.0
    r = min(cx, cy) * radius_scale

    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij"
    )

    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[dist2 <= r * r] = 255
    return mask


def binarize_from_image(img_gray, thresh=-1):
    """
    从灰度图做二值化：
    - thresh < 0: 使用 OTSU 自动阈值
    - thresh >=0: 使用固定阈值
    """
    if thresh < 0:
        # OTSU 自动阈值
        _, binary = cv2.threshold(
            img_gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        _, binary = cv2.threshold(
            img_gray, thresh, 255,
            cv2.THRESH_BINARY
        )
    return binary


def morphology_refine(binary, ksize=5, iterations=1):
    """
    对二值图 binary 做形态学处理：
    1) 开运算（去小噪点）
    2) 闭运算（填小洞）
    """
    if ksize < 1:
        return binary

    if ksize % 2 == 0:
        ksize += 1  # 保证是奇数

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    opened = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel, iterations=iterations
    )
    closed = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, kernel, iterations=iterations
    )

    return closed


def extra_erode(mask, erode_pixels):
    """
    对最终掩码再腐蚀 erode_pixels 像素，进一步缩一圈边缘（可选）。
    """
    if erode_pixels <= 0:
        return mask

    ksize = 2 * erode_pixels + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ksize, ksize)
    )
    eroded = cv2.erode(mask, kernel, iterations=1)
    return eroded


def main():
    args = parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] image_dir = {image_dir}")
    print(f"[INFO] out_dir   = {out_dir}")

    image_paths = sorted(
        [p for p in image_dir.iterdir()
         if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]]
    )
    print(f"[INFO] Found {len(image_paths)} images.")

    if not image_paths:
        print(f"[WARN] No images found in {image_dir}")
        return

    for p in image_paths:
        print(f"[INFO] Processing: {p.name}")
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read image: {p}")
            continue

        h, w = img.shape[:2]

        # 1) 灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2) 二值化
        binary = binarize_from_image(gray, thresh=args.binary_thresh)

        # 3) 形态学处理
        morph = morphology_refine(
            binary,
            ksize=args.morph_ksize,
            iterations=args.morph_iter
        )

        # 4) 圆形 FOV 掩码
        fov_mask = generate_circle_fov(h, w, radius_scale=args.radius_scale)

        # 5) 最终掩码 = 形态学结果 ∧ FOV
        final_mask = cv2.bitwise_and(morph, fov_mask)

        # 6) （可选）再腐蚀一圈，避免边缘伪影
        final_mask = extra_erode(final_mask, args.erode_final)

        # --- 保存中间结果与最终掩码 ---
        bin_out = out_dir / f"{p.stem}_bin.png"
        morph_out = out_dir / f"{p.stem}_morph.png"
        final_out = out_dir / f"{p.stem}_mask.png"

        cv2.imwrite(str(bin_out), binary)
        cv2.imwrite(str(morph_out), morph)
        cv2.imwrite(str(final_out), final_mask)

        print(f"[OK] Saved: {bin_out.name}, {morph_out.name}, {final_out.name}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
