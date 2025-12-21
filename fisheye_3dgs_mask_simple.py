
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简易鱼眼 3DGS 掩码生成（不需要标定文件）

思路：
1. 以图像中心为圆心，min(H, W)/2 * radius_scale 为半径，画一个圆形 FOV 掩码。
2. 可选：根据亮度阈值去掉明显的黑边。
3. 可选：对掩码做腐蚀，缩一圈边缘，避免黑边/畸变区进入训练。

输出：
    对每张图片 xxx.png 生成 xxx_mask.png（255=有效区域, 0=无效）。
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple fisheye 3DGS mask generator (no calib needed)."
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="输入鱼眼图像文件夹"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="掩码输出文件夹"
    )
    parser.add_argument(
        "--radius_scale", type=float, default=0.98,
        help="圆形 FOV 半径缩放系数，默认为 0.98（越小圆越小）"
    )
    parser.add_argument(
        "--erode_pixels", type=int, default=2,
        help="对掩码腐蚀的像素数量（>0 会缩小有效区域，防止边缘伪影）"
    )
    parser.add_argument(
        "--brightness_thresh", type=int, default=0,
        help="可选：根据亮度剔除明显黑边的阈值（0 表示不启用）"
    )
    return parser.parse_args()


def generate_circle_mask(h, w, radius_scale=0.98):
    """
    以图像中心为圆心，min(h,w)/2 * radius_scale 为半径，生成圆形掩码。
    """
    cx = w / 2.0
    cy = h / 2.0
    r = min(cx, cy) * radius_scale

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32),
                         indexing="ij")

    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[dist2 <= r * r] = 255
    return mask


def apply_brightness_mask(mask, image, brightness_thresh):
    """
    只在当前 mask=255 的地方，根据亮度进一步剔除黑边。
    """
    if brightness_thresh <= 0:
        return mask

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    out = mask.copy()
    valid = out == 255
    to_zero = (gray < brightness_thresh) & valid
    out[to_zero] = 0
    return out


def apply_morphology(mask, erode_pixels):
    """
    对掩码做腐蚀（缩小一圈），避免黑边和畸变严重区域进来。
    """
    if erode_pixels <= 0:
        return mask

    ksize = 2 * erode_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    eroded = cv2.erode(mask, kernel, iterations=1)
    return eroded


def main():
    args = parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [p for p in image_dir.iterdir()
         if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]]
    )
    if not image_paths:
        print(f"[Warn] No images found in {image_dir}")
        return

    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[Warn] Failed to read image: {p}")
            continue

        h, w = img.shape[:2]
        mask = generate_circle_mask(h, w, radius_scale=args.radius_scale)
        mask = apply_brightness_mask(mask, img, args.brightness_thresh)
        mask = apply_morphology(mask, args.erode_pixels)

        out_path = out_dir / f"{p.stem}_mask.png"
        cv2.imwrite(str(out_path), mask)
        print(f"[OK] Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
