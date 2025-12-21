#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#这段代码是一个 Python 脚本，用于处理 COLMAP 格式的相机数据和图像，包括特征提取、匹配、束调整和图像去畸变等步骤。

import os
import logging
from argparse import ArgumentParser
import shutil
source_path = r'D:\SLAM\gaussian-splatting-main\dataplayroom'
# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="RADIAL_FISHEYE", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching创建用于特征匹配的 COLMAP 命令。
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment BA 创建用于调整相机和场景点位置的 COLMAP 命令。
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion创建用于将图像去畸变到理想针孔模型的 COLMAP 命令。
## We need to undistort our images into ideal pinhole intrinsics.
# img_undist_cmd = (colmap_command + " image_undistorter \
#     --image_path " + args.source_path + "/input \
#     --input_path " + args.source_path + "/distorted/sparse/0 \
#     --output_path " + args.source_path + "\
#     --output_type COLMAP")
# exit_code = os.system(img_undist_cmd)#执行去畸变：运行去畸变命令并检查错误。
# if exit_code != 0:
#     logging.error(f"Mapper failed with code {exit_code}. Exiting.")
#     exit(exit_code)

sparse_path = args.source_path + "/distorted/sparse/0"

# 确保输出目录存在
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)

# 将原始的重建结果复制到输出目录
for file in os.listdir(sparse_path):
    source_file = os.path.join(sparse_path, file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.copy2(source_file, destination_file)

# Convert sparse reconstruction to text format
model_converter_cmd = (colmap_command + " model_converter \
    --input_path " + args.source_path + "/sparse/0 \
    --output_path " + args.source_path + "/sparse/0 \
    --output_type TXT")
exit_code = os.system(model_converter_cmd)
if exit_code != 0:
    logging.error(f"Model conversion failed with code {exit_code}. Exiting.")
    exit(exit_code)

print("Done.")
