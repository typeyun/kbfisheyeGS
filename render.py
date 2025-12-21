#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    #model_path: 模型的路径。
    #name: 数据集名称（如train or test？）
    #iteration: 当前迭代的编号。
    #views: 需要渲染的视图列表。
    #gaussians: 高斯模型，用于渲染。
    #pipeline: 渲染管道的参数。
    #background: 背景颜色。
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    #生成渲染输出和真实图像（ground truth, gt）的保存路径。
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    #创建保存路径，如果已存在则不会报错。
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):#遍历 views 列表，使用 tqdm 显示进度条。
        rendering = render(view, gaussians, pipeline, background)["render"] #渲染当前视图并获取渲染图像。
        gt = view.original_image[0:3, :, :] #提取视图的原始图像作为真实图像。

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))#将渲染的图像和真实图像保存为 PNG 文件，文件名格式为五位数（如 "00001.png"）。
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    #dataset: 数据集参数对象。iteration: 当前迭代的编号。pipeline: 渲染管道的参数。skip_train: 是否跳过训练集渲染。skip_test: 是否跳过测试集渲染。
    with torch.no_grad():#在不计算梯度的上下文中执行，节省内存。
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        #初始化高斯模型和场景对象。
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        #根据数据集的背景设置背景颜色。
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        #根据参数决定是否渲染训练集和测试集。
        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # 当你直接运行此脚本时，main()函数会被调用，。但是如果这个脚本被其他脚本导入时，main()函数不会被自动调用。
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")#创建命令行参数解析器。
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)#初始化模型和管道参数的解析。
    parser.add_argument("--iteration", default=-1, type=int)#添加命令行参数选项。
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")#添加命令行参数选项。
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)#获取组合的参数并打印模型路径。

    # Initialize system state (RNG)
    safe_state(args.quiet)#初始化系统状态，可能是设置随机数生成器（RNG）。？

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)#调用 render_sets 函数，传入解析后的参数。