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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace#argparse 是 Python 的一个标准库，主要用于处理命令行参数和选项。它提供了一种简单而灵活的方法来解析命令行输入，使得开发者可以轻松地为自己的程序添加参数支持。以下是 argparse 的主要功能和特点：
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # dataset: 数据集的参数。opt: 优化选项。 pipe: 渲染管道参。testing_iterations: 测试的迭代次数。
    # saving_iterations: 保存的迭代次数。 checkpoint_iterations: 保存检查点的迭代次数。checkpoint: 起始检查点文件的路径。debug_from: 从哪个迭代开始调试。
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)#准备输出目录和记录器
    gaussians = GaussianModel(dataset.sh_degree)#创建高斯模型和场景对象。
    scene = Scene(dataset, gaussians)#调用场景函数
    gaussians.training_setup(opt)#设置优化器？？
    if checkpoint:#如果提供了检查点，加载模型参数和迭代次数。？？
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]#根据数据集的背景属性设置背景颜色。
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)#迭代器起始终止
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")#使用 tqdm 创建进度条，跟踪训练进度
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1): #在每个迭代中，检查与 GUI 的连接，并通过网络接收自定义相机参数，进行训练。

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render随机选择一个相机进行渲染。
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()#计算损失（L1损失和SSIM损失），并进行反向传播。


        iter_end.record()

        with torch.no_grad():#这个上下文管理器用于禁用梯度计算。
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            #loss.item(): 这是当前迭代的损失值。.item() 方法将一个单元素张量转换为 Python 数值。
            # 指数移动平均 (EMA): 通过将当前损失值和之前的 EMA 结合，这行代码平滑了损失曲线，减少了波动。
            # 0.4 和 0.6 是平滑因子，其中 0.4 是当前损失的权重，0.6 是历史值的权重。
            if iteration % 10 == 0:#每经过 10 次迭代更新一次进度条。
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                #更新进度条的后缀信息，这里显示当前的 EMA 损失，格式化为 7 位小数。
                progress_bar.update(10)#进度条向前移动 10 个单位，表示已经完成了 10 次迭代
            if iteration == opt.iterations:#在达到指定的最大迭代次数后，关闭进度条，以清理界面并结束训练过程的可视化。
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification在特定迭代内进行高斯模型的稠密化和修剪。
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        # 该函数负责准备输出文件夹和日志记录器。
        #
        # 输出文件夹:
        # 如果没有提供模型路径，则生成一个唯一的路径。
        # 创建路径并保存配置参数。
        # TensorBoard:
        # 检查是否可用，并创建TensorBoard的记录器。
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    # 该函数用于记录训练过程中的损失和性能指标。
    #
    # 记录损失:
    # 如果存在TensorBoard记录器，记录L1损失、总损失和迭代时间。
    # 测试报告:
    # 在指定迭代时进行模型评估，记录L1损失和PSNR（峰值信噪比）指标
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    #这行代码用于检查当前脚本是否是主程序执行的。如果是，下面的代码块将被运行。这是Python的标准做法，用于区分模块导入和直接运行。
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    #创建一个命令行参数解析器 (ArgumentParser) 以处理用户在命令行中输入的参数。description 参数提供了该脚本的简要说明。
    lp = ModelParams(parser)
    #解析与模型相关的参数，并将其存储在 lp 变量中。ModelParams 是一个自定义类，负责定义和解析特定的参数。
    op = OptimizationParams(parser)
    #解析与优化相关的参数，存储在 op 变量中。OptimizationParams 是另一个自定义类，类似于 ModelParams。
    pp = PipelineParams(parser)
    #解析与渲染管道相关的参数，存储在 pp 变量中。PipelineParams 也可能是自定义的参数解析类
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    #添加一个命令行参数 --ip，用于指定 IP 地址，默认值为 "127.0.0.1"（本地主机）
    parser.add_argument('--port', type=int, default=6009)
    #添加一个命令行参数 --port，用于指定端口号，默认值为 6009。
    parser.add_argument('--debug_from', type=int, default=-1)
    #添加一个命令行参数 --debug_from，用于指定从哪个迭代开始调试，默认值为 -1，表示不调试。
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    #添加一个布尔参数 --detect_anomaly，如果在命令行中提供该参数，将其设置为 True，用于开启异常检测。
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    #添加一个命令行参数 --test_iterations，用于指定测试的迭代次数，可以接受多个整数，默认值为 [7000, 30000]。
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    #解析命令行参数，将结果存储在 args 变量中。sys.argv[1:] 代表从命令行中获取的参数（不包括脚本名）。
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    #初始化网络 GUI 服务器，使用指定的 IP 地址和端口，准备与用户界面进行通信。
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    #设置 PyTorch 的异常检测功能。如果 args.detect_anomaly 为 True，则开启该功能，以便在训练过程中捕捉可能出现的异常。
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    #调用 training 函数，传入解析的参数，开始模型训练。这里提取了之前解析的模型参数、优化参数和管道参数。
    # All done
    print("\nTraining complete.")
