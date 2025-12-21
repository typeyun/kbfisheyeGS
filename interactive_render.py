"""
交互式渲染工具
提供简单的命令行界面来渲染不同视角
"""

import os
import sys
from render_novel_view import render_novel_view, render_orbit_views


def print_menu():
    print("\n" + "="*50)
    print("3D高斯渲染 - 交互式工具")
    print("="*50)
    print("1. 渲染单个自定义视角")
    print("2. 渲染预设视角（前/后/左/右/上/斜）")
    print("3. 渲染360度环绕视角")
    print("4. 渲染高清特写")
    print("5. 渲染全景视图")
    print("0. 退出")
    print("="*50)


def get_model_path():
    """获取模型路径"""
    while True:
        model_path = input("\n请输入模型路径（例如: output/my_model）: ").strip()
        if os.path.exists(model_path):
            return model_path
        else:
            print(f"错误: 路径 '{model_path}' 不存在，请重新输入")


def render_custom_view(model_path):
    """渲染自定义视角"""
    print("\n--- 自定义视角渲染 ---")
    
    # 获取相机位置
    print("\n相机位置（世界坐标系）:")
    x = float(input("  X坐标 [默认: 3]: ") or "3")
    y = float(input("  Y坐标 [默认: 2]: ") or "2")
    z = float(input("  Z坐标 [默认: 5]: ") or "5")
    position = [x, y, z]
    
    # 获取朝向点
    print("\n朝向点:")
    lx = float(input("  X坐标 [默认: 0]: ") or "0")
    ly = float(input("  Y坐标 [默认: 0]: ") or "0")
    lz = float(input("  Z坐标 [默认: 0]: ") or "0")
    look_at = [lx, ly, lz]
    
    # 获取图像尺寸
    print("\n图像尺寸:")
    width = int(input("  宽度 [默认: 1920]: ") or "1920")
    height = int(input("  高度 [默认: 1080]: ") or "1080")
    
    # 获取视场角
    print("\n视场角（度）:")
    fov_x = float(input("  水平FOV [默认: 60]: ") or "60")
    fov_y = float(input("  垂直FOV [默认: 45]: ") or "45")
    
    # 输出路径
    output = input("\n输出文件名 [默认: custom_view.png]: ").strip() or "custom_view.png"
    
    # 背景颜色
    white_bg = input("\n使用白色背景? (y/n) [默认: n]: ").strip().lower() == 'y'
    
    print("\n开始渲染...")
    render_novel_view(
        model_path=model_path,
        iteration=-1,
        position=position,
        look_at=look_at,
        fov_x=fov_x,
        fov_y=fov_y,
        width=width,
        height=height,
        output_path=output,
        white_background=white_bg
    )


def render_preset_views(model_path):
    """渲染预设视角"""
    print("\n--- 预设视角渲染 ---")
    
    output_dir = input("输出目录 [默认: preset_views]: ").strip() or "preset_views"
    os.makedirs(output_dir, exist_ok=True)
    
    radius = float(input("相机距离 [默认: 5]: ") or "5")
    
    presets = [
        ("前视图", [0, 0, radius], "front.png"),
        ("后视图", [0, 0, -radius], "back.png"),
        ("左视图", [-radius, 0, 0], "left.png"),
        ("右视图", [radius, 0, 0], "right.png"),
        ("俯视图", [0, radius, 0], "top.png"),
        ("斜视图", [radius*0.6, radius*0.6, radius*0.6], "diagonal.png"),
    ]
    
    print(f"\n将渲染 {len(presets)} 个预设视角...")
    
    for i, (name, position, filename) in enumerate(presets, 1):
        print(f"\n[{i}/{len(presets)}] 渲染{name}...")
        output_path = os.path.join(output_dir, filename)
        render_novel_view(
            model_path=model_path,
            iteration=-1,
            position=position,
            look_at=[0, 0, 0],
            width=1920,
            height=1080,
            output_path=output_path
        )
    
    print(f"\n所有预设视角渲染完成！保存在: {output_dir}/")


def render_orbit_360(model_path):
    """渲染360度环绕"""
    print("\n--- 360度环绕渲染 ---")
    
    output_dir = input("输出目录 [默认: orbit_360]: ").strip() or "orbit_360"
    
    print("\n环绕中心点:")
    cx = float(input("  X坐标 [默认: 0]: ") or "0")
    cy = float(input("  Y坐标 [默认: 0]: ") or "0")
    cz = float(input("  Z坐标 [默认: 0]: ") or "0")
    center = [cx, cy, cz]
    
    radius = float(input("\n环绕半径 [默认: 5]: ") or "5")
    height_offset = float(input("高度偏移 [默认: 0]: ") or "0")
    num_views = int(input("视角数量 [默认: 120]: ") or "120")
    
    print("\n图像尺寸:")
    width = int(input("  宽度 [默认: 1920]: ") or "1920")
    height = int(input("  高度 [默认: 1080]: ") or "1080")
    
    print("\n开始渲染...")
    render_orbit_views(
        model_path=model_path,
        iteration=-1,
        center_point=center,
        radius=radius,
        num_views=num_views,
        height_offset=height_offset,
        width=width,
        height=height,
        output_dir=output_dir
    )
    
    print(f"\n提示: 使用以下命令合成视频:")
    print(f"ffmpeg -framerate 30 -i {output_dir}/view_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4")


def render_closeup(model_path):
    """渲染高清特写"""
    print("\n--- 高清特写渲染 ---")
    
    print("\n相机位置（建议使用较近的距离）:")
    x = float(input("  X坐标 [默认: 1]: ") or "1")
    y = float(input("  Y坐标 [默认: 0.5]: ") or "0.5")
    z = float(input("  Z坐标 [默认: 1]: ") or "1")
    position = [x, y, z]
    
    output = input("\n输出文件名 [默认: closeup.png]: ").strip() or "closeup.png"
    
    print("\n开始渲染高清特写...")
    render_novel_view(
        model_path=model_path,
        iteration=-1,
        position=position,
        look_at=[0, 0, 0],
        fov_x=90,  # 广角
        fov_y=67.5,
        width=2560,
        height=1440,
        output_path=output
    )


def render_panorama(model_path):
    """渲染全景视图"""
    print("\n--- 全景视图渲染 ---")
    
    print("\n相机位置（建议使用较远的距离）:")
    x = float(input("  X坐标 [默认: 10]: ") or "10")
    y = float(input("  Y坐标 [默认: 5]: ") or "5")
    z = float(input("  Z坐标 [默认: 10]: ") or "10")
    position = [x, y, z]
    
    output = input("\n输出文件名 [默认: panorama.png]: ").strip() or "panorama.png"
    
    print("\n开始渲染全景视图...")
    render_novel_view(
        model_path=model_path,
        iteration=-1,
        position=position,
        look_at=[0, 0, 0],
        fov_x=45,  # 长焦
        fov_y=33.75,
        width=3840,
        height=2160,
        output_path=output
    )


def main():
    print("\n欢迎使用3D高斯渲染交互式工具！")
    
    # 获取模型路径
    model_path = get_model_path()
    print(f"\n已选择模型: {model_path}")
    
    while True:
        print_menu()
        choice = input("\n请选择操作 (0-5): ").strip()
        
        if choice == "0":
            print("\n再见！")
            break
        elif choice == "1":
            render_custom_view(model_path)
        elif choice == "2":
            render_preset_views(model_path)
        elif choice == "3":
            render_orbit_360(model_path)
        elif choice == "4":
            render_closeup(model_path)
        elif choice == "5":
            render_panorama(model_path)
        else:
            print("\n无效的选择，请重新输入")
        
        input("\n按回车键继续...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
