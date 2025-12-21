# 渲染任意新视角 - 使用示例

## 功能说明

`render_novel_view.py` 脚本可以让你渲染训练好的3D高斯模型的任意新视角，支持：

1. **单个自定义视角渲染**：指定相机位置、朝向、视场角等参数
2. **环绕视角渲染**：自动生成围绕某个中心点的多个视角
3. **鱼眼相机支持**：可以使用鱼眼相机模型（KB模型）

## 基本用法

### 1. 渲染单个视角

```bash
# 最简单的用法（使用默认参数）
python render_novel_view.py --model_path output/your_model

# 自定义相机位置和朝向
python render_novel_view.py \
    --model_path output/your_model \
    --position 3 2 5 \
    --look_at 0 0 0 \
    --output my_view.png

# 自定义图像尺寸和视场角
python render_novel_view.py \
    --model_path output/your_model \
    --position 3 2 5 \
    --look_at 0 0 0 \
    --width 1920 \
    --height 1080 \
    --fov_x 70 \
    --fov_y 50 \
    --output hd_view.png

# 使用白色背景
python render_novel_view.py \
    --model_path output/your_model \
    --position 3 2 5 \
    --white_background \
    --output white_bg_view.png
```

### 2. 渲染环绕视角（360度旋转）

```bash
# 生成36个环绕视角（每10度一个）
python render_novel_view.py \
    --model_path output/your_model \
    --orbit \
    --orbit_center 0 0 0 \
    --orbit_radius 5 \
    --orbit_views 36 \
    --orbit_dir orbit_360

# 从更高的位置环绕（俯视效果）
python render_novel_view.py \
    --model_path output/your_model \
    --orbit \
    --orbit_center 0 0 0 \
    --orbit_radius 5 \
    --orbit_height 2 \
    --orbit_views 36 \
    --orbit_dir orbit_high

# 生成高清环绕视频帧
python render_novel_view.py \
    --model_path output/your_model \
    --orbit \
    --orbit_center 0 0 0 \
    --orbit_radius 5 \
    --orbit_views 120 \
    --width 1920 \
    --height 1080 \
    --orbit_dir orbit_hd
```

### 3. 使用鱼眼相机模型

```bash
# 使用鱼眼相机渲染
python render_novel_view.py \
    --model_path output/your_model \
    --position 3 2 5 \
    --fisheye \
    --kb_k1 0.5 \
    --kb_k2 0.3 \
    --output fisheye_view.png
```

## 参数说明

### 基本参数
- `--model_path`: 训练好的模型路径（必需）
- `--iteration`: 使用的迭代次数，-1表示最新（默认：-1）
- `--output`: 输出图像路径（默认：novel_view.png）

### 相机位置参数
- `--position X Y Z`: 相机在世界坐标系中的位置（默认：0 0 5）
- `--look_at X Y Z`: 相机朝向的目标点（默认：0 0 0）
- `--up X Y Z`: 相机的上方向向量（默认：0 1 0）

### 视角参数
- `--fov_x`: 水平视场角，单位度（默认：60）
- `--fov_y`: 垂直视场角，单位度（默认：45）
- `--width`: 图像宽度（默认：800）
- `--height`: 图像高度（默认：600）

### 渲染参数
- `--white_background`: 使用白色背景（默认：黑色）
- `--fisheye`: 使用鱼眼相机模型

### 环绕模式参数
- `--orbit`: 启用环绕模式
- `--orbit_center X Y Z`: 环绕中心点（默认：0 0 0）
- `--orbit_radius`: 环绕半径（默认：5.0）
- `--orbit_views`: 生成的视角数量（默认：36）
- `--orbit_height`: 相机高度偏移（默认：0.0）
- `--orbit_dir`: 输出目录（默认：orbit_views）

### KB模型参数（鱼眼）
- `--kb_k1`, `--kb_k2`, `--kb_k3`, `--kb_k4`: KB模型畸变参数

## 坐标系说明

- **X轴**：通常指向右方
- **Y轴**：通常指向上方
- **Z轴**：通常指向前方（相机朝向的反方向）

例如：
- `--position 5 0 0`：相机在右侧
- `--position 0 5 0`：相机在上方（俯视）
- `--position 0 0 5`：相机在前方

## 实用场景示例

### 场景1：从不同高度观察物体

```bash
# 平视
python render_novel_view.py --model_path output/model --position 5 0 0 --output view_level.png

# 俯视
python render_novel_view.py --model_path output/model --position 5 3 0 --output view_top.png

# 仰视
python render_novel_view.py --model_path output/model --position 5 -2 0 --output view_bottom.png
```

### 场景2：创建视频帧（用于制作旋转视频）

```bash
# 生成120帧（3秒@30fps或4秒@30fps）
python render_novel_view.py \
    --model_path output/model \
    --orbit \
    --orbit_views 120 \
    --width 1920 \
    --height 1080 \
    --orbit_dir video_frames

# 然后使用ffmpeg合成视频
# ffmpeg -framerate 30 -i video_frames/view_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4
```

### 场景3：特写镜头

```bash
# 近距离观察（小半径）
python render_novel_view.py \
    --model_path output/model \
    --position 1 0.5 1 \
    --look_at 0 0 0 \
    --fov_x 90 \
    --output closeup.png
```

### 场景4：全景视图

```bash
# 远距离观察（大半径）
python render_novel_view.py \
    --model_path output/model \
    --position 10 5 10 \
    --look_at 0 0 0 \
    --fov_x 45 \
    --output panorama.png
```

## 提示和技巧

1. **找到合适的相机位置**：
   - 先用小图像尺寸（如400x300）快速测试不同位置
   - 确定好位置后再用高分辨率渲染

2. **视场角选择**：
   - 标准镜头：FOV 40-60度
   - 广角镜头：FOV 70-90度
   - 长焦镜头：FOV 20-40度

3. **环绕视频制作**：
   - 30fps视频：每秒需要30帧
   - 想要3秒旋转一圈：需要90帧（`--orbit_views 90`）
   - 想要更流畅：增加帧数到120或更多

4. **性能优化**：
   - 渲染大量视角时，可以先用低分辨率测试
   - 使用GPU加速（自动使用CUDA）

5. **批量渲染**：
   - 可以写脚本循环调用，生成多个不同参数的视角
   - 环绕模式已经自动批量渲染

## 故障排除

### 问题1：渲染结果是黑色的
- 检查相机位置是否在场景范围内
- 尝试调整 `--look_at` 参数指向场景中心
- 检查模型是否正确加载

### 问题2：图像看起来很奇怪
- 检查 `--up` 向量是否正确（通常是 0 1 0）
- 调整视场角参数
- 确保相机位置和朝向合理

### 问题3：内存不足
- 减小图像尺寸
- 减少环绕视角数量
- 使用较早的迭代次数（模型更小）

## 进阶：在Python代码中使用

```python
from render_novel_view import render_novel_view, render_orbit_views

# 渲染单个视角
render_novel_view(
    model_path="output/your_model",
    iteration=-1,
    position=[3, 2, 5],
    look_at=[0, 0, 0],
    fov_x=60.0,
    fov_y=45.0,
    width=1920,
    height=1080,
    output_path="my_view.png"
)

# 渲染环绕视角
render_orbit_views(
    model_path="output/your_model",
    iteration=-1,
    center_point=[0, 0, 0],
    radius=5.0,
    num_views=36,
    height_offset=2.0,
    output_dir="orbit_views"
)
```

## 相关文件

- `render_novel_view.py`: 主脚本
- `scene/cameras.py`: 相机类定义
- `gaussian_renderer/__init__.py`: 渲染器
