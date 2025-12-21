@echo off
REM 渲染360度环绕视频帧
REM 使用方法: render_360.bat <model_path> [num_frames]

if "%1"=="" (
    echo 用法: render_360.bat ^<model_path^> [num_frames]
    echo 示例: render_360.bat output/my_model 120
    exit /b 1
)

set MODEL_PATH=%1
set NUM_FRAMES=%2
if "%NUM_FRAMES%"=="" set NUM_FRAMES=120

echo ========================================
echo 渲染360度环绕视频帧
echo 模型路径: %MODEL_PATH%
echo 帧数: %NUM_FRAMES%
echo ========================================

python render_novel_view.py ^
    --model_path %MODEL_PATH% ^
    --orbit ^
    --orbit_center 0 0 0 ^
    --orbit_radius 5 ^
    --orbit_views %NUM_FRAMES% ^
    --width 1920 ^
    --height 1080 ^
    --orbit_dir orbit_360

echo.
echo ========================================
echo 渲染完成！
echo 视频帧保存在: orbit_360/
echo.
echo 使用ffmpeg合成视频:
echo ffmpeg -framerate 30 -i orbit_360/view_%%04d.png -c:v libx264 -pix_fmt yuv420p output_360.mp4
echo ========================================
