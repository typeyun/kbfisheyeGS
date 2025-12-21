@echo off
REM 快速渲染脚本 - Windows批处理文件
REM 使用方法: quick_render.bat <model_path>

if "%1"=="" (
    echo 用法: quick_render.bat ^<model_path^>
    echo 示例: quick_render.bat output/my_model
    exit /b 1
)

set MODEL_PATH=%1

echo ========================================
echo 快速渲染预设视角
echo 模型路径: %MODEL_PATH%
echo ========================================

REM 创建输出目录
if not exist "quick_renders" mkdir quick_renders

echo.
echo [1/6] 渲染前视图...
python render_novel_view.py --model_path %MODEL_PATH% --position 0 0 5 --look_at 0 0 0 --output quick_renders/front.png

echo.
echo [2/6] 渲染后视图...
python render_novel_view.py --model_path %MODEL_PATH% --position 0 0 -5 --look_at 0 0 0 --output quick_renders/back.png

echo.
echo [3/6] 渲染左视图...
python render_novel_view.py --model_path %MODEL_PATH% --position -5 0 0 --look_at 0 0 0 --output quick_renders/left.png

echo.
echo [4/6] 渲染右视图...
python render_novel_view.py --model_path %MODEL_PATH% --position 5 0 0 --look_at 0 0 0 --output quick_renders/right.png

echo.
echo [5/6] 渲染俯视图...
python render_novel_view.py --model_path %MODEL_PATH% --position 0 5 0 --look_at 0 0 0 --output quick_renders/top.png

echo.
echo [6/6] 渲染斜视图...
python render_novel_view.py --model_path %MODEL_PATH% --position 3 3 3 --look_at 0 0 0 --output quick_renders/diagonal.png

echo.
echo ========================================
echo 渲染完成！
echo 图像保存在: quick_renders/
echo ========================================
