import os
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# 函数：在图像上绘制patch方框
def draw_patches_on_image(image_path, coords_path, patch_size, save_path="./noise_show.png"):
    """
    在图像上绘制patch的方框，并保存为图片。

    :param image_path: 原始图像路径 (MRC文件或TIF文件)
    :param coords_path: 坐标文件路径
    :param patch_size: patch大小 (height, width)
    :param save_path: 保存图像路径
    """
    # 根据文件后缀名选择正确的读取方法
    file_extension = os.path.splitext(image_path)[1].lower()
    
    if file_extension == '.mrc':
        # 读取MRC图像
        with mrcfile.open(image_path, permissive=True) as mrc:
            image_array = mrc.data
    elif file_extension in ['.tif', '.tiff']:
        # 读取TIF图像
        image = Image.open(image_path)
        image_array = np.array(image)
    else:
        raise ValueError(f"不支持的文件格式: {file_extension}")

    # 归一化到 0-255 并转换为 PIL 图像
    # image_array = ((image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255).astype(np.uint8)
    image = Image.fromarray(image_array).convert("RGB")  # 转换为 RGB，避免白色背景

    # 绘制红色方框
    draw = ImageDraw.Draw(image)
    with open(coords_path, 'r') as f:
        coords = [line.strip().split('\t') for line in f]
        coords = [(int(float(c[0])), int(float(c[1]))) for c in coords]

    patch_height, patch_width = patch_size
    for center in coords:
        
        ## EMPIAR-10499、EMPIAR-10651
        top_left = (center[1] - patch_width // 2, center[0] - patch_height // 2)
        bottom_right = (center[1] + patch_width // 2, center[0] + patch_height // 2)
        
        ## EMPIAR-10164
        # new_center = (image_array.shape[0] - center[0], center[1])
        # top_left = (new_center[1] - patch_width // 2, new_center[0] - patch_height // 2)
        # bottom_right = (new_center[1] + patch_width // 2, new_center[0] + patch_height // 2)
        draw.rectangle([top_left, bottom_right], outline="red", width=2)

    # 直接保存，不显示白色背景
    image.save(save_path)
    print(f"图像已保存至: {save_path}")

# EMPIAR-10499
# image_directory = '/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10499/tilt_series_topaz'  # 原始图像目录
# coords_directory = '/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10499/tilt_series_noise_coord'  # 坐标文件目录

# EMPIAR-10164
image_directory = '/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10164/tilt_series_earse'  # 原始图像目录
coords_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPAIR-10164/noise_coords'  # 坐标文件目录

# EMPIAR-10651
image_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10651/tilt'  # 原始图像目录
coords_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10651/noise_coords'  # 坐标文件目录
file_extension = '.mrc'


# 参数配置
patch_size = (192, 192)  # 与之前一致

# 示例处理一个图像
example_file = sorted(os.listdir(coords_directory))[0]  # 仅处理第一个文件
image_file = os.path.join(image_directory, os.path.splitext(example_file)[0] + file_extension)
coords_file = os.path.join(coords_directory, example_file)

print(f"image_file={image_file}")
print(f"coords_file={coords_file}")

if os.path.exists(image_file):
    draw_patches_on_image(image_file, coords_file, patch_size)
else:
    print(f"对应图像文件不存在: {image_file}")
