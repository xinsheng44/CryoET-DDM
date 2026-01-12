import os
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# 函数：在图像上绘制patch方框
def draw_patches_on_image(image_path, coords_path, patch_size, save_path="./particle_show.png"):
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
        from PIL import Image as PILImage
        image = PILImage.open(image_path)
        image_array = np.array(image)
    else:
        raise ValueError(f"不支持的文件格式: {file_extension}")

    # 归一化到 0-255 并转换为 PIL 图像
    # image_array = ((image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255).astype(np.uint8)
    image = Image.fromarray(image_array).convert("RGB")  # 转换为 RGB，避免白色背景

    # 绘制红色方框
    draw = ImageDraw.Draw(image)
    with open(coords_path, 'r') as f:
        coords = []
        for line in f:
            # 尝试按制表符分割，如果只有一个元素，则按空格分割
            parts = line.strip().split('\t')
            if len(parts) < 2:
                parts = line.strip().split()
            if len(parts) >= 2:  # 确保至少有两个坐标值
                coords.append((int(float(parts[0])), int(float(parts[1]))))

    patch_height, patch_width = patch_size
    for center in coords:
        # top_left = (center[1] - patch_width // 2, center[0] - patch_height // 2)
        # bottom_right = (center[1] + patch_width // 2, center[0] + patch_height // 2)
        
        if x_flip:
            new_center_x = image_array.shape[0] - center[0]
        else:
            new_center_x = center[0]
        
        if y_flip:
            new_center_y = image_array.shape[1] - center[1]
        else:
            new_center_y = center[1]
        
        new_center = (new_center_x, new_center_y)
        

        # top_left = (new_center[1] - patch_width // 2, new_center[0] - patch_height // 2)
        # bottom_right = (new_center[1] + patch_width // 2, new_center[0] + patch_height // 2)
        
        ## EMPIAR-10164、EMPIAR-10651
        top_left = (new_center[0] - patch_width // 2, new_center[1] - patch_height // 2)
        bottom_right = (new_center[0] + patch_width // 2, new_center[1] + patch_height // 2)
        
        draw.rectangle([top_left, bottom_right], outline="red", width=2)

    # 直接保存，不显示白色背景
    image.save(save_path)
    print(f"图像已保存至: {save_path}")

# EMPIAR-10499
# image_directory = '/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10499/tilt_series_topaz'  # 原始图像目录
# coords_directory = '/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10499/tilt_series_noise_coord'  # 坐标文件目录

# EMPIAR-10164
image_directory = '/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10164/tilt_series_earse'  # 原始图像目录
coords_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPAIR-10164/particle_coords/TS_01.tomostar'  # 坐标文件目录
x_flip = False
y_flip= False

# EMPIAR-10651
image_path = '/data/wxs/tomo/EMPIAR_10651_project/0006/k2dft20s_14apra0006_slices/k2dft20s_14apra0006_slice_011.mrc'  # 原始图像文件
coords_path = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10651/particle_coords/k2dft20s_14apra0006_thin_tilt010_-29.1.coords'  # 坐标文件
x_flip = False
y_flip= True
file_extension = '.mrc'  # 或者根据实际情况设置

# 参数配置
patch_size = (192, 192)  # 与之前一致

# 处理图像和坐标文件
if os.path.isdir(image_path) and os.path.isdir(coords_path):
    # 如果都是目录，则处理目录中的第一个文件
    example_file = sorted(os.listdir(coords_path))[0]
    coords_file = os.path.join(coords_path, example_file)
    
    image_file = os.path.join(image_path, os.path.splitext(example_file)[0] + file_extension)
elif os.path.isfile(image_path) and os.path.isfile(coords_path):
    # 如果都是文件，则直接使用
    image_file = image_path
    coords_file = coords_path
else:
    raise ValueError("图像路径和坐标路径必须同时为目录或同时为文件")

print(f"image_file={image_file}")
print(f"coords_file={coords_file}")

if os.path.exists(image_file):
    draw_patches_on_image(image_file, coords_file, patch_size)
else:
    print(f"对应图像文件不存在: {image_file}")
