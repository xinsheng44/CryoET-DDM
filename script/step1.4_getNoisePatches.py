import os
import mrcfile
import numpy as np
from PIL import Image

# 函数：在图像上绘制patch方框并提取噪声块
def extract_and_save_patches(image_path, coords_path, patch_size, save_dir, x_flip=False):
    """
    根据坐标文件从图像中提取patch，并保存为新的mrc文件。

    :param image_path: 原始图像路径 (MRC文件或TIF文件)
    :param coords_path: 坐标文件路径
    :param patch_size: patch大小 (height, width)
    :param save_dir: 保存提取的patch的目录
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

    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    
    # 读取坐标
    with open(coords_path, 'r') as f:
        coords = [line.strip().split('\t') for line in f]
        coords = [(int(c[0]), int(c[1])) for c in coords]

    # 获取patch大小
    patch_height, patch_width = patch_size

    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    # 遍历坐标，提取patch并保存为MRC文件
    for idx, center in enumerate(coords):
        
        if x_flip:
            new_center = (image_array.shape[0] - center[0], center[1])
        else:
            new_center = center
        
        top_left = (new_center[1] - patch_width // 2, new_center[0] - patch_height // 2)
        bottom_right = (new_center[1] + patch_width // 2, new_center[0] + patch_height // 2)

        # 提取patch
        patch = image_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # 保存为新的MRC文件
        patch_filename = f"{os.path.basename(coords_path).replace('.txt', '')}_{idx+1:03}.mrc"
        patch_filepath = os.path.join(save_dir, patch_filename)

        # 将数据转换为float32类型，这是MRC文件支持的类型
        patch = patch.astype(np.float32)

        with mrcfile.new(patch_filepath, overwrite=True) as mrc_out:
            mrc_out.set_data(patch)

        print(f"保存了噪声块 {patch_filename} 到 {save_dir}")
        count +=1
    
    return count

# EMPIAR-10499
# image_directory = '/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10499/tilt_series_earse'  # 原始图像目录
# coords_directory = './data/EMPIAR-10499/noise_coords'  # 坐标文件目录
# save_directory = './data/EMPIAR-10499/noise_patches'  # 保存噪声块的目录

# EMPIAR-10164
# image_directory = '/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10164/tilt_series_earse'  # 原始图像目录
# coords_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPAIR-10164/noise_coords'  # 坐标文件目录
# save_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPAIR-10164/noise_patches'  # 保存噪声块的目录
# x_flip = True


# EMPIAR-10651
image_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10651/tilt'  # 原始图像目录
coords_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10651/noise_coords'  # 坐标文件目录
save_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10651/noise_patches'  # 保存噪声块的目录
x_flip = False
file_extension = '.mrc'

# 参数配置
patch_size = (192, 192)  # 与之前一致

count_num = 0
# 遍历coords目录中的所有文件并处理
for coords_file in os.listdir(coords_directory):
    coords_path = os.path.join(coords_directory, coords_file)
    
    # 获取对应的MRC图像文件
    image_file = os.path.join(image_directory, os.path.splitext(coords_file)[0] + file_extension)
    
    if os.path.exists(image_file):
        print(f"正在处理文件: {coords_file}")
        count = extract_and_save_patches(image_file, coords_path, patch_size, save_directory, x_flip=x_flip)
        count_num += count
    else:
        print(f"对应的图像文件不存在: {image_file}")

print(f"共保存噪声块{count_num}个")
