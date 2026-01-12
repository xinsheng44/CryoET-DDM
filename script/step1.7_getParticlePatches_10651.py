import os
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from glob import glob
from tqdm import tqdm

# 函数：从图像上裁剪patch块并保存为MRC文件
def extract_and_save_patches(image_path, coords_path, patch_size, save_dir, mask_path=None, exclude_tag=None):
    """
    根据坐标文件从图像中提取patch，并保存为新的mrc文件。
    
    :param image_path: 原始图像路径 (MRC文件)
    :param coords_path: 坐标文件路径
    :param patch_size: patch大小 (height, width)
    :param save_dir: 保存提取的patch的目录
    :param noise_mask_dir: 噪声掩码目录路径
    :param exclude_tag: 排除标记的像素值
    """
    # 读取MRC图像
    with mrcfile.open(image_path, permissive=True) as mrc:
        image_array = mrc.data
        
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

    # 如果提供了噪声掩码目录，则读取对应的掩码图像
    mask_array = None
    if mask_path and exclude_tag:
        if os.path.exists(mask_path):
            # 使用PIL或skimage读取TIF文件
            mask_array = np.array(Image.open(mask_path))
        else:
            print(f"警告：未找到对应的掩码文件: {mask_path}")

    # 读取坐标
    with open(coords_path, 'r') as f:
        coords = [line.strip().split() for line in f]
        coords = [(int(float(c[0])), int(float(c[1]))) for c in coords]  # 转换为整数坐标对

    # 获取patch大小
    patch_height, patch_width = patch_size

    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 统计符合要求的颗粒数
    valid_particles = 0

    # 遍历坐标，提取patch并保存为MRC文件
    for idx, center in tqdm(enumerate(coords), ncols=80):
        
        if x_flip:
            new_center_x = image_array.shape[0] - center[0]
        else:
            new_center_x = center[0]
        
        if y_flip:
            new_center_y = image_array.shape[1] - center[1]
        else:
            new_center_y = center[1]
        
        new_center = (new_center_x, new_center_y)
        
        
        top_left = (new_center[1] - patch_width // 2, new_center[0] - patch_height // 2)
        bottom_right = (new_center[1] + patch_width // 2, new_center[0] + patch_height // 2)

        # 确保切割范围不会超出图像边界
        top_left = (max(top_left[0], 0), max(top_left[1], 0))
        bottom_right = (min(bottom_right[0], image_array.shape[1]), min(bottom_right[1], image_array.shape[0]))

        # 提取patch
        patch = image_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # 检查patch是否符合要求的大小
        if patch.shape != (patch_height, patch_width):
            continue  # 跳过不符合大小要求的patch

        # 检查patch是否包含噪声掩码中的排除标记
        if mask_array is not None and exclude_tag is not None:
            mask_patch = mask_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            # 检查掩码区域是否包含排除标记
            if np.any(mask_patch == exclude_tag):
                continue  # 跳过包含排除标记的patch

        # 如果符合要求，统计颗粒数
        valid_particles += 1

        # 保存为新的MRC文件
        patch_filename = f"{os.path.basename(coords_path).replace('.txt', '')}_{idx+1:03}.mrc"
        patch_filepath = os.path.join(save_dir, patch_filename)

        with mrcfile.new(patch_filepath, overwrite=True) as mrc_out:
            mrc_out.set_data(patch)

    return valid_particles  # 返回符合要求的颗粒数

# 示例调用
# image_directory = '/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10499/tilt_series_earse'  # 原始图像目录
# coords_directory = './data/EMPIAR-10499/particle_coords'  # 坐标文件目录
# save_directory = './data/EMPIAR-10499/particle_patches'  # 保存噪声块的目录


# image_directory = '/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10164/tilt_series_earse'  # 原始图像目录
# coords_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10164/particle_coords'  # 坐标文件目录
# save_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10164/particle_patches'  # 保存噪声块的目录

# EMPIAR-10651
image_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10651/tilt'  # 原始图像目录
coords_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10651/particle_coords'  # 坐标文件目录
save_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10651/particle_patches'  # 保存噪声块的目录
x_flip = False
y_flip = True
noise_mask = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10651/noise_mask'
exclude_tag = (0,0,255)


# 参数配置
patch_size = (192, 192)  # 与之前一致

# 总颗粒数
total_particles = 0
valid_particles_total = 0  # 统计总的符合要求的颗粒数

# 遍历coords目录中的所有文件并处理

for coords_path, image_path, mask_path in zip(sorted(glob(os.path.join(coords_directory, "*"))), \
                                                sorted(glob(os.path.join(image_directory, "*"))), \
                                                sorted(glob(os.path.join(noise_mask, "*")))):
    
        # 统计当前 coords 文件的颗粒数
    with open(coords_path, 'r') as f:
        num_particles = len(f.readlines())
    total_particles += num_particles
    
    # 处理对应的MRC图像
    if os.path.exists(image_path):
        print(f"  正在处理文件: {coords_path}")
        valid_particles_in_file = extract_and_save_patches(image_path, coords_path, patch_size, save_directory, mask_path, exclude_tag)
        valid_particles_total += valid_particles_in_file  # 累加符合要求的颗粒数
    else:
        print(f"  对应的图像文件不存在: {image_path}")


# 输出总颗粒数和符合要求的颗粒数
print(f"\n总颗粒数: {total_particles}")
print(f"总符合要求的颗粒数: {valid_particles_total}")
