import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def process_directory(input_dir, output_dir, patch_size, overlap, noise_tag=0, y_flip=False, threshold=1.0):
    """
    读取目录下的所有图像文件，将图像切割为patch，并提取符合噪声比例阈值的patch，
    统计每个图像中符合要求的patch数量，以及总的patch数量和平均每个图像的patch数量。

    :param input_dir: 输入目录，包含图像文件
    :param output_dir: 输出目录，用于保存txt文件
    :param patch_size: patch的大小 (height, width)
    :param overlap: patch之间的重叠量 (height_overlap, width_overlap)
    :param noise_tag: 噪声区域的像素值
    :param y_flip: 是否翻转y坐标
    :param threshold: 判断patch有效的噪声像素比例阈值 (0.0-1.0)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入目录中的所有文件
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # 定义patch大小和步长
    patch_height, patch_width = patch_size
    step_height = patch_height - overlap[0]
    step_width = patch_width - overlap[1]

    total_patches = 0
    image_patch_counts = {}

    for file in tqdm(files, desc="Processing files"):
        try:
            # 读取图像
            image_path = os.path.join(input_dir, file)
            image = Image.open(image_path).convert('L')  # 转换为灰度图
            image_array = np.array(image)

            # 切割图像并提取patch
            valid_patches = []
            for i in range(0, image_array.shape[0] - patch_height + 1, step_height):
                for j in range(0, image_array.shape[1] - patch_width + 1, step_width):
                    patch = image_array[i:i + patch_height, j:j + patch_width]
                    # 计算patch中噪声像素的比例
                    noise_ratio = np.sum(patch == noise_tag) / (patch_height * patch_width)
                    if noise_ratio >= threshold:  # 检查噪声像素比例是否达到阈值
                        if y_flip:
                            center = (i + patch_height // 2, image_array.shape[1] - (j + patch_width // 2))
                        else:
                            center = (i + patch_height // 2, j + patch_width // 2)
                        valid_patches.append(center)

            # 去重并保存
            unique_patch_centers = list(set(valid_patches))
            output_file = os.path.join(output_dir, os.path.splitext(file)[0] + '.txt')

            with open(output_file, 'w') as f:
                for center in unique_patch_centers:
                    f.write(f"{center[0]:>6}\t{center[1]:>6}\n")

            # 统计patch数量
            image_patch_counts[file] = len(unique_patch_centers)
            total_patches += len(unique_patch_centers)

        except Exception as e:
            print(f"处理失败: {file}, 错误: {e}")

    # 统计总数和平均数
    average_patches = total_patches / len(image_patch_counts) if image_patch_counts else 0
    print(f"\n总符合要求的patch数量: {total_patches}")
    print(f"平均每个图像的patch数量: {average_patches}")

    return total_patches, average_patches, image_patch_counts

# EMPIAR-10499
# input_directory = '/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10499/tilt_series_mask_v1'  # 替换为你的输入目录
# output_directory = './data/EMPIAR-10499/noise_coords'  # 替换为你的输出目录
# patch_size = (192, 192)  # 指定patch大小
# overlap = (0, 0)  # 指定重叠大小

# EMPIAR-10164
# input_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPAIR-10164/noise_mask'  # 替换为你的输入目录
# output_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPAIR-10164/noise_coords'  # 替换为你的输出目录
# patch_size = (192, 192)  # 指定patch大小
# overlap = (0, 0)  # 指定重叠大小
# noise_tag = 255
# y_flip = False #由于图像坐标系不同吗，根据实际需要转换y

# EMPAIR-10651
input_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10651/noise_mask'  # 替换为你的输入目录
output_directory = '/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10651/noise_coords'  # 替换为你的输出目录
patch_size = (192, 192)  # 指定patch大小
overlap = (156, 156)  # 指定重叠大小
noise_tag = 255
y_flip = False #由于图像坐标系不同吗，根据实际需要转换y
threshold = 0.90  # 设置噪声像素比例阈值，例如95%

total_patches, average_patches, image_patch_counts = process_directory(input_directory, output_directory, patch_size, overlap, noise_tag=noise_tag, y_flip=y_flip, threshold=threshold)
