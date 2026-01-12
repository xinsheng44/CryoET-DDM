import cv2
import numpy as np
import glob
import os
from pathlib import Path
import mrcfile

def check_background_pixel(img, x, y, target_base=0.368):
    """
    检查当前像素及其周围相邻像素的千分位是否都匹配目标值
    现在只检查到千分位，即0.368*都算匹配
    """
    h, w = img.shape[:2]
    
    # 定义8个相邻像素的偏移量
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # 检查当前像素（千分位匹配）
    current_value = round(img[y, x], 3)  # 四舍五入到千分位
    if abs(current_value - target_base) >= 0.001:
        return False
    
    # 检查相邻像素
    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        # 边界检查 - 确保不越界
        if 0 <= nx < w and 0 <= ny < h:
            neighbor_value = round(img[ny, nx], 3)  # 四舍五入到千分位
            if abs(neighbor_value - target_base) >= 0.001:
                return False
    
    return True

def read_mrc_image(mrc_path):
    """
    读取mrc文件
    """
    try:
        with mrcfile.open(mrc_path, mode='r') as mrc:
            # 获取图像数据
            data = mrc.data
            
            # 如果是3D数据，取第一个切片
            if len(data.shape) == 3:
                img = data[0]  # 取第一个z切片
                print(f"3D mrc文件，取第一个切片，原始形状: {data.shape}")
            else:
                img = data
            
            # 转换为float32类型
            img = img.astype(np.float32)
            
            print(f"mrc图像数据范围: {img.min()} - {img.max()}")
            return img
            
    except Exception as e:
        print(f"读取mrc文件失败: {e}")
        return None

def generate_mask(img_path, target_base=0.368):
    """
    为单个图像生成mask
    """
    # 读取mrc图像
    img = read_mrc_image(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return None
    
    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    print(f"正在处理图像: {os.path.basename(img_path)}, 尺寸: {w}x{h}")
    print(f"使用目标值: {target_base}")
    
    # 遍历每个像素
    for y in range(h):
        if y % 100 == 0:  # 每100行打印一次进度
            print(f"处理进度: {y}/{h} ({100*y/h:.1f}%)")
        
        for x in range(w):
            if check_background_pixel(img, x, y, target_base):
                mask[y, x] = 0  # 背景
            else:
                mask[y, x] = 255  # 前景
    
    return mask

def process_mrc_images(input_folder, output_folder=None):
    """
    处理文件夹下所有mrc图像，为三张图分别使用不同的目标值
    """
    if output_folder is None:
        output_folder = os.path.join(input_folder, "masks")
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 查找所有mrc文件
    mrc_patterns = [
        os.path.join(input_folder, "k2dft20s_*_slice_021.mrc")
    ]
    
    mrc_files = []
    for pattern in mrc_patterns:
        mrc_files.extend(glob.glob(pattern))
    
    if not mrc_files:
        print(f"在文件夹 {input_folder} 中未找到mrc图像文件")
        return
    
    # 对文件进行排序
    mrc_files = sorted(mrc_files)
    print(f"找到 {len(mrc_files)} 个mrc文件（已排序）")
    
    # 定义三张图对应的目标值
    target_values = [0.368, 0.388, 0.295]
    
    # 确保只处理前3张图像
    num_files_to_process = min(len(mrc_files), 3)
    
    # 处理每个图像
    for i in range(num_files_to_process):
        mrc_path = mrc_files[i]
        target_base = target_values[i]
        
        print(f"\n处理第 {i+1}/{num_files_to_process} 个文件")
        print(f"文件: {os.path.basename(mrc_path)}")
        print(f"目标值: {target_base}")
        
        # 生成mask
        mask = generate_mask(mrc_path, target_base)
        
        if mask is not None:
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(mrc_path))[0]
            output_path = os.path.join(output_folder, f"{base_name}_mask_{target_base}.png")
            
            # 保存mask
            cv2.imwrite(output_path, mask)
            print(f"mask已保存到: {output_path}")
        else:
            print(f"处理失败: {mrc_path}")
    
    if len(mrc_files) > 3:
        print(f"\n注意: 找到{len(mrc_files)}个文件，但只处理了前3个")
    
    print(f"\n所有处理完成！mask保存在: {output_folder}")

if __name__ == "__main__":
    # 设置输入文件夹路径
    input_folder = "/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10651/tilt_series_ddm_0.01_5_epoch_20"
    output_folder = "/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10651/masks"
    
    print(f"\n开始处理...")
    print(f"输入文件夹: {input_folder}")
    print(f"目标像素值: [0.368, 0.388, 0.295] (对应sorted后的前3张图)")
    print(f"输出文件夹: {output_folder or os.path.join(input_folder, 'masks')}")
    
    # 处理图像
    process_mrc_images(input_folder, output_folder)
