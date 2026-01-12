import os
import mrcfile
import numpy as np

def convert_mrc_to_float32(input_mrc_path, output_mrc_path):
    """
    读取 2D MRC 文件，转换为 float32 并保存。

    :param input_mrc_path: 输入的 MRC 文件路径
    :param output_mrc_path: 输出的 MRC 文件路径（float32 格式）
    """
    # 读取 MRC 文件
    with mrcfile.open(input_mrc_path, permissive=True) as mrc:
        image_data = mrc.data  # 读取 MRC 数据
    
    # 确保数据是 2D
    if image_data.ndim != 2:
        raise ValueError(f"输入 MRC 不是 2D，当前形状: {image_data.shape}")

    # 转换为 float32
    image_data_float32 = image_data.astype(np.float32)

    # 保存为新的 MRC 文件
    with mrcfile.new(output_mrc_path, overwrite=True) as mrc_out:
        mrc_out.set_data(image_data_float32)
        mrc_out.header.dmean = image_data_float32.mean()  # 更新头信息
        mrc_out.header.dmax = image_data_float32.max()
        mrc_out.header.dmin = image_data_float32.min()
        mrc_out.update_header_from_data()

    print(f"✅ 2D MRC 转换完成！已保存为 float32: {output_mrc_path}")

# 示例：转换一个 MRC 文件
input_mrc = "/data/jiazhuo/10164/data/warp_frameseries/average/odd/TS_01_000_0.0.mrc"  # 替换为你的 MRC 文件路径
output_mrc = "./TS_01_000_0.0_float32.mrc"  # 设置输出路径

convert_mrc_to_float32(input_mrc, output_mrc)
