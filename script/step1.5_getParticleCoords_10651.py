import numpy as np
import os
import struct
import argparse
from pathlib import Path
import math

# 全局路径设置
DATA_ROOT = "/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10651"
COORD_PATH = f"{DATA_ROOT}/coord_tomograms/k2dft20s_14apra0006_thin.coords"
TILT_PATH = f"{DATA_ROOT}/coord_tomograms/k2dft20s_14apra0006.tlt"
XF_PATH = None
ALI_PATH = f"{DATA_ROOT}/coord_tomograms/k2dft20s_14apra0006_thin_xf.par"
OUTPUT_PATH = f"{DATA_ROOT}/particle_coords/"

def read_coords(coords_file):
    """读取颗粒的三维坐标文件"""
    coords = np.loadtxt(coords_file)
    return coords

def read_tlt_file(tlt_file):
    """读取倾斜角度文件"""
    with open(tlt_file, 'r') as f:
        tilt_angles = [float(line.strip()) for line in f if line.strip()]
    return np.array(tilt_angles)

def read_xf_file(xf_file):
    """读取对齐变换矩阵文件"""
    with open(xf_file, 'r') as f:
        xf_data = []
        for line in f:
            if line.strip():
                values = [float(x) for x in line.strip().split()]
                if len(values) == 6:
                    xf_data.append(values)
    return np.array(xf_data)

def read_ali_file(ali_file, save_txt=True):
    """读取二进制对齐文件，并可选择保存为文本文件
    
    参数:
        ali_file: 对齐文件路径
        save_txt: 是否保存为文本文件
    """
    with open(ali_file, 'rb') as f:
        data = f.read()
    
    # 读取帧数（第一个 int32）
    n_frames = struct.unpack("<i", data[:4])[0]
    print(f"检测到 {n_frames} 个倾斜图像帧")
    
    # 计算每帧大小，分帧
    total_size = len(data)
    bytes_per_frame = total_size // n_frames
    frames = [data[i * bytes_per_frame:(i + 1) * bytes_per_frame] for i in range(n_frames)]
    
    # 提取每帧的参数
    alignment_params = []
    all_params = []
    
    for idx, frame in enumerate(frames):
        # 解码前 80 字节为 float32（20 个值）
        f32_vals = struct.unpack("<20f", frame[:80])
        all_params.append(f32_vals)
        
        angle_deg = f32_vals[12]  # 角度（degrees）
        shift_x = f32_vals[9]     # X方向平移
        shift_y = f32_vals[10]    # Y方向平移
        
        # 生成旋转矩阵
        theta = np.deg2rad(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # IMOD xf 格式：a11 a12 a21 a22 dx dy
        xf_params = [cos_t, -sin_t, sin_t, cos_t, shift_x, shift_y]
        alignment_params.append(xf_params)
    
    # 如果需要，保存所有参数为文本文件
    if save_txt:
        txt_file = ali_file + '.txt'
        with open(txt_file, 'w') as f_out:
            f_out.write(f"# 总帧数: {n_frames}\n")
            f_out.write(f"# 每帧字节数: {bytes_per_frame}\n")
            f_out.write("# 格式: [frame_index] [param_1] [param_2] ... [param_20]\n")
            f_out.write("#\n")
            
            for i, params in enumerate(all_params):
                f_out.write(f"{i}")
                for val in params:
                    f_out.write(f" {val}")
                f_out.write("\n")
        print(f"参数已保存为文本文件: {txt_file}")
    
    return np.array(alignment_params), n_frames

def save_params_to_txt(params, output_file):
    """将参数数组保存为文本文件"""
    n_tilts, n_params = params.shape
    
    with open(output_file, 'w') as f:
        # 写入文件头
        f.write(f"# 倾斜图像数量: {n_tilts}\n")
        f.write(f"# 每个图像的参数数量: {n_params}\n")
        f.write("# 格式: [tilt_index] [param_1] [param_2] ... [param_n]\n")
        f.write("#\n")
        
        # 写入参数
        for i in range(n_tilts):
            f.write(f"{i}")
            for j in range(n_params):
                f.write(f" {params[i, j]}")
            f.write("\n")

def extract_alignment_params(tilt_params):
    """
    从对齐参数中提取有用的变换信息
    
    这里我们假设前9个参数可能是3x3旋转矩阵，接下来3个参数是平移向量
    实际情况可能需要根据具体文件格式调整
    """
    n_tilts = tilt_params.shape[0]
    
    # 提取可能有用的参数
    # 我们尝试提取前12个参数，假设它们包含旋转和平移信息
    useful_params = []
    
    for i in range(n_tilts):
        params = tilt_params[i]
        
        # 检查参数是否有效（不是NaN或无穷大）
        valid_params = []
        for p in params[:12]:  # 只看前12个参数
            if math.isfinite(p) and abs(p) < 1e10:  # 排除极端值
                valid_params.append(p)
            else:
                valid_params.append(0.0)  # 无效值替换为0
        
        # 如果有足够的有效参数，构建变换矩阵
        if len(valid_params) >= 6:
            # 尝试构建一个简化的变换矩阵 [a11 a12 a21 a22 dx dy]
            # 这类似于.xf文件的格式
            transform = valid_params[:6]
            useful_params.append(transform)
        else:
            # 如果没有足够的有效参数，使用单位变换
            useful_params.append([1, 0, 0, 1, 0, 0])
    
    return np.array(useful_params)

def project_particles(coords_3d, tilt_angles, alignment_params=None, tilt_axis='y'):
    """
    将3D颗粒坐标投影到每个倾斜角度的2D平面上
    
    参数:
    coords_3d: 颗粒的3D坐标 (N, 3)
    tilt_angles: 倾斜角度列表
    alignment_params: 对齐参数，格式为 [a11 a12 a21 a22 dx dy]
    tilt_axis: 倾斜轴方向，可以是 'x', 'y', 或 'z'
    
    返回:
    projected_coords: 每个倾斜角度下的2D坐标 (n_tilts, n_particles, 2)
    """
    n_particles = coords_3d.shape[0]
    n_tilts = len(tilt_angles)
    projected_coords = np.zeros((n_tilts, n_particles, 2))
    
    for i, tilt_angle in enumerate(tilt_angles):
        # 将角度转换为弧度
        theta = np.radians(tilt_angle)
        
        # 根据倾斜轴选择旋转矩阵
        if tilt_axis == 'y':
            # 绕Y轴旋转
            rotation_matrix = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif tilt_axis == 'x':
            # 绕X轴旋转
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
        elif tilt_axis == 'z':
            # 绕Z轴旋转
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        
        # 对每个颗粒进行投影
        for j, coord in enumerate(coords_3d):
            # 应用旋转
            rotated_coord = rotation_matrix @ coord
            
            # 投影到XY平面 (保留X和Y坐标)
            projected_coord = rotated_coord[:2]
            
            # 如果有对齐参数，应用它们
            if alignment_params is not None:
                xf = alignment_params[i]
                # 变换矩阵格式: [a11 a12 a21 a22 dx dy]
                transformed_x = xf[0] * projected_coord[0] + xf[1] * projected_coord[1] + xf[4]
                transformed_y = xf[2] * projected_coord[0] + xf[3] * projected_coord[1] + xf[5]
                projected_coord = np.array([transformed_x, transformed_y])
            
            projected_coords[i, j] = projected_coord
    
    return projected_coords

def main():
    parser = argparse.ArgumentParser(description='计算颗粒在每个倾斜图像上的坐标')
    parser.add_argument('--coords', type=str, default=COORD_PATH,
                        help=f'3D颗粒坐标文件路径 (默认: {COORD_PATH})')
    parser.add_argument('--tlt', type=str, default=TILT_PATH,
                        help=f'倾斜角度文件路径 (默认: {TILT_PATH})')
    parser.add_argument('--xf', type=str, default=XF_PATH,
                        help=f'对齐变换矩阵文件路径 (默认: {XF_PATH})')
    parser.add_argument('--ali', type=str, default=ALI_PATH,
                        help=f'二进制对齐文件路径 (默认: {ALI_PATH})')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH,
                        help=f'输出目录 (默认: {OUTPUT_PATH})')
    parser.add_argument('--tilt-axis', type=str, default='y', choices=['x', 'y', 'z'], 
                        help='倾斜轴方向 (默认: y)')
    parser.add_argument('--save-params', action='store_true', default=True,
                        help='将对齐参数保存为文本文件')
    
    args = parser.parse_args()
    
    # 读取3D坐标
    coords_3d = read_coords(args.coords)
    print(f"读取了 {coords_3d.shape[0]} 个颗粒的3D坐标")
    
    # 读取倾斜角度
    tilt_angles = read_tlt_file(args.tlt)
    print(f"读取了 {len(tilt_angles)} 个倾斜角度")
    
    # 获取对齐参数
    alignment_params = None
    
    # 优先使用.xf文件
    if args.xf is not None and os.path.exists(args.xf):
        alignment_params = read_xf_file(args.xf)
        print(f"从.xf文件读取了 {len(alignment_params)} 个对齐变换矩阵")
    
    # 如果没有.xf文件但有.ali文件，从.ali文件提取参数
    elif os.path.exists(args.ali):
        alignment_params, n_frames = read_ali_file(args.ali, save_txt=args.save_params)
        print(f"从.ali文件读取了 {n_frames} 个对齐变换矩阵")
        
        # 如果需要，保存提取的对齐参数为xf格式
        if args.save_params:
            xf_file = args.ali + '.xf'
            with open(xf_file, 'w') as f:
                for params in alignment_params:
                    f.write(f"{params[0]} {params[1]} {params[4]}\n")
                    f.write(f"{params[2]} {params[3]} {params[5]}\n")
            print(f"对齐参数已保存为XF格式: {xf_file}")
    
    # 计算投影坐标
    projected_coords = project_particles(coords_3d, tilt_angles, alignment_params, args.tilt_axis)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 保存每个倾斜角度的坐标，排除负值坐标
    base_name = Path(args.coords).stem
    for i, angle in enumerate(tilt_angles):
        # 筛选出坐标值均为正的点
        coords_2d = projected_coords[i]
        positive_coords = coords_2d[(coords_2d[:, 0] >= 0) & (coords_2d[:, 1] >= 0)]
        
        output_file = os.path.join(args.output, f"{base_name}_tilt{i:03d}_{angle:.1f}.coords")
        np.savetxt(output_file, positive_coords, fmt='%.6f')
        
        # 输出筛选信息
        filtered_count = coords_2d.shape[0] - positive_coords.shape[0]
        if filtered_count > 0:
            print(f"倾斜角 {angle:.1f}°: 排除了 {filtered_count} 个负坐标点，保留了 {positive_coords.shape[0]} 个点")
    
    print(f"已将投影坐标保存到 {args.output} 目录")

if __name__ == "__main__":
    main()


"""

tomoalign \
  -a k2dft20s_14apra0006.tlt \
  -i k2dft20s_14apra0006_ali.fid.txt \
  -o k2dft20s_14apra0006_thin_xf.par \
  -t thin \
  -3 \
  -P

point2model -input k2dft20s_14apra0006_thin_tilt020_-3.0.coords output k2dft20s_14apra0006_thin_tilt020_-3.0.mod -scat -sp 10 -co 255,0,0 -w 2
"""