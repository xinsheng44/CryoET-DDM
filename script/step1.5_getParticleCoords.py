import numpy as np
import json
import os
import re

from collections import defaultdict


## tomo坐标转换到tilt坐标

class BuildParticleSPA:
    def __init__(self, path_dict) -> None:
        self.path_dict = path_dict
    
    def build(self):
        

        save_path = self.path_dict["save_path"]
        coords = self.read_particles(self.path_dict["particle_path"], self.path_dict["particles_row_start"], self.path_dict["particles_col_n"])
        mdocs,exposureDose = self.read_mdoc(self.path_dict["mdoc_path"], coords.keys(), self.path_dict["extension"])
        projMatrix,tilt_exposure = self.read_tomograms(self.path_dict["tomogram_path"], self.path_dict["tomograms_frame_start"], self.path_dict["tomograms_row_start"])

      
        
        for tomo_name, coord in coords.items():
            tomo_dir = os.path.join(self.path_dict["save_path"], tomo_name)
            os.makedirs(tomo_dir, exist_ok=True)
            for tilt_id, tilt_proj in enumerate(projMatrix[tomo_name]):
                coord_tilt = self.proj_particle_single(coord, tilt_proj)
                
                # 获取tilt名字
                if exposureDose == 0:
                    exposureDose = self.path_dict["ExposureDose"]
                tilt_id = round(tilt_exposure[tomo_name][tilt_id] / exposureDose)
                tilt_temp_path = mdocs[tomo_name][tilt_id]["SubFramePath"]
                tilt_type = self.path_dict["tilt_type"]
                match = re.search(fr'([^\\]+)\.{tilt_type}$', tilt_temp_path)
                tilt_name = match.group(1)
                
                save_path = os.path.join(tomo_dir, f"{tilt_name}.txt")
                with open(save_path, "w") as f:
                    for coo_tilt in coord_tilt:
                        coord_temp = "\t".join([f"{item:10}" for item in coo_tilt])
                        f.write(coord_temp + "\n")

            


    def proj_particle_single(self, coord, proj_matrix):
        # 对单个颗粒三维坐标进行仿射变换,得到tilt上坐标
        single_result = []
        coord_bin = self.path_dict["coord_bin"]
        for coo in coord:
            p4 = np.array([float(coo[0]) * coord_bin, float(coo[1]) * coord_bin, float(coo[2]) * coord_bin, 1])
            pl4 = np.dot(proj_matrix, p4)

            x,y = pl4[0],pl4[1]
            
            if x < 0:
                x += self.path_dict["tilt_size"][0]
            if y < 0:
                y += self.path_dict["tilt_size"][1]
            
            y += self.path_dict["y_shift"]
            x += self.path_dict["x_shift"]
          
            single_result.append([x,y])

        return single_result


    def read_particles(self, path, row_start=63, col_n=2):
        
        particle_coords = defaultdict(list)
        
        with open(path, "r") as file:
            lines = file.read().split("\n")

        for line in lines[row_start:]:
            if line.strip() == "" or line == "\n":continue
            line_list = line.split()
            particle_coords[line_list[0]].append(line_list[col_n:col_n+3])

        return particle_coords
    
    def read_mdoc(self, mdoc_dir, tomo_names, extension=".mdoc"):
        """
        {"tomo_id":{
            "ZValue(value)": {"TiltAngle":...,} 
            }
        }
        """
        
        mdocs_dict = {}
        
        for tomo_name in tomo_names:
            t_name, t_extension = os.path.splitext(tomo_name)
            new_name = tomo_name.replace(t_extension, extension)
            file_path = os.path.join(mdoc_dir, new_name)
            
            mdocs_dict[tomo_name] = {}
            
            current_zvalue = None
            tilt_info = {}

            with open(file_path, "r") as file:
                for line in file:
                    line = line.strip()

                    # 跳过空行
                    if not line:
                        continue

                    # 检测 ZValue 开头
                    if line.startswith("[ZValue"):
                        # 如果当前有未保存的 tilt_info，则保存到字典
                        if current_zvalue is not None:
                            mdocs_dict[tomo_name][current_zvalue] = tilt_info

                        # 提取新的 ZValue
                        current_zvalue = int(line.replace("[","").replace("]", "").split("=")[1].strip())
                        tilt_info = {}  # 初始化新的 tilt 信息字典

                    # 如果是普通键值对
                    elif "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()

                        # 转换为合适的数据类型
                        try:
                            if "." in value:  # 浮点数
                                value = float(value)
                            else:  # 整数
                                value = int(value)
                        except ValueError:
                            pass  # 如果无法转换，保留为字符串

                        tilt_info[key] = value

                # 保存最后一个 ZValue 的数据
                if current_zvalue is not None:
                    mdocs_dict[tomo_name][current_zvalue] = tilt_info
                exposureDose = mdocs_dict[tomo_name][current_zvalue]["ExposureDose"]

        return mdocs_dict,exposureDose
        

    def read_tomograms(self, path, frame_start=17, row_start=40):
        
        with open(path, "r") as file:
            lines = file.read().split('\n')
            
        tomo_names = []
        tomogram_xfs = defaultdict(list)
        tilt_exposure = defaultdict(list)

        # Step 1: Extract tomogram names
        for line in lines[frame_start:]:
            if line.strip() == "":break
            line_list = line.split()
            tomo_names.append(line_list[0])
        
        
        # Step 2: Extract affine matrices
        
        row_idx = row_start
        tomo_id = 0
        while row_idx < len(lines):
            if lines[row_idx] == "_rlnMicrographPreExposure #9":
                for temp_idx,temp_line in enumerate(lines[row_idx+1:]):
                    if temp_line.strip() == "":break
                    line_values = temp_line.split()
                    matrix = [eval(val) for val in line_values[:4]]  # Convert to float
                    matrix = np.array(matrix)
                    assert matrix.shape == (4,4)
                    
                    tomogram_xfs[tomo_names[tomo_id]].append(matrix)
                    tilt_exposure[tomo_names[tomo_id]].append(float(line_values[-1]))
                    
                tomo_id += 1
                row_idx += temp_idx
            else:
                row_idx += 1
                
        assert len(tomo_names) == len(tomogram_xfs)
        
        
        return tomogram_xfs,tilt_exposure
        



if __name__ == "__main__": 
      
    ## EMPIAR-10499
    # path_dict = {
    #     "save_path" : "./data/EMPIAR-10499/particle_coords",
    #     "particle_path" : "/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10499/coord_tomograms/bin1_particles.star",
    #     "tomogram_path" : "/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10499/coord_tomograms/bin1_tomograms.star",
    #     "mdoc_path" : "/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10499/mdoc",
    #     "extension" : ".mdoc",
    #     "tomograms_frame_start" : 17,
    #     "tomograms_row_start" : 40,
    #     "particles_row_star":63,
    #     "particles_col_n" : 2,
    #     "tilt_type" : "tif",
    #     "coord_bin" : 1
    # }


    path_dict = {
        "save_path" : "/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPAIR-10164/particle_coords",
        "particle_path" : "/data/jiazhuo/10164/data/relion/Select/job011/particles.star",
        "tomogram_path" : "/data/jiazhuo/10164/data/relion/pix1.35_2d_tomograms.star",
        "mdoc_path" : "/data/jiazhuo/10164/data/mdoc",
        "extension" : ".mrc.mdoc",
        "tomograms_frame_start" : 17,
        "tomograms_row_start" : 26,
        "particles_row_start":58,
        "particles_col_n":2,
        "ExposureDose": 3.4,
        "tilt_type" : "mrc",
        "coord_bin" : 2,
        "tilt_size" : (7420,7676),
        "y_shift" : -480,
        "x_shift" : -50
        

    }
    

    buildSPA = BuildParticleSPA(path_dict)
    buildSPA.build()

"""
point2model -input /data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPAIR-10164/particle_coords/TS_01.tomostar/TS_01_000_0.0.txt output TS_01_000_0.0.mod -scat -sp 4 -co 255,0,0 -w 2
3dmod /data/jiazhuo/10164/data3/warp_frameseries/average/even/TS_01_000_0.0.mrc /data/wxs/tomo_denoise/TiltSeriesDDM_v2/TS_01_000_0.0.mod
3dmod /data/jiazhuo/10164/data3/warp_frameseries/average/even/TS_03_000_0.0.mrc /data/wxs/tomo_denoise/TiltSeriesDDM_v2/TS_03_000_0.0.mod
3dmod /data/jiazhuo/10164/data3/warp_frameseries/average/even/TS_45_000_-0.0.mrc /data/wxs/tomo_denoise/TiltSeriesDDM_v2/TS_45_000_0.0.mod
3dmod /data/jiazhuo/10164/data3/warp_frameseries/average/even/TS_54_000_-0.0.mrc /data/wxs/tomo_denoise/TiltSeriesDDM_v2/TS_54_000_0.0.mod

"""