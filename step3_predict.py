import mrcfile as mf
import torch
import numpy as np
import os
import time
import sys
import random
import argparse
from tqdm import tqdm
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR
import model.UNet2d as unet2d
import script.utils as utils
import copy

from loguru import logger


def normal_batch(mrc_data):
    """Normalize batch of mrc data."""
    return np.array([(img - img.min()) / (img.max() - img.min()) for img in mrc_data], dtype=np.float32)


def normal(mrc_data):
    """Normalize a single mrc image."""
    return (mrc_data - mrc_data.min()) / (mrc_data.max() - mrc_data.min())


def main(test_raw_path, test_out_path, test_model_path, gpu, aim_shape, x, y, data_num):

    
    step = 1
    test_list = utils.get_rawdata_list(test_raw_path)[:data_num] if data_num != -1 else utils.get_rawdata_list(test_raw_path)
    
    padding = 64
    center_shape = [aim_shape - 2 * padding, aim_shape - 2 * padding]
    batch_size = 32
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Load original model and assign its state dict
    model = unet2d.UDenoiseNet().to(device)
    model.load_state_dict(torch.load(test_model_path, map_location=device))
    model.eval()

    
    for file_name in tqdm(test_list, desc="Processing MRC Files"):
        logger.info(f'Reading {file_name}\n')
        mrc_data_np = np.array(mf.read(os.path.join(test_raw_path, file_name)))
        test_data = normal(mrc_data_np)
        
        test_crops, sizes, matches = utils.crop_data(test_data, center_shape, padding=padding, cval=0)
        
        all_output = []
        with torch.no_grad():
            for b in range(0, len(test_crops), batch_size):
                inputs = torch.tensor(test_crops[b:b + batch_size], dtype=torch.float32).unsqueeze(1).to(device)
                for _ in range(step):
                    outputs = model(inputs)
                    inputs = outputs
                all_output.append(outputs.squeeze(1).detach().cpu().numpy())
        
        all_output = np.vstack(all_output)
        mrc_out = utils.concat_data(all_output, sizes, matches, padding)[:y, :x]
        mrc_out = normal(mrc_out)
        
        mf.write(os.path.join(test_out_path, file_name), mrc_out.astype(np.float32), overwrite=True)


if __name__ == '__main__':
    
    ## EMPIAR-10499
    alpha = 0.01
    add_noise_num = 5
    parser = argparse.ArgumentParser(description='Test Denoising Model')
    parser.add_argument('--input_path', '-i', type=str, default='/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10499/tilt_series_earse_all', help='Raw MRC Path')
    # parser.add_argument('--out_path', '-o', type=str, default='/data/wxs/empiar_pipeline/EMPIAR-10499/DDM_0.01_0.1_all/warp_frameseries/average', help='Output Path')
    parser.add_argument('--out_path', '-o', type=str, default='/data/wxs/empiar_pipeline/EMPIAR-10499/tilt_series_ddm_0.01_5/warp_frameseries/average', help='Output Path')
    parser.add_argument('--model_path', '-m', type=str, default='/data/wxs/tomo_denoise/TiltSeriesDDM_v2/save/EMPIAR-10499_0.01_5/best_model.pth', help='Model Path')
    parser.add_argument('--data_num', type=int, default=-1, help='Number of Data Samples (-1 for all)')
    parser.add_argument('--gpus', '-d', type=str, default='2', help='GPU ID')
    parser.add_argument('--lenx', '-x', type=int, default=3838, help='Image Width')
    parser.add_argument('--leny', '-y', type=int, default=3710, help='Image Height')
    args = parser.parse_args()
    
    ## EMPIAR-10164
    # alpha = 0.01
    # add_noise_num = 5
    # parser = argparse.ArgumentParser(description='Test Denoising Model')
    # parser.add_argument('--input_path', '-i', type=str, default='/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10164/tilt_series_earse', help='Raw MRC Path')
    # parser.add_argument('--out_path', '-o', type=str, default=f'/data/wxs/empiar_pipeline/EMPIAR-10164/tilt_series_ddm_0.01_5/warp_frameseries/average/', help='Output Path')
    # # parser.add_argument('--out_path', '-o', type=str, default=f'/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10164/tilt_series_ddm_{alpha}_{add_noise_num}_epoch_8/', help='Output Path')
    # # parser.add_argument('--input_path', '-i', type=str, default='/data/jiazhuo/10164/data/warp_frameseries/average', help='Raw MRC Path')
    # # parser.add_argument('--out_path', '-o', type=str, default=f'/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10164/tilt_series_ddm_{alpha}_ori/', help='Output Path')
    # parser.add_argument('--model_path', '-m', type=str, default=f'/data/wxs/tomo_denoise/TiltSeriesDDM_v2/save/EMPIAR-10164_{alpha}_{add_noise_num}/best_model.pth', help='Model Path')
    # parser.add_argument('--data_num', type=int, default=-1, help='Number of Data Samples (-1 for all)')
    # parser.add_argument('--gpus', '-d', type=str, default='2', help='GPU ID')
    # parser.add_argument('--lenx', '-x', type=int, default=7420, help='Image Width')
    # parser.add_argument('--leny', '-y', type=int, default=7676, help='Image Height')
    # args = parser.parse_args()
    
    ## EMPIAR-10651
    alpha = 0.05
    add_noise_num = 5
    parser = argparse.ArgumentParser(description='Test Denoising Model')
    parser.add_argument('--input_path', '-i', type=str, default='/data/wxs/tomo/EMPIAR_10651_project/slice_all', help='Raw MRC Path')
    parser.add_argument('--out_path', '-o', type=str, default=f'/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10651/tilt_series_ddm_{alpha}_{add_noise_num}/', help='Output Path')
    parser.add_argument('--model_path', '-m', type=str, default=f'/data/wxs/tomo_denoise/TiltSeriesDDM_v2/save/EMPIAR-10651_{alpha}_{add_noise_num}/best_model.pth', help='Model Path')
    parser.add_argument('--data_num', type=int, default=-1, help='Number of Data Samples (-1 for all)')
    parser.add_argument('--gpus', '-d', type=str, default='3', help='GPU ID')
    parser.add_argument('--lenx', '-x', type=int, default=3710, help='Image Width')
    parser.add_argument('--leny', '-y', type=int, default=3710, help='Image Height')
    args = parser.parse_args()
    
    aim_shape = 256  # 192 + 64 (target shape for processing)
    
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    
    main(args.input_path, args.out_path, args.model_path, args.gpus, aim_shape, args.lenx, args.leny, args.data_num)
