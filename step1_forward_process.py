import numpy as np
import mrcfile as mf
import os
from glob import glob
from tqdm import tqdm

import argparse
import random
import mrcfile
from loguru import logger 



def get_patches(patches_path, patch_num):
    """
    Load MRC files from the specified path and return the specified number of patches.
    """
    mrc_files = glob(os.path.join(patches_path, "*.mrc"))

    if patch_num == -1:
        selected_files = mrc_files
    else:
        if len(mrc_files) < patch_num:
            logger.warning(f"Only {len(mrc_files)} MRC files in directory, returning all files.")
            patch_num = len(mrc_files)
            selected_files = random.sample(mrc_files, patch_num)
        else:
            selected_files = random.sample(mrc_files, patch_num)

    mrc_arrays = []
    for mrc_file in tqdm(selected_files, desc="Loading MRC files", ncols=100):
        with mrcfile.open(mrc_file, permissive=True) as mrc:
            mrc_arrays.append(mrc.data)
    
    logger.info(f"Loaded {len(mrc_arrays)} patches.")
    return np.array(mrc_arrays)

def get_diffuse_dataset(
    particle_patches,
    noise_patches,
    particle_num,
    noise_num,
    save_path,
    random_seed=42,
):
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Get particle and noise patches
    particles_patches_array = get_patches(particle_patches, particle_num)
    noise_patches_array = get_patches(noise_patches, noise_num)

    logger.info(f"Particles patch shape: {particles_patches_array.shape}")
    logger.info(f"Noise patch shape: {noise_patches_array.shape}")

    # Randomly select noise data
    if len(particles_patches_array) <= len(noise_patches_array):
        random_index = np.random.choice(len(noise_patches_array), len(particles_patches_array), replace=False)
        noise_patch = noise_patches_array[random_index]
    else:
        times = len(particles_patches_array) // len(noise_patches_array) + 1
        noise_patch = np.tile(noise_patches_array, (times, 1, 1))[:len(particles_patches_array)]
        np.random.shuffle(noise_patch)

    logger.info("Diffusing process started...")

    # Create save directories
    train_path = os.path.join(save_path, "train")
    val_path = os.path.join(save_path, "val")
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    
    logger.info(f"Saving train data to: {train_path}")
    logger.info(f"Saving validation data to: {val_path}")

    inputs = []
    outputs = []
    noises = []
    input_mrc = particles_patches_array 
    for i in range(add_noise_num):
        output_mrc = (1 - alpha * (i + 1)) * input_mrc + (alpha * (i + 1)) * noise_patch
        outputs.append(output_mrc)
        inputs.append(input_mrc)
        noises.append(noise_patch)
        np.random.shuffle(noise_patch)
        input_mrc = output_mrc

    # Stack all data
    all_inputs = np.vstack(inputs)
    all_outputs = np.vstack(outputs)
    all_noises = np.vstack(noises)
    
    # Get total number of samples
    total_samples = len(all_inputs)
    
    # Create indices and shuffle randomly
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    # Split data with 8:2 ratio
    split_idx = int(total_samples * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Training samples: {len(train_indices)} (80%)")
    logger.info(f"Validation samples: {len(val_indices)} (20%)")
    
    # Save training set
    mf.write(
        os.path.join(train_path, "inputs.mrcs"),
        all_inputs[train_indices].astype(np.float32),
        overwrite=True,
    )
    mf.write(
        os.path.join(train_path, "outputs.mrcs"),
        all_outputs[train_indices].astype(np.float32),
        overwrite=True,
    )
    mf.write(
        os.path.join(train_path, "noises.mrcs"),
        all_noises[train_indices].astype(np.float32),
        overwrite=True,
    )
    
    # Save validation set
    mf.write(
        os.path.join(val_path, "inputs.mrcs"),
        all_inputs[val_indices].astype(np.float32),
        overwrite=True,
    )
    mf.write(
        os.path.join(val_path, "outputs.mrcs"),
        all_outputs[val_indices].astype(np.float32),
        overwrite=True,
    )
    mf.write(
        os.path.join(val_path, "noises.mrcs"),
        all_noises[val_indices].astype(np.float32),
        overwrite=True,
    )
    
    logger.info("Data splitting and saving completed successfully.")




if __name__ == "__main__":
    
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    ## EMPIAR-10499
    # alpha = 0.01
    # add_noise_num = 5
    # parser = argparse.ArgumentParser(description="forward process")
    # parser.add_argument(
    #     "--particle_patches",
    #     default="/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10499/particle_patches",
    #     type=str,
    #     help="org_mrc_path",
    # )
    # parser.add_argument(
    #     "--noise_patches",
    #     default="/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10499/noise_patches",
    #     type=str,
    #     help="particles coordinate .star",
    # )
    # parser.add_argument(
    #     "--particle_num", default=4000, type=int, help="particles coordinate .star"
    # )
    # parser.add_argument("--noise_num", default=-1, type=int, help="-1 is all")
    # parser.add_argument(
    #     "--out_path",
    #     type=str,
    #     default=f"/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10499/DDM_data_{alpha}_{add_noise_num}",
    #     help="out_put_path",
    # )
    
    
    ## EMPIAR-10164
    # alpha = 0.01
    # add_noise_num = 5
    
    # parser = argparse.ArgumentParser(description="forward process")
    # parser.add_argument(
    #     "--particle_patches",
    #     default="/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10164/particle_patches",
    #     type=str,
    #     help="org_mrc_path",
    # )
    # parser.add_argument(
    #     "--noise_patches",
    #     default="/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10164/noise_patches",
    #     type=str,
    #     help="particles coordinate .star",
    # )
    # parser.add_argument(
    #     "--particle_num", default=4000, type=int, help="particles coordinate .star"
    # )
    # parser.add_argument("--noise_num", default=-1, type=int, help="-1 is all")
    # parser.add_argument(
    #     "--out_path",
    #     type=str,
    #     default=f"/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10164/DDM_data_{alpha}_{add_noise_num}",
    #     help="out_put_path",
    # )
    
    
    ## EMPIAR-10651
    alpha = 0.05
    add_noise_num = 5
    
    parser = argparse.ArgumentParser(description="forward process")
    parser.add_argument(
        "--particle_patches",
        default="/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10651/particle_patches",
        type=str,   
        help="org_mrc_path",
    )
    parser.add_argument(
        "--noise_patches",
        default="/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10651/noise_patches",
        type=str,
        help="particles coordinate .star",
    )
    parser.add_argument(
        "--particle_num", default=4000, type=int, help="particles coordinate .star"
    )
    parser.add_argument("--noise_num", default=-1, type=int, help="-1 is all")
    parser.add_argument(
        "--out_path",
        type=str,
        default=f"/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10651/DDM_data_{alpha}_{add_noise_num}",
        help="out_put_path",
    )
    

    args = parser.parse_args()

    # Use loguru to log information
    logger.info(f"Starting forward process with parameters: {args}")

    get_diffuse_dataset(
        particle_patches=args.particle_patches,
        noise_patches=args.noise_patches,
        particle_num=args.particle_num,
        noise_num=args.noise_num,
        save_path=args.out_path,
        random_seed=random_seed,
    )
