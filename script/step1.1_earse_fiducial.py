import mrcfile
import torch
import os
from package.fidder.predict import predict_fiducial_mask
from package.fidder.erase import erase_masked_region

from glob import glob
from tqdm import tqdm

## EMPIAR-10499
# base_dir = "/data/ts/data/EMPIAR-10499/data/warp_frameseries/average"
# save_dir = "/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10499/tilt_series_earse"
# pixel_size = 1.701

## EMPIAR-10164
base_dir = '/data/jiazhuo/10164/data/warp_frameseries/average'
save_dir = '/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10164/tilt_series_earse'
pixel_size = 0.675
model_path = "/data/wxs/software/fidder/fidder_v5.ckpt"
model_name = model_path.split(os.sep)[-1]

for path in tqdm(sorted(glob(os.path.join(base_dir, "*.mrc")))):
    file_name = path.split(os.sep)[-1]
    
    # load your image
    image = torch.tensor(mrcfile.read(path))

    # use a pretrained model to predict a mask
    mask, probabilities = predict_fiducial_mask(
        image, pixel_spacing=pixel_size, probability_threshold=0.5, model_checkpoint_file=model_path
    )

    # erase fiducials
    erased_image = erase_masked_region(image=image, mask=mask)

    if erased_image.is_cuda:
        erased_image = erased_image.cpu()
    erased_image_numpy = erased_image.numpy()

    mrcfile.new(os.path.join(save_dir, f"{model_name}_"+file_name), erased_image_numpy, overwrite=True)

