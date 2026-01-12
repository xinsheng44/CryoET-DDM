# CryoET-DDM: restoring high-fidelity signals in cryo-electron tomography via a data-driven diffusion model

A deep learning-based denoising framework for cryo-electron tomography (CryoET) tilt series using diffusion models.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training Pipeline](#training-pipeline)
- [Inference](#inference)
- [Datasets & Pre-trained Models](#datasets--pre-trained-models)

## Overview

This repository implements a three-step pipeline for training and applying denoising models to CryoET data:

1. **Forward Process (Step 1)**: Creates diffusion datasets by progressively adding noise to particle patches
2. **Training (Step 2)**: Trains a U-Net based denoising model using the diffusion dataset
3. **Prediction (Step 3)**: Applies the trained model to denoise tilt series images

## Installation

### Install Dependencies

Using requirements.txt (recommended):

```bash
pip install -r requirements.txt
```


## Quick Start

### Using Pre-trained Models

1. Download pre-trained models (see [Datasets & Pre-trained Models](#datasets--pre-trained-models))
2. Run inference:

```bash
python step3_predict.py \
    --input_path /path/to/tilt_series \
    --out_path /path/to/output \
    --model_path /path/to/pretrained_model/best_model.pth \
    --gpus 0 \
    --lenx 3710 \
    --leny 3710
```

### Training from Scratch

1. Prepare your data (see [Data Preparation](#data-preparation))
2. Create diffusion dataset: `step1_forward_process.py`
3. Train the model: `step2_train.py`
4. Run inference: `step3_predict.py`

## Data Preparation

Before training, you need to prepare particle patches and noise patches from your CryoET tilt series. The data preparation pipeline includes the following steps:

### Step 0: Data Preprocessing Pipeline

The data preparation scripts are located in the `script/` directory. The typical workflow is:

#### 0.1 Remove Fiducial Markers

Remove fiducial markers from tilt series using [Fidder](https://github.com/teamtomo/fidder):

```bash
python script/step1.1_earse_fiducial.py \
    --input_path /path/to/tilt_series \
    --output_path /path/to/tilt_series_erased \
    [other parameters]
```

**Note**: This step uses [Fidder](https://github.com/teamtomo/fidder), a Python package for detecting and erasing gold fiducials in cryo-EM images. The Fidder package is included in `package/fidder/` directory. 

#### 0.2 Extract Noise Patches

1. Get noise coordinates:
```bash
python script/step1.2_getNoiseCoords.py \
    --input_path /path/to/tilt_series \
    --output_path /path/to/noise_coords \
    [other parameters]
```

2. Check noise patches (optional):
```bash
python script/step1.3_checkNoisePatches.py \
    --input_path /path/to/noise_coords \
    [other parameters]
```

3. Extract noise patches:
```bash
python script/step1.4_getNoisePatches.py \
    --input_path /path/to/noise_coords \
    --output_path /path/to/noise_patches \
    [other parameters]
```

#### 0.3 Extract Particle Patches

1. Get particle coordinates:
```bash
python script/step1.5_getParticleCoords.py \
    --input_path /path/to/tilt_series \
    --output_path /path/to/particle_coords \
    [other parameters]
```

2. Check particle patches (optional):
```bash
python script/step1.6_checkParticlePatches.py \
    --input_path /path/to/particle_coords \
    [other parameters]
```

3. Extract particle patches:
```bash
python script/step1.7_getParticlePatches.py \
    --input_path /path/to/particle_coords \
    --output_path /path/to/particle_patches \
    [other parameters]
```

**Note**: Detailed parameters for each script can be found in the script files. Adjust parameters based on your dataset characteristics.

### Output Structure

After data preparation, you should have:

```
data/
├── particle_patches/    # Particle patch MRC files
│   ├── patch_0000.mrc
│   ├── patch_0001.mrc
│   └── ...
└── noise_patches/       # Noise patch MRC files
    ├── patch_0000.mrc
    ├── patch_0001.mrc
    └── ...
```

## Training Pipeline

### Step 1: Forward Process - Create Diffusion Dataset

This step generates training data by progressively adding noise to particle patches.

**Usage:**
```bash
python step1_forward_process.py \
    --particle_patches /path/to/particle_patches \
    --noise_patches /path/to/noise_patches \
    --particle_num 4000 \
    --noise_num -1 \
    --out_path /path/to/output
```

**Parameters:**
- `--particle_patches`: Path to directory containing particle patch MRC files
- `--noise_patches`: Path to directory containing noise patch MRC files
- `--particle_num`: Number of particle patches to use (-1 for all)
- `--noise_num`: Number of noise patches to use (-1 for all)
- `--out_path`: Output path for the diffusion dataset

**Configuration:**
Edit the script to configure:
- `alpha`: Diffusion coefficient (default: 0.05)
- `add_noise_num`: Number of noise addition steps (default: 5)

**Output:**
Creates a directory structure:
```
output_path/
├── train/
│   ├── inputs.mrcs
│   ├── outputs.mrcs
│   └── noises.mrcs
└── val/
    ├── inputs.mrcs
    ├── outputs.mrcs
    └── noises.mrcs
```

### Step 2: Training

Trains the denoising model using the diffusion dataset.

**Usage:**
```bash
python step2_train.py \
    --input_path /path/to/diffusion_dataset \
    --out_path /path/to/save/model \
    --batch_size 32 \
    --gpus 0 \
    --patience 10
```

**Parameters:**
- `--input_path` / `-i`: Path to the diffusion dataset (from Step 1)
- `--out_path` / `-o`: Directory to save trained models
- `--batch_size` / `-b`: Batch size for training (default: 32)
- `--gpus` / `-d`: GPU ID to use (default: "0")
- `--patience` / `-p`: Early stopping patience (default: 10)

**Output:**
- `best_model.pth`: Best model based on validation loss
- `model_epoch_N.pth`: Checkpoint models saved every 2 epochs
- `log.txt`: Training log with loss values

**Training Configuration:**
- Epochs: 51
- Learning rate: 0.001
- Optimizer: AdamW (weight_decay=0.01)
- Scheduler: StepLR (step_size=100, gamma=0.9)
- Loss: MSE Loss + Consistency Loss (weight: 0.1)

## Inference

### Step 3: Prediction

Applies the trained model to denoise tilt series images.

**Usage:**
```bash
python step3_predict.py \
    --input_path /path/to/tilt_series \
    --out_path /path/to/output \
    --model_path /path/to/best_model.pth \
    --data_num -1 \
    --gpus 0 \
    --lenx 3710 \
    --leny 3710
```

**Parameters:**
- `--input_path` / `-i`: Path to directory containing input MRC files
- `--out_path` / `-o`: Output directory for denoised images
- `--model_path` / `-m`: Path to trained model checkpoint
- `--data_num`: Number of files to process (-1 for all)
- `--gpus` / `-d`: GPU ID to use
- `--lenx` / `-x`: Image width
- `--leny` / `-y`: Image height

**Output:**
Denoised MRC files saved to the output directory.

## Model Architecture

The default model is a U-Net based architecture (`UDenoiseNet`) adapted from the Noise2Noise/Topaz approach:

- **Encoder**: 6 downsampling layers with LeakyReLU activation
- **Decoder**: 5 upsampling layers with skip connections
- **Base filters**: 48 channels
- **Input/Output**: Single channel 2D images

The model can be found in `model/UNet2d.py`.

## Datasets & Pre-trained Models

### Datasets

The following datasets were used for training and evaluation:

- **EMPIAR-10499**: [Link to EMPIAR-10499](https://www.ebi.ac.uk/empiar/EMPIAR-10499/)
- **EMPIAR-10164**: [Link to EMPIAR-10164](https://www.ebi.ac.uk/empiar/EMPIAR-10164/)
- **EMPIAR-10651**: [Link to EMPIAR-10651](https://www.ebi.ac.uk/empiar/EMPIAR-10651/)

### Pre-trained Models

Pre-trained models are available for download:

- **Model for EMPIAR-10499**: [Download Link](https://pan.baidu.com/s/1-tb9elDmZV62UlsKbepjMg?pwd=i65j)
- **Model for EMPIAR-10164**: [Download Link](https://pan.baidu.com/s/1-tb9elDmZV62UlsKbepjMg?pwd=i65j)
- **Model for EMPIAR-10651**: [Download Link](https://pan.baidu.com/s/1-tb9elDmZV62UlsKbepjMg?pwd=i65j)

**Note**: Please update the download links with actual URLs where you host the pre-trained models.

### Processed Training Data

If you want to skip the data preparation step, processed training data (particle patches and noise patches) can be downloaded:

- **Data for EMPIAR-10499**: [Download Link](https://pan.baidu.com/s/1UiPxwLu7zbK0fJdo9lm1ng?pwd=djp7)
- **Data for EMPIAR-10164**: [Download Link](https://pan.baidu.com/s/1UiPxwLu7zbK0fJdo9lm1ng?pwd=djp7)
- **Data for EMPIAR-10651**: [Download Link](https://pan.baidu.com/s/1UiPxwLu7zbK0fJdo9lm1ng?pwd=djp7)
