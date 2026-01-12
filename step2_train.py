import mrcfile as mf
import torch
import numpy as np
import os
import time
import sys
import random
import argparse
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as data
import model.UNet2d as unet2d
from dataset.dataset import CustomDataset_noise
import copy


def main(preprocess_path, model_save_path, gpus, batch_size, patience=10):
    """
    Main function to train the model with early stopping and best model saving.
    """
    temp = str(time.asctime()) + ".log"
    log_file = open(os.path.join(model_save_path, "log.txt"), "a")
    log_file.write("Command: python " + " ".join(sys.argv) + "\n")

    # Load training data
    train_path = os.path.join(preprocess_path, "train")
    log_file.write("Reading training data from " + train_path + "\n")
    train_input_data = mf.read(os.path.join(train_path, "inputs.mrcs"))
    train_output_data = mf.read(os.path.join(train_path, "outputs.mrcs"))
    train_noise_data = mf.read(os.path.join(train_path, "noises.mrcs"))
    train_dataset = CustomDataset_noise(train_input_data, train_output_data, train_noise_data)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load validation data
    val_path = os.path.join(preprocess_path, "val")
    log_file.write("Reading validation data from " + val_path + "\n")
    val_input_data = mf.read(os.path.join(val_path, "inputs.mrcs"))
    val_output_data = mf.read(os.path.join(val_path, "outputs.mrcs"))
    val_noise_data = mf.read(os.path.join(val_path, "noises.mrcs"))
    val_dataset = CustomDataset_noise(val_input_data, val_output_data, val_noise_data)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Set device
    device = torch.device(f"cuda:{gpus}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    model = unet2d.UDenoiseNet().to(device)

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Define loss, optimizer, and scheduler
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

    EPOCHS = 51
    loss_weight = 0.1
    best_val_loss = float('inf')
    patience_counter = 0
    
    log_file.write("Training started...\n")

    for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
        model.train()
        train_loss = []
        for particle, particle_noise, noises in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training", leave=False):
            particle, particle_noise, noises = particle.unsqueeze(1).to(device), particle_noise.unsqueeze(1).to(device), noises.unsqueeze(1).to(device)
            predicts = model(particle_noise)
            noise_outputs = model(noises)
            noise1 = abs(predicts - particle_noise)
            noise2 = abs(noises - noise_outputs)
            
            optimizer.zero_grad()
            mse_loss = criterion(predicts, particle)
            consistency_loss = criterion(noise1, noise2)
            loss = mse_loss + consistency_loss * loss_weight
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Maximum gradient norm
            optimizer.step()
            
            train_loss.append(loss.item())
            # Log loss values for each batch (commented out)
            # log_file.write(f"Epoch {epoch+1}, Batch {len(train_loss)}: MSE Loss: {mse_loss.item():.6f}, Consistency Loss: {consistency_loss.item():.6f}, Total Loss: {loss.item():.6f}\n")
        
        scheduler.step()
        avg_train_loss = np.mean(train_loss)

        # Validation phase
        model.eval()
        valid_loss = []
        val_mse_losses = []
        val_consistency_losses = []
        with torch.no_grad():
            for particle, particle_noise, noises in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation", leave=False):
                particle, particle_noise, noises = particle.unsqueeze(1).to(device), particle_noise.unsqueeze(1).to(device), noises.unsqueeze(1).to(device)
                predicts = model(particle_noise)
                noise_outputs = model(noises)
                noise1 = abs(predicts - particle_noise)
                noise2 = abs(noises - noise_outputs)
                
                mse_loss = criterion(predicts, particle)
                consistency_loss = criterion(noise1, noise2)
                loss = mse_loss + consistency_loss * loss_weight
                
                valid_loss.append(loss.item())
                val_mse_losses.append(mse_loss.item())
                val_consistency_losses.append(consistency_loss.item())

        avg_valid_loss = np.mean(valid_loss)
        avg_val_mse_loss = np.mean(val_mse_losses)
        avg_val_consistency_loss = np.mean(val_consistency_losses)
        
        if avg_train_loss > 1 or avg_valid_loss > 1:
            log_file.write("(warning)--Early stopping triggered at epoch " + str(epoch + 1) + "\n")
            break
        
        # Log detailed training and validation loss for each epoch
        log_file.write(f"Epoch {epoch+1} Summary:\n")
        log_file.write(f"  Train Loss: {avg_train_loss:.6f}\n")
        log_file.write(f"  Validation Loss: {avg_valid_loss:.6f}\n")
        log_file.write(f"  Validation MSE Loss: {avg_val_mse_loss:.6f}\n")
        log_file.write(f"  Validation Consistency Loss: {avg_val_consistency_loss:.6f}\n")
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_valid_loss:.6f}")
        
        # Save model checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join(model_save_path, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            log_file.write(f"Model checkpoint saved at epoch {epoch+1}\n")
        
        # Save best model
        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_save_path, "best_model.pth"))
            log_file.write("Best model saved at epoch " + str(epoch + 1) + "\n")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_file.write("Early stopping triggered at epoch " + str(epoch + 1) + "\n")
                break
    
    log_file.close()



if __name__ == "__main__":
    
    # alpha = 0.01
    # add_noise_num = 5
    
    
    # parser = argparse.ArgumentParser(description="Step 2 training")
    # parser.add_argument("--input_path", "-i", default=f"/data/wxs/tomo_denoise/TiltSeriesDDM_v2/data/EMPIAR-10499/DDM_data_{alpha}_{add_noise_num}", type=str, help="Preprocess path")
    # parser.add_argument("--out_path", "-o", type=str, default=f"./save/EMPIAR-10499_{alpha}_{add_noise_num}", help="Model save path")
    # parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    # parser.add_argument("--gpus", "-d", type=str, default="2", help="GPU ID")
    # parser.add_argument("--patience", "-p", type=int, default=10, help="Early stopping patience")
    
    # parser = argparse.ArgumentParser(description="Step 2 training")
    # parser.add_argument("--input_path", "-i", default=f"/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10164/DDM_data_{alpha}_{add_noise_num}", type=str, help="Preprocess path")
    # parser.add_argument("--out_path", "-o", type=str, default=f"./save/EMPIAR-10164_{alpha}_{add_noise_num}", help="Model save path")
    # parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    # parser.add_argument("--gpus", "-d", type=str, default="2", help="GPU ID")
    # parser.add_argument("--patience", "-p", type=int, default=100, help="Early stopping patience")
    
    
    # alpha = 0.01
    # add_noise_num = 5
    # parser = argparse.ArgumentParser(description="Step 2 training")
    # parser.add_argument("--input_path", "-i", default=f"/data/wxs/tomo_denoise/tiltDenoise/data/EMPIAR-10651/DDM_data_{alpha}_{add_noise_num}", type=str, help="Preprocess path")
    # parser.add_argument("--out_path", "-o", type=str, default=f"./save/EMPIAR-10651_{alpha}_{add_noise_num}", help="Model save path")
    # parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    # parser.add_argument("--gpus", "-d", type=str, default="3", help="GPU ID")
    # parser.add_argument("--patience", "-p", type=int, default=10, help="Early stopping patience")
    
    # args = parser.parse_args()

    # if not os.path.exists(args.out_path):
    #     os.makedirs(args.out_path)

    # main(args.input_path, args.out_path, args.gpus, args.batch_size, args.patience)
    
    
    ## emd_11603
    alpha = 0.01
    add_noise_num = 5
    parser = argparse.ArgumentParser(description="Step 2 training")
    parser.add_argument("--input_path", "-i", default=f"/data/wxs/tomo_denoise/tiltDenoise/data/emd_11603/DDM_data_{alpha}_{add_noise_num}", type=str, help="Preprocess path")
    parser.add_argument("--out_path", "-o", type=str, default=f"./save/emd_11603_{alpha}_{add_noise_num}", help="Model save path")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--gpus", "-d", type=str, default="1", help="GPU ID")
    parser.add_argument("--patience", "-p", type=int, default=10, help="Early stopping patience")
    
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    main(args.input_path, args.out_path, args.gpus, args.batch_size, args.patience)