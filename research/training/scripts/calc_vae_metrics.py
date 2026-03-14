import numpy as np
import torch
import cv2
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(r"c:\Projects\elsworld\research\training\src")))

from elworld.utils.real_vs_vision import load_vision_model

def calculate_metrics(data_path, checkpoint_path, num_frames=100, device='cuda'):
    model, _ = load_vision_model(checkpoint_path, device)
    
    data = np.load(data_path)
    obs = data['obs']
    num_frames = min(len(obs), num_frames)
    
    mse_list = []
    ssim_list = []
    
    print(f"Calculating metrics for {num_frames} frames...")
    
    with torch.no_grad():
        for i in range(num_frames):
            real = obs[i] # RGB (H, W, C)
            
            # Preprocess
            x = real.transpose(2, 0, 1)
            x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(device)
            if x_tensor.max() > 1.0:
                x_tensor = x_tensor / 255.0
                
            # Recon
            output = model(x_tensor)
            recon = output['x_recon'].squeeze(0).cpu().numpy().transpose(1, 2, 0)
            recon = (recon * 255.0).clip(0, 255).astype(np.uint8)
            
            # MSE
            mse = np.mean((real.astype(np.float32) - recon.astype(np.float32))**2)
            mse_list.append(mse)
            
            # PSNR
            if mse > 0:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            else:
                psnr = 100
            ssim_list.append(psnr) # Reuse the list for PSNR
            
            if (i+1) % 20 == 0:
                print(f"  Frame {i+1}/{num_frames}")
                
    print("\n--- VAE RECONSTRUCTION METRICS ---")
    print(f"Average MSE: {np.mean(mse_list):.4f}")
    print(f"Average PSNR: {np.mean(ssim_list):.4f} dB")
    print("----------------------------------")

if __name__ == "__main__":
    calculate_metrics(
        data_path=r"c:\Projects\elsworld\research\training\recorded\elsword_gameplay_01.npz",
        checkpoint_path=r"c:\Projects\elsworld\research\training\src\checkpoints\vision\vision_model_checkpoint_31"
    )
