#!/usr/bin/env python3

import torch
import tinycudann as tcnn
import json
import time
import os
import numpy as np
from PIL import Image
from tqdm import trange

def load_image(path, target_size=None):
    """Load and preprocess image"""
    img = Image.open(path).convert('RGB')
    if target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    img_array = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_array)

def get_coordinates(height, width, device):
    """Generate normalized coordinate grid"""
    y = torch.linspace(0, 1, height, device=device, dtype=torch.half)
    x = torch.linspace(0, 1, width, device=device, dtype=torch.half)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([X, Y], dim=-1).reshape(-1, 2)
    return coords

def calculate_psnr(pred, target):
    """Calculate PSNR between prediction and target"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def write_image(tensor, filepath):
    """Save tensor as PNG image"""
    if tensor.dim() == 3 and tensor.shape[-1] == 3:
        img_array = tensor.detach().cpu().numpy()
    else:
        img_array = tensor.reshape(tensor.shape[0], tensor.shape[1], 3).detach().cpu().numpy()
    
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(filepath, 'PNG')

def main():
    print("InstantNGP PSNR Debug Test")
    print("==========================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test different configurations and resolutions
    configs_to_test = [
        ("config_highest_psnr.json", "Best PSNR Config"),
        ("config_hash.json", "Original Config"),
    ]
    
    resolutions_to_test = [
        ((512, 432), "Medium Resolution"),
        ((1024, 864), "High Resolution"),
        ((256, 216), "Low Resolution"),
    ]
    
    img_path = "/groups/saalfeld/home/allierc/Py/ParticleGraph/src/ParticleGraph/scripts/instantNGP/Girl_with_a_Pearl_Earring.jpg"
    
    print("\nTesting different configs and resolutions...")
    print("=" * 80)
    
    for config_file, config_name in configs_to_test:
        if not os.path.exists(config_file):
            print(f"Skipping {config_name} - file {config_file} not found")
            continue
            
        with open(config_file) as f:
            config = json.load(f)
        
        print(f"\n{config_name} ({config_file}):")
        
        for target_size, res_name in resolutions_to_test:
            print(f"  {res_name} {target_size[0]}x{target_size[1]}:", end=" ")
            
            try:
                # Load image
                img_tensor = load_image(img_path, target_size)
                height, width = img_tensor.shape[:2]
                
                # Get coordinates
                coords = get_coordinates(height, width, device)
                target = img_tensor.reshape(-1, 3).to(device).half()
                
                # Create model
                model = tcnn.NetworkWithInputEncoding(
                    n_input_dims=2,
                    n_output_dims=3,
                    encoding_config=config['encoding'],
                    network_config=config['network']
                ).to(device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
                
                # Quick training test (1000 iterations)
                start_time = time.time()
                for i in range(1000):
                    optimizer.zero_grad()
                    output = model(coords)
                    loss = torch.mean((output - target) ** 2)
                    loss.backward()
                    optimizer.step()
                
                # Final PSNR
                with torch.no_grad():
                    final_output = model(coords)
                    final_psnr = calculate_psnr(final_output, target).item()
                
                train_time = time.time() - start_time
                param_count = sum(p.numel() for p in model.parameters())
                
                print(f"PSNR: {final_psnr:6.2f} dB, {param_count:,} params, {train_time:.1f}s")
                
            except Exception as e:
                print(f"ERROR: {e}")
    
    print("\n" + "=" * 80)
    print("Analysis: If PSNR is still low across all tests, there may be")
    print("a fundamental issue with the training setup or target format.")

if __name__ == "__main__":
    main()