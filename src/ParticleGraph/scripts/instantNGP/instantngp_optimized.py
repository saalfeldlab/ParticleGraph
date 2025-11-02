#!/usr/bin/env python3

import torch
import tinycudann as tcnn
import json
import time
import os
import numpy as np
from PIL import Image
from tqdm import trange
import matplotlib.pyplot as plt

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

def create_config(n_levels, log2_hashmap_size, n_features_per_level, n_neurons, n_hidden_layers):
    """Create InstantNGP configuration"""
    return {
        "encoding": {
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": 16,
            "per_level_scale": 1.5
        },
        "network": {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": n_neurons,
            "n_hidden_layers": n_hidden_layers
        }
    }

def count_parameters(config, device):
    """Count parameters in a configuration"""
    try:
        model = tcnn.NetworkWithInputEncoding(
            n_input_dims=2,
            n_output_dims=3,
            encoding_config=config['encoding'],
            network_config=config['network']
        ).to(device)
        return sum(p.numel() for p in model.parameters())
    except:
        return 0

def test_configuration(config, img_tensor, coords, target, device, max_iterations=2000):
    """Test a single configuration and return PSNR and timing"""
    try:
        # Create model
        model = tcnn.NetworkWithInputEncoding(
            n_input_dims=2,
            n_output_dims=3,
            encoding_config=config['encoding'],
            network_config=config['network']
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        # Convert target to half precision
        target_half = target.to(device).half()
        
        # Training loop
        start_time = time.time()
        best_psnr = 0
        
        for i in range(max_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(coords)
            loss = torch.mean((output - target_half) ** 2)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate PSNR every 100 iterations
            if i % 100 == 0:
                with torch.no_grad():
                    current_psnr = calculate_psnr(output, target_half)
                    best_psnr = max(best_psnr, current_psnr.item())
        
        training_time = time.time() - start_time
        
        # Final PSNR
        with torch.no_grad():
            final_output = model(coords)
            final_psnr = calculate_psnr(final_output, target_half).item()
            best_psnr = max(best_psnr, final_psnr)
        
        return best_psnr, training_time, True
        
    except Exception as e:
        print(f"Error: {e}")
        return 0, 0, False

def main():
    print("InstantNGP Configuration Optimization")
    print("=====================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load image at FULL resolution for high-res optimization
    img_path = "/groups/saalfeld/home/allierc/Py/ParticleGraph/src/ParticleGraph/scripts/instantNGP/Girl_with_a_Pearl_Earring.jpg"
    target_size = None  # Use full resolution (2138x1808)
    
    img_tensor = load_image(img_path, target_size)
    height, width = img_tensor.shape[:2]
    
    print(f"Image size: {width}x{height}")
    
    # Get coordinates
    coords = get_coordinates(height, width, device)
    target = img_tensor.reshape(-1, 3)
    
    # Define configuration space optimized for FULL RESOLUTION
    configs_to_test = [
        # Format: (n_levels, log2_hashmap_size, n_features_per_level, n_neurons, n_hidden_layers, name)
        # High-capacity configs for full resolution
        (16, 15, 2, 64, 2, "Original_Baseline"),
        (18, 16, 2, 64, 2, "Extra_Large_18_16"),
        (20, 16, 2, 64, 2, "Ultra_Large_20_16"),
        (16, 16, 2, 64, 2, "Large_Hash_16"),
        (16, 17, 2, 64, 2, "XL_Hash_17"),
        (16, 15, 4, 64, 2, "More_Features_4"),
        (16, 15, 2, 128, 2, "Large_Network_128"),
        (16, 15, 2, 64, 3, "Deep_Network_3"),
        (18, 15, 2, 128, 2, "Hybrid_18_128"),
        (16, 16, 4, 128, 3, "Maximum_Capacity"),
    ]
    
    results = []
    
    print(f"\nTesting {len(configs_to_test)} configurations...")
    print("=" * 80)
    
    for i, (n_levels, log2_hashmap_size, n_features_per_level, n_neurons, n_hidden_layers, name) in enumerate(configs_to_test):
        config = create_config(n_levels, log2_hashmap_size, n_features_per_level, n_neurons, n_hidden_layers)
        param_count = count_parameters(config, device)
        
        if param_count == 0:
            print(f"{i+1:2d}. {name:20s}: FAILED (Invalid config)")
            continue
        
        print(f"{i+1:2d}. {name:20s}: {param_count:7,} params - Testing...", end=" ", flush=True)
        
        psnr, train_time, success = test_configuration(config, img_tensor, coords, target, device)
        
        if success:
            reduction_factor = 715536 / param_count
            print(f"PSNR: {psnr:5.2f} dB, Time: {train_time:4.1f}s, Reduction: {reduction_factor:.1f}x")
            
            results.append({
                'name': name,
                'config': config,
                'params': param_count,
                'psnr': psnr,
                'time': train_time,
                'reduction': reduction_factor
            })
        else:
            print("FAILED")
    
    # Analyze results
    if results:
        print("\n" + "=" * 80)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("=" * 80)
        
        # Sort by different criteria
        by_psnr = sorted(results, key=lambda x: x['psnr'], reverse=True)
        by_params = sorted(results, key=lambda x: x['params'])
        by_reduction = sorted(results, key=lambda x: x['reduction'], reverse=True)
        
        print("\nTop 5 by PSNR:")
        for i, r in enumerate(by_psnr[:5]):
            print(f"{i+1}. {r['name']:20s}: {r['psnr']:5.2f} dB ({r['params']:,} params, {r['reduction']:.1f}x smaller)")
        
        print("\nTop 5 by Parameter Reduction:")
        for i, r in enumerate(by_reduction[:5]):
            print(f"{i+1}. {r['name']:20s}: {r['reduction']:4.1f}x smaller ({r['params']:,} params, {r['psnr']:5.2f} dB)")
        
        print("\nBALANCED RECOMMENDATIONS:")
        print("-" * 50)
        
        # Good quality with significant reduction
        quality_reduced = [r for r in results if r['psnr'] > 25 and r['reduction'] > 2]
        if quality_reduced:
            best_quality = max(quality_reduced, key=lambda x: x['psnr'])
            print(f"Best Quality + Reduction: {best_quality['name']}")
            print(f"  {best_quality['psnr']:.2f} dB, {best_quality['params']:,} params ({best_quality['reduction']:.1f}x smaller)")
        
        # Best parameter efficiency
        for r in results:
            r['efficiency'] = r['psnr'] / (r['params'] / 1000)
        
        best_efficient = max(results, key=lambda x: x['efficiency'])
        print(f"Most Parameter Efficient: {best_efficient['name']}")
        print(f"  {best_efficient['psnr']:.2f} dB, {best_efficient['params']:,} params")
        
        # Save best configurations
        configs_to_save = []
        configs_to_save.append(('config_highest_psnr.json', by_psnr[0]['config']))
        configs_to_save.append(('config_most_efficient.json', best_efficient['config']))
        configs_to_save.append(('config_smallest.json', by_params[0]['config']))
        
        if quality_reduced:
            configs_to_save.append(('config_best_balanced.json', best_quality['config']))
        
        print(f"\nSaving {len(configs_to_save)} optimized configurations...")
        for filename, config in configs_to_save:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"  Saved: {filename}")
        
    else:
        print("\nNo successful configurations found!")

if __name__ == "__main__":
    main()
