#!/usr/bin/env python3

# Full resolution InstantNGP optimization based on the fixed instantngp_girl_timed.py
import argparse
import json
import numpy as np
import os
import sys
import torch
import time
from tqdm import trange
import shutil

try:
	import tinycudann as tcnn
except ImportError:
	print("This sample requires the tiny-cuda-nn extension for PyTorch.")
	sys.exit()

from PIL import Image as PILImage
import numpy as np

def read_image(filename):
    """Read image and convert to numpy array with values in [0,1]"""
    img = PILImage.open(filename).convert('RGB')
    return np.array(img).astype(np.float32) / 255.0

def write_image(filename, img_array):
    """Write numpy array to image file with explicit RGB mode"""
    img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
    img = PILImage.fromarray(img_array, mode='RGB')
    img.save(filename)

class Image(torch.nn.Module):
	def __init__(self, filename, device):
		super(Image, self).__init__()
		self.data = read_image(filename)
		self.shape = self.data.shape
		self.data = torch.from_numpy(self.data).float().to(device)

	def forward(self, xs):
		with torch.no_grad():
			# Bilinearly filtered lookup from the image. Not super fast,
			# but less than ~20% of the overall runtime of this example.
			shape = self.shape

			xs = xs * torch.tensor([shape[1], shape[0]], device=xs.device).float()
			indices = xs.long()
			lerp_weights = xs - indices.float()

			x0 = indices[:, 0].clamp(min=0, max=shape[1]-1)
			y0 = indices[:, 1].clamp(min=0, max=shape[0]-1)
			x1 = (x0 + 1).clamp(max=shape[1]-1)
			y1 = (y0 + 1).clamp(max=shape[0]-1)

			return (
				self.data[y0, x0] * (1.0 - lerp_weights[:,0:1]) * (1.0 - lerp_weights[:,1:2]) +
				self.data[y0, x1] * lerp_weights[:,0:1] * (1.0 - lerp_weights[:,1:2]) +
				self.data[y1, x0] * (1.0 - lerp_weights[:,0:1]) * lerp_weights[:,1:2] +
				self.data[y1, x1] * lerp_weights[:,0:1] * lerp_weights[:,1:2]
			)

def create_config(n_levels, log2_hashmap_size, n_features_per_level, n_neurons, n_hidden_layers):
    """Create InstantNGP configuration"""
    return {
        "otype": "NetworkWithInputEncoding",
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
            "output_activation": "Sigmoid",
            "n_neurons": n_neurons,
            "n_hidden_layers": n_hidden_layers
        }
    }

def test_config(config_name, config, device, test_iterations=1000):
    """Test a configuration and return PSNR and parameter count"""
    try:
        print(f"Testing {config_name}...", end=" ", flush=True)
        
        # Load image
        image = Image("Girl_with_a_Pearl_Earring.jpg", device)
        img_shape = image.shape
        
        # Create model  
        with open("temp_config.json", "w") as f:
            json.dump(config, f)
        
        model = tcnn.NetworkWithInputEncoding(
            n_input_dims=2,
            n_output_dims=3,
            encoding_config=config["encoding"],
            network_config=config["network"]
        ).to(device)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        # Generate coordinates (same as in original)
        n_pixels = img_shape[0] * img_shape[1]
        batch_size = min(4194304, n_pixels)  # Same batch size logic
        
        xs = torch.rand(batch_size, 2, device=device)
        
        # Training loop
        start_time = time.time()
        best_loss = float('inf')
        
        for i in range(test_iterations):
            xs = torch.rand(batch_size, 2, device=device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                targets = image(xs)
            
            outputs = model(xs)
            loss = torch.mean((outputs - targets) ** 2)
            
            loss.backward()
            optimizer.step()
            
            best_loss = min(best_loss, loss.item())
        
        train_time = time.time() - start_time
        
        # Calculate PSNR from loss
        psnr = -10.0 * np.log10(best_loss) if best_loss > 0 else 50.0
        
        print(f"PSNR: {psnr:5.2f} dB, {param_count:,} params, {train_time:.1f}s")
        
        # Cleanup
        if os.path.exists("temp_config.json"):
            os.remove("temp_config.json")
            
        return psnr, param_count, train_time, True
        
    except Exception as e:
        print(f"FAILED: {e}")
        if os.path.exists("temp_config.json"):
            os.remove("temp_config.json")
        return 0, 0, 0, False

def main():
    print("InstantNGP Full Resolution Optimization")
    print("Based on fixed instantngp_girl_timed.py")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Image: Girl_with_a_Pearl_Earring.jpg (Full resolution: 2138x1808)")
    
    # Configuration space optimized for full resolution
    configs_to_test = [
        # Format: (n_levels, log2_hashmap_size, n_features_per_level, n_neurons, n_hidden_layers, name)
        (16, 15, 2, 64, 2, "Original_Baseline"),
        (18, 16, 2, 64, 2, "Large_18_16"),
        (20, 16, 2, 64, 2, "XLarge_20_16"),
        (16, 16, 2, 64, 2, "Large_Hash_16"),
        (16, 17, 2, 64, 2, "XL_Hash_17"),
        (16, 15, 4, 64, 2, "More_Features_4"),
        (16, 15, 2, 128, 2, "Large_Network_128"),
        (16, 15, 2, 64, 3, "Deep_Network_3"),
        (18, 15, 2, 128, 2, "Hybrid_18_128"),
        (22, 17, 4, 128, 3, "Maximum_Capacity"),
    ]
    
    results = []
    
    print(f"\nTesting {len(configs_to_test)} configurations at FULL resolution...")
    print("=" * 80)
    
    for i, (n_levels, log2_hashmap_size, n_features_per_level, n_neurons, n_hidden_layers, name) in enumerate(configs_to_test):
        config = create_config(n_levels, log2_hashmap_size, n_features_per_level, n_neurons, n_hidden_layers)
        
        print(f"{i+1:2d}. {name:20s}: ", end="")
        
        psnr, params, train_time, success = test_config(name, config, device, test_iterations=1000)
        
        if success:
            baseline_params = 715536  # Original config parameters
            reduction_factor = baseline_params / params if params > 0 else 0
            
            results.append({
                'name': name,
                'config': config,
                'params': params,
                'psnr': psnr,
                'time': train_time,
                'reduction': reduction_factor
            })
    
    # Analyze results
    if results:
        print("\n" + "=" * 80)
        print("FULL RESOLUTION OPTIMIZATION RESULTS")
        print("=" * 80)
        
        # Sort by different criteria
        by_psnr = sorted(results, key=lambda x: x['psnr'], reverse=True)
        by_params = sorted(results, key=lambda x: x['params'])
        
        print("\nTop 5 by PSNR (Quality):")
        for i, r in enumerate(by_psnr[:5]):
            ratio = r['params'] / 715536
            print(f"{i+1}. {r['name']:20s}: {r['psnr']:5.2f} dB ({r['params']:,} params, {ratio:.1f}x size)")
        
        print("\nTop 5 Most Efficient (PSNR per 100K params):")
        for r in results:
            r['efficiency'] = r['psnr'] / (r['params'] / 100000)
        
        by_efficiency = sorted(results, key=lambda x: x['efficiency'], reverse=True)
        for i, r in enumerate(by_efficiency[:5]):
            print(f"{i+1}. {r['name']:20s}: {r['efficiency']:5.2f} eff ({r['psnr']:.2f} dB, {r['params']:,} params)")
        
        # Save best configurations
        print(f"\nSaving optimized configurations for full resolution...")
        
        configs_to_save = [
            ('config_fullres_best_quality.json', by_psnr[0]['config']),
            ('config_fullres_most_efficient.json', by_efficiency[0]['config']),
            ('config_fullres_largest.json', sorted(results, key=lambda x: x['params'], reverse=True)[0]['config'])
        ]
        
        for filename, config in configs_to_save:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"  Saved: {filename}")
        
        print(f"\n🎯 RECOMMENDATION for full resolution (2138x1808):")
        best = by_psnr[0]
        print(f"   Use: {best['name']}")
        print(f"   PSNR: {best['psnr']:.2f} dB")
        print(f"   Parameters: {best['params']:,}")
        print(f"   Config file: config_fullres_best_quality.json")
        
    else:
        print("\nNo successful configurations found!")

if __name__ == "__main__":
    main()