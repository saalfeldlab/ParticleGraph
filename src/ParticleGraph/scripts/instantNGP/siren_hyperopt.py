#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import time
from PIL import Image as PILImage
import os
import json
from itertools import product

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                           1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                           np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                 outermost_linear=False, first_omega_0=30, hidden_omega_0=30):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                            np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output

def read_image(filename):
    """Read image and convert to numpy array with values in [0,1]"""
    img = PILImage.open(filename).convert('RGB')
    return np.array(img).astype(np.float32) / 255.0

def write_image(filename, img_array):
    """Write numpy array to image file with explicit RGB mode"""
    img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
    img = PILImage.fromarray(img_array, mode='RGB')
    img.save(filename)

def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

def test_siren_config(config, original_img, coords, target_pixels, device, test_iterations=20):
    """Test a specific SIREN configuration"""
    
    height, width, channels = original_img.shape
    
    # Create model with config
    model = Siren(
        in_features=2, 
        out_features=3, 
        hidden_features=config['hidden_features'],
        hidden_layers=config['hidden_layers'], 
        outermost_linear=True,
        first_omega_0=config['first_omega_0'],
        hidden_omega_0=config['hidden_omega_0']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training parameters
    batch_size = 50000
    n_pixels = coords.shape[0]
    n_batches = (n_pixels + batch_size - 1) // batch_size
    
    # Train for fixed number of iterations and measure time
    start_time = time.perf_counter()
    
    for i in range(test_iterations):
        optimizer.zero_grad()
        
        total_loss = 0.0
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_pixels)
            
            batch_coords = coords[start_idx:end_idx]
            batch_targets = target_pixels[start_idx:end_idx]
            
            batch_predicted = model(batch_coords)
            batch_loss = torch.nn.functional.mse_loss(batch_predicted, batch_targets)
            
            batch_weight = (end_idx - start_idx) / n_pixels
            scaled_loss = batch_loss * batch_weight
            scaled_loss.backward()
            
            total_loss += batch_loss.item() * batch_weight
        
        optimizer.step()
    
    training_time = time.perf_counter() - start_time
    
    # Calculate final PSNR
    with torch.no_grad():
        predicted_full = torch.zeros_like(target_pixels)
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_pixels)
            batch_coords = coords[start_idx:end_idx]
            predicted_full[start_idx:end_idx] = model(batch_coords)
        
        predicted_image = predicted_full.reshape(height, width, channels)
        predicted_image = torch.clamp(predicted_image, 0, 1)
        
        reconstructed_img = predicted_image.cpu().numpy()
        psnr = calculate_psnr(original_img, reconstructed_img)
    
    return {
        'psnr': psnr,
        'training_time': training_time,
        'iterations_per_second': test_iterations / training_time,
        'final_loss': total_loss,
        'model_parameters': sum(p.numel() for p in model.parameters())
    }

def main():
    print("================================================================")
    print("SIREN Hyperparameter Optimization")
    print("Testing different configurations for Girl with a Pearl Earring")
    print("================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load image
    image_path = "/groups/saalfeld/home/allierc/Py/ParticleGraph/src/ParticleGraph/scripts/instantNGP/Girl_with_a_Pearl_Earring.jpg"
    original_img = read_image(image_path)
    height, width, channels = original_img.shape
    print(f"Image dimensions: {height}x{width} with {channels} channels")
    
    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    x_coords = x_coords.astype(np.float32) / (width - 1)  # Normalize to [0, 1]
    y_coords = y_coords.astype(np.float32) / (height - 1)  # Normalize to [0, 1]
    coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)
    coords = torch.from_numpy(coords).to(device)
    
    target_pixels = torch.from_numpy(original_img.reshape(-1, channels)).to(device)
    
    # Define focused hyperparameter grid (most promising ranges)
    config_grid = {
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'hidden_features': [256, 512],
        'hidden_layers': [4, 5],
        'first_omega_0': [30, 60],
        'hidden_omega_0': [30, 60]
    }
    
    print(f"Testing {np.prod([len(v) for v in config_grid.values()])} configurations...")
    print("Each config tested with 20 iterations for fair comparison")
    print("================================================================")
    
    results = []
    best_psnr = 0
    best_config = None
    
    total_configs = np.prod([len(v) for v in config_grid.values()])
    config_count = 0
    
    # Test all combinations
    for lr, hf, hl, fo, ho in product(
        config_grid['learning_rate'],
        config_grid['hidden_features'], 
        config_grid['hidden_layers'],
        config_grid['first_omega_0'],
        config_grid['hidden_omega_0']
    ):
        config_count += 1
        
        config = {
            'learning_rate': lr,
            'hidden_features': hf,
            'hidden_layers': hl,
            'first_omega_0': fo,
            'hidden_omega_0': ho
        }
        
        print(f"\nConfig {config_count:3d}/{total_configs}: lr={lr:.0e}, hf={hf}, hl={hl}, fo={fo}, ho={ho}")
        
        try:
            result = test_siren_config(config, original_img, coords, target_pixels, device)
            result['config'] = config
            results.append(result)
            
            print(f"  PSNR: {result['psnr']:6.2f} dB, Time: {result['training_time']:6.2f}s, "
                  f"Speed: {result['iterations_per_second']:6.2f} it/s, Params: {result['model_parameters']:,}")
            
            if result['psnr'] > best_psnr:
                best_psnr = result['psnr']
                best_config = config
                print(f"  *** NEW BEST PSNR: {best_psnr:.2f} dB ***")
                
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue
    
    # Sort results by PSNR
    results.sort(key=lambda x: x['psnr'], reverse=True)
    
    print("\n================================================================")
    print("OPTIMIZATION RESULTS")
    print("================================================================")
    print("Top 10 configurations by PSNR:")
    print()
    
    for i, result in enumerate(results[:10]):
        config = result['config']
        print(f"{i+1:2d}. PSNR: {result['psnr']:6.2f} dB | "
              f"lr={config['learning_rate']:.0e} hf={config['hidden_features']} "
              f"hl={config['hidden_layers']} fo={config['first_omega_0']} ho={config['hidden_omega_0']} | "
              f"Speed: {result['iterations_per_second']:5.2f} it/s | "
              f"Params: {result['model_parameters']:,}")
    
    print(f"\nBest configuration:")
    print(f"Learning rate: {best_config['learning_rate']}")
    print(f"Hidden features: {best_config['hidden_features']}")  
    print(f"Hidden layers: {best_config['hidden_layers']}")
    print(f"First omega_0: {best_config['first_omega_0']}")
    print(f"Hidden omega_0: {best_config['hidden_omega_0']}")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    
    # Save results
    with open('siren_optimization_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = []
        for result in results:
            json_result = {
                'config': result['config'],
                'psnr': float(result['psnr']),
                'training_time': float(result['training_time']),
                'iterations_per_second': float(result['iterations_per_second']),
                'final_loss': float(result['final_loss']),
                'model_parameters': int(result['model_parameters'])
            }
            json_results.append(json_result)
        
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to 'siren_optimization_results.json'")
    print("================================================================")

if __name__ == "__main__":
    main()