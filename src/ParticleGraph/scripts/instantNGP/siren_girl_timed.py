#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import time
from PIL import Image as PILImage
import os

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
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
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
        output = self.net(coords)
        return output

def read_image(filename):
    """Read image and convert to numpy array with values in [0,1]"""
    img = PILImage.open(filename).convert('RGB')
    return np.array(img).astype(np.float32) / 255.0

def write_image(filename, img_array):
    """Write numpy array to image file"""
    img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
    img = PILImage.fromarray(img_array)
    img.save(filename)

def get_mgrid(height, width, dim=2):
    """Generates a flattened grid of (x,y) coordinates normalized to [0,1]."""
    x_coords = torch.linspace(0, 1, steps=width)
    y_coords = torch.linspace(0, 1, steps=height)
    mgrid = torch.stack(torch.meshgrid(y_coords, x_coords, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

if __name__ == "__main__":
    print("================================================================")
    print("SIREN Neural Network Image Reconstruction")
    print("Reconstructing: Girl with a Pearl Earring (Time-based)")
    print("================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get script directory and construct path to image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "Girl_with_a_Pearl_Earring.jpg")
    
    # Load image at full scale for comparison
    print(f"Loading image: {image_path}")
    image_data = read_image(image_path)  # Full scale image
    height, width, channels = image_data.shape
    print(f"Processing image dimensions: {height}x{width} with {channels} channels")

    # Convert to tensors
    target_image = torch.from_numpy(image_data).float().to(device)
    target_pixels = target_image.reshape(-1, channels)  # Flatten to (H*W, 3)

    # Create coordinate grid
    coords = get_mgrid(height, width, dim=2).to(device)
    print(f"Coordinate grid shape: {coords.shape}")
    print(f"Target pixels shape: {target_pixels.shape}")

    # Create SIREN model (smaller for efficiency)
    model = Siren(
        in_features=2,          # (x, y) coordinates
        hidden_features=256,    # Medium hidden layer
        hidden_layers=5,        # Reasonable depth
        out_features=3,         # RGB channels
        outermost_linear=True,
        first_omega_0=30,
        hidden_omega_0=30.
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create output directory and clear it
    import shutil
    if os.path.exists("siren_outputs"):
        shutil.rmtree("siren_outputs")
    os.makedirs("siren_outputs", exist_ok=True)
    print("Cleared and created output directory: siren_outputs/")

    print("PHASE 1: Calibration - measuring iterations per 250ms without I/O...")
    
    # Calibration phase - pure training for 40 seconds to get 40 save points (every 250ms)
    calibration_start = time.perf_counter()
    calibration_iterations = []
    calibration_times = []
    i = 0
    
    # Training parameters for calibration
    batch_size = 50000  # Process in smaller batches to fit GPU memory
    n_pixels = coords.shape[0]
    
    while time.perf_counter() - calibration_start < 40.0:
        optimizer.zero_grad()

        # Process in batches to avoid OOM
        total_loss = 0.0
        n_batches = (n_pixels + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_pixels)
            
            batch_coords = coords[start_idx:end_idx]
            batch_targets = target_pixels[start_idx:end_idx]
            
            # Forward pass
            batch_predicted = model(batch_coords)
            
            # Compute loss for this batch
            batch_loss = torch.nn.functional.mse_loss(batch_predicted, batch_targets)
            
            # Scale loss by batch proportion and accumulate
            batch_weight = (end_idx - start_idx) / n_pixels
            scaled_loss = batch_loss * batch_weight
            scaled_loss.backward()
            
            total_loss += batch_loss.item() * batch_weight

        # Step optimizer after all batches
        optimizer.step()
        
        current_time = time.perf_counter()
        elapsed_time = current_time - calibration_start
        
        # Record every 250ms
        if len(calibration_times) == 0 or elapsed_time >= (len(calibration_times) * 0.25):
            calibration_iterations.append(i)
            calibration_times.append(elapsed_time)
            print(f"Calibration: {elapsed_time:.3f}s = iteration {i}")
        
        i += 1
    
    print(f"Calibration completed: {i} iterations in 40 seconds")
    print(f"Save points (250ms intervals): {calibration_iterations}")
    
    print("\nPHASE 2: Training with iteration-based saving...")
    
    # Reset model for actual training
    model = Siren(in_features=2, out_features=3, hidden_features=256, 
                  hidden_layers=5, outermost_linear=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    start_time = time.perf_counter()
    total_training_time = 0.0
    save_counter = 0
    i = 0
    
    # Save initial state (t=0)
    path = f"siren_outputs/time_000_000ms.jpg"
    print(f"Writing '{path}'... ", end="")
    with torch.no_grad():
        predicted_image = model(coords).reshape(height, width, channels)
        predicted_image = torch.clamp(predicted_image, 0, 1)
        write_image(path, predicted_image.cpu().numpy())
    print("done.")
    save_counter += 1
    
    # Training loop using calibrated iteration points
    while i < calibration_iterations[-1]:  # Train to same number as calibration
        iter_start = time.perf_counter()
        
        optimizer.zero_grad()

        # Process in batches to avoid OOM
        total_loss = 0.0
        n_batches = (n_pixels + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_pixels)
            
            batch_coords = coords[start_idx:end_idx]
            batch_targets = target_pixels[start_idx:end_idx]
            
            # Forward pass
            batch_predicted = model(batch_coords)
            
            # Compute loss for this batch
            batch_loss = torch.nn.functional.mse_loss(batch_predicted, batch_targets)
            
            # Scale loss by batch proportion and accumulate
            batch_weight = (end_idx - start_idx) / n_pixels
            scaled_loss = batch_loss * batch_weight
            scaled_loss.backward()
            
            total_loss += batch_loss.item() * batch_weight

        # Step optimizer after all batches
        optimizer.step()
        
        iter_end = time.perf_counter()
        total_training_time += (iter_end - iter_start)
        
        # Check if this iteration corresponds to a save point
        if i in calibration_iterations:
            save_idx = calibration_iterations.index(i)
            expected_time_ms = (save_idx + 1) * 250  # +1 because we start from 250ms
            
            path = f"siren_outputs/time_{save_idx + 1:03d}_{expected_time_ms:03d}ms.jpg"
            print(f"Writing '{path}' (iteration {i})... ", end="")
            
            with torch.no_grad():
                # Reconstruct full image in batches
                predicted_full = torch.zeros_like(target_pixels)
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, n_pixels)
                    batch_coords = coords[start_idx:end_idx]
                    predicted_full[start_idx:end_idx] = model(batch_coords)
                
                predicted_image = predicted_full.reshape(height, width, channels)
                predicted_image = torch.clamp(predicted_image, 0, 1)
                write_image(path, predicted_image.cpu().numpy())
            
            print("done.")
        
        i += 1

    total_wall_time = time.perf_counter() - start_time
    print(f"\nTraining completed: Wall time={total_wall_time:.3f}s, Pure training time={total_training_time:.3f}s, {i} iterations")
    print(f"Training efficiency: {total_training_time/total_wall_time*100:.1f}% (rest is I/O overhead)")

    print("All outputs saved in 'siren_outputs/' directory")
    print(f"Images saved at calibrated 250ms intervals (total: {len(calibration_iterations)} saves)")