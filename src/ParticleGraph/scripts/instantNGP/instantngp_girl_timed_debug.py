#!/usr/bin/env python3

# Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# @file   mlp_learning_an_image_pytorch.py
# @author Thomas Müller, NVIDIA
# @brief  Replicates the behavior of the CUDA mlp_learning_an_image.cu sample
#         using tiny-cuda-nn's PyTorch extension. Runs ~2x slower than native.

import argparse
import json
import numpy as np
import os
import sys
import torch
import time
from tqdm import trange

try:
	import tinycudann as tcnn
except ImportError:
	print("This sample requires the tiny-cuda-nn extension for PyTorch.")
	print("You can install it by running:")
	print("============================================================")
	print("tiny-cuda-nn$ cd bindings/torch")
	print("tiny-cuda-nn/bindings/torch$ python setup.py install")
	print("============================================================")
	sys.exit()

from PIL import Image as PILImage
import numpy as np

def read_image(filename):
    """Read image and convert to numpy array with values in [0,1]"""
    img = PILImage.open(filename).convert('RGB')
    return np.array(img).astype(np.float32) / 255.0

def write_image(filename, img_array):
    """Write numpy array to image file"""
    print(f"  DEBUG write_image: Input array shape={img_array.shape}, dtype={img_array.dtype}")
    print(f"  DEBUG write_image: Input range: min={img_array.min():.3f}, max={img_array.max():.3f}")
    
    # Sample a few values to see what they are
    sample_values = img_array[100:103, 100:103]
    print(f"  DEBUG write_image: Sample values:")
    for i in range(sample_values.shape[0]):
        for j in range(sample_values.shape[1]):
            r_val, g_val, b_val = sample_values[i, j]
            print(f"    [{i},{j}]: R={r_val:.3f}, G={g_val:.3f}, B={b_val:.3f}")
    
    img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
    print(f"  DEBUG write_image: After scaling - range: min={img_array.min()}, max={img_array.max()}")
    
    img = PILImage.fromarray(img_array, mode='RGB')  # Explicitly specify RGB mode
    print(f"  DEBUG write_image: PIL image mode: {img.mode}")
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

def get_args():
	parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

	parser.add_argument("image", nargs="?", default="Girl_with_a_Pearl_Earring.jpg", help="Image to match")
	parser.add_argument("config", nargs="?", default="config_hash.json", help="JSON config for tiny-cuda-nn")
	parser.add_argument("n_steps", nargs="?", type=int, default=10000000, help="Number of training steps")
	parser.add_argument("result_filename", nargs="?", default="", help="Number of training steps")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	print("================================================================")
	print("InstantNGP RGB Debug - Girl with a Pearl Earring")
	print("Investigating RGB vs Grayscale issue")
	print("================================================================")

	print(f"Using PyTorch version {torch.__version__} with CUDA {torch.version.cuda}")

	device = torch.device("cuda")
	args = get_args()

	# Get script directory and construct paths
	script_dir = os.path.dirname(os.path.abspath(__file__))
	config_path = os.path.join(script_dir, args.config)
	image_path = os.path.join(script_dir, args.image)

	with open(config_path) as config_file:
		config = json.load(config_file)

	# Load image with resizing for fair comparison
	print(f"Loading image: {image_path}")
	image = Image(image_path, device)
	n_channels = image.data.shape[2]
	
	print(f"DEBUG: Image loaded with shape {image.data.shape} and {n_channels} channels")

	model = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=n_channels, encoding_config=config["encoding"], network_config=config["network"]).to(device)
	
	print(model)
	print("Using modern tiny-cuda-nn with automatic kernel optimization.")

	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

	# Variables for saving/displaying image results
	resolution = image.data.shape[0:2]
	img_shape = resolution + torch.Size([image.data.shape[2]])
	n_pixels = resolution[0] * resolution[1]

	half_dx =  0.5 / resolution[0]
	half_dy =  0.5 / resolution[1]
	xs = torch.linspace(half_dx, 1-half_dx, resolution[0], device=device)
	ys = torch.linspace(half_dy, 1-half_dy, resolution[1], device=device)
	xv, yv = torch.meshgrid([xs, ys])

	xy = torch.stack((yv.flatten(), xv.flatten())).t()

	path = f"reference.jpg"
	print(f"Writing '{path}'... ")
	write_image(path, image(xy).reshape(img_shape).detach().cpu().numpy())
	print("done.")

	batch_size = 2**22  # 4,194,304 - Optimized for RTX A6000 (47.4 GB VRAM)

	print(f"Beginning optimization with {args.n_steps} training steps.")
	print(f"Using optimized batch size: {batch_size:,} samples")
	print(f"Image resolution: {resolution[0]}x{resolution[1]} pixels")

	try:
		batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
		traced_image = torch.jit.trace(image, batch)
	except:
		# If tracing causes an error, fall back to regular execution
		print(f"WARNING: PyTorch JIT trace failed. Performance will be slightly worse than regular.")
		traced_image = image

	# Create output directory and clear it
	import shutil
	if os.path.exists("instantngp_outputs"):
		shutil.rmtree("instantngp_outputs")
	os.makedirs("instantngp_outputs", exist_ok=True)
	print("Cleared and created output directory: instantngp_outputs/")
	
	print("Starting short test (saving after 100 iterations)...")
	
	# Save initial state (t=0)
	path = f"instantngp_outputs/initial.jpg"
	print(f"Writing '{path}'... ")
	with torch.no_grad():
		model_output = model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy()
		print(f"DEBUG: Initial model output shape: {model_output.shape}")
		write_image(path, model_output)
	print("done.")
	
	# Train for 100 iterations
	for i in range(100):
		batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
		targets = traced_image(batch)
		output = model(batch)

		relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
		loss = relative_l2_error.mean()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	# Save after training
	path = f"instantngp_outputs/after_100_iterations.jpg"
	print(f"Writing '{path}'... ")
	with torch.no_grad():
		model_output = model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy()
		print(f"DEBUG: After training model output shape: {model_output.shape}")
		write_image(path, model_output)
	print("done.")

	print("Training completed. Now testing images...")
	
	# Test if images are RGB or grayscale
	print("\n" + "="*60)
	print("RGB vs Grayscale Test Results:")
	print("="*60)
	
	def test_image_is_rgb(filename):
		"""Test if an image is RGB (color) or grayscale with detailed analysis"""
		try:
			img = PILImage.open(filename)
			img_array = np.array(img)
			
			print(f"\n{filename}:")
			print(f"  PIL mode: {img.mode}")
			print(f"  Array shape: {img_array.shape}")
			print(f"  Array dtype: {img_array.dtype}")
			
			if len(img_array.shape) == 3 and img_array.shape[2] == 3:
				# Check if all channels are different (RGB) or same (grayscale)
				r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
				
				# Calculate variance between channels
				rg_diff = np.abs(r.astype(float) - g.astype(float)).mean()
				rb_diff = np.abs(r.astype(float) - b.astype(float)).mean()
				gb_diff = np.abs(g.astype(float) - b.astype(float)).mean()
				
				avg_channel_diff = (rg_diff + rb_diff + gb_diff) / 3
				
				# Sample a few pixel values to see what they look like
				sample_pixels = img_array[100:103, 100:103]  # 3x3 sample
				print(f"  Sample RGB values at (100-103,100-103):")
				for i in range(sample_pixels.shape[0]):
					for j in range(sample_pixels.shape[1]):
						r_val, g_val, b_val = sample_pixels[i, j]
						print(f"    [{i},{j}]: R={r_val}, G={g_val}, B={b_val}")
				
				# Overall statistics
				print(f"  Channel means: R={r.mean():.1f}, G={g.mean():.1f}, B={b.mean():.1f}")
				print(f"  Channel stds: R={r.std():.1f}, G={g.std():.1f}, B={b.std():.1f}")
				print(f"  Channel diffs: RG={rg_diff:.2f}, RB={rb_diff:.2f}, GB={gb_diff:.2f}")
				
				is_rgb = avg_channel_diff > 1.0  # Threshold for color vs grayscale
				
				print(f"  Result: {'RGB' if is_rgb else 'GRAYSCALE'} (avg channel diff: {avg_channel_diff:.2f})")
				return is_rgb
			else:
				print(f"  Result: GRAYSCALE (single channel or wrong shape)")
				return False
		except Exception as e:
			print(f"  ERROR: {e}")
			return False
	
	# Test original image
	test_image_is_rgb("Girl_with_a_Pearl_Earring.jpg")
	
	# Test reference image  
	test_image_is_rgb("reference.jpg")
	
	# Test generated images
	test_image_is_rgb("instantngp_outputs/initial.jpg")
	test_image_is_rgb("instantngp_outputs/after_100_iterations.jpg")
	
	print("="*60)

	tcnn.free_temporary_memory()