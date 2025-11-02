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

# @file   instantngp_girl_timed.py
# @author Based on Thomas Müller's mlp_learning_an_image_pytorch.py
# @brief  Time-based image reconstruction for performance comparison

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
	print("InstantNGP Performance Test - Girl with a Pearl Earring")
	print("Time-based Image Reconstruction (250ms intervals)")
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

	# Load image
	print(f"Loading image: {image_path}")
	image = Image(image_path, device)
	n_channels = image.data.shape[2]

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

	path = f"reference.png"
	print(f"Writing '{path}'... ", end="")
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
	
	print("PHASE 1: Calibration - measuring iterations per 250ms without I/O...")
	
	# Calibration phase - pure training for 10 seconds to measure iteration rate
	calibration_start = time.perf_counter()
	calibration_iterations = []
	calibration_times = []
	i = 0
	
	while time.perf_counter() - calibration_start < 10.0:
		batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
		targets = traced_image(batch)
		output = model(batch)

		relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
		loss = relative_l2_error.mean()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		current_time = time.perf_counter()
		elapsed_time = current_time - calibration_start
		
		# Record every 250ms
		if len(calibration_times) == 0 or elapsed_time >= (len(calibration_times) * 0.25):
			calibration_iterations.append(i)
			calibration_times.append(elapsed_time)
			print(f"Calibration: {elapsed_time:.3f}s = iteration {i}")
		
		i += 1
	
	print(f"Calibration completed: {i} iterations in 10 seconds")
	print(f"Save points (250ms intervals): {calibration_iterations}")
	
	print("\nPHASE 2: Training with iteration-based saving...")
	
	# Reset model for actual training
	model = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=n_channels, encoding_config=config["encoding"], network_config=config["network"]).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
	
	start_time = time.perf_counter()
	total_training_time = 0.0
	save_counter = 0
	i = 0
	
	# Save initial state (t=0)
	path = f"instantngp_outputs/time_000_000ms.png"
	print(f"Writing '{path}'... ", end="")
	with torch.no_grad():
		write_image(path, model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
	print("done.")
	save_counter += 1
	
	# Training loop with iteration-based saving
	max_iterations = calibration_iterations[-1] if calibration_iterations else 1000
	
	while i <= max_iterations:
		# Pure training step
		train_start = time.perf_counter()
		
		batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
		targets = traced_image(batch)
		output = model(batch)

		relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
		loss = relative_l2_error.mean()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		total_training_time += time.perf_counter() - train_start
		
		# Check if we hit a save point
		if save_counter < len(calibration_iterations) and i == calibration_iterations[save_counter]:
			wall_time = time.perf_counter() - start_time
			expected_ms = save_counter * 250
			
			print(f"Iteration {i}: Expected {expected_ms}ms, Training time: {total_training_time:.3f}s, loss={loss.item():.6f}")
			
			# Save image
			path = f"instantngp_outputs/time_{save_counter:03d}_{expected_ms:04d}ms.png"
			print(f"Writing '{path}'... ", end="")
			with torch.no_grad():
				write_image(path, model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
			print("done.")
			
			save_counter += 1
		
		i += 1

	total_wall_time = time.perf_counter() - start_time
	print(f"\n================================================================")
	print(f"TRAINING COMPLETED")
	print(f"================================================================")
	print(f"Wall time: {total_wall_time:.3f}s")
	print(f"Pure training time: {total_training_time:.3f}s") 
	print(f"Total iterations: {i}")
	print(f"Training efficiency: {total_training_time/total_wall_time*100:.1f}% (rest is I/O overhead)")
	print(f"Images saved: {save_counter} (every 250ms from 0ms to {int((save_counter-1)*250)}ms)")
	print(f"Output directory: instantngp_outputs/")
	print(f"================================================================")

	if args.result_filename:
		print(f"Writing '{args.result_filename}'... ", end="")
		with torch.no_grad():
			write_image(args.result_filename, model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
		print("done.")

	tcnn.free_temporary_memory()