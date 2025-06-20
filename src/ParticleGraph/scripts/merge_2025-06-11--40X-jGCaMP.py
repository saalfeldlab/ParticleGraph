import os
import numpy as np
from PIL import Image
import tifffile
from tqdm import trange

# File paths
path = "/groups/saalfeld/home/allierc/signaling/MDCK/jGcamp7b-mdcks-h42 11sec interval 5hrs/"
blue_path = path + "2025-06-11--40X-H42.tif"
green_path = path + "2025-06-11--40X-jGCamp7b.tif"
output_dir = path + "merged"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# load tiff stacks into memory
print("loading tiff stacks...")
blue_stack = tifffile.imread(blue_path)
green_stack = tifffile.imread(green_path)

print(f"blue stack shape: {blue_stack.shape}")
print(f"green stack shape: {green_stack.shape}")

n_frames = blue_stack.shape[0]
print(f"processing {n_frames} frames...")

for i in range(n_frames):
    # Get individual frames
    blue_frame = blue_stack[i]
    green_frame = green_stack[i]

    # Keep 16-bit data
    red_frame = np.zeros_like(blue_frame)  # No red channel

    # Stack into RGB (16-bit)
    rgb_frame = np.stack([red_frame, green_frame, blue_frame], axis=-1)

    # Save as 16-bit TIF
    output_filename = f"{i:04d}.tif"
    output_path = os.path.join(output_dir, output_filename)
    tifffile.imwrite(output_path, rgb_frame)

    if (i + 1) % 100 == 0:
        print(f"processed {i + 1}/{n_frames} frames")

print("done!")