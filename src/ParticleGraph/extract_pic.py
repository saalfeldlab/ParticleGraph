import tifffile as tiff
import os
import numpy as np

# File path to the .tif file

if False:
    for channel in range(2):
        input_path = f"/groups/saalfeld/home/allierc/signaling/MDCK/MDCK gCAMP7b 2nd observation/10 sec wound region/41925_mdck gCamp7b_10sec_scan_bidirect_.lif_Region 2_Merged_ch{channel}.tif"\
        # Output directory to save individual frames
        output_dir = f"/groups/saalfeld/home/allierc/signaling/MDCK/MDCK gCAMP7b 2nd observation/10 sec wound region/ch{channel}"
        os.makedirs(output_dir, exist_ok=True)
        print('load the multi-frame TIFF ...')
        images = tiff.imread(input_path)
        print(f'save the {len(images)} frames ...')
        for i in range(len(images)):
            frame = images[i]
            output_path = os.path.join(output_dir, f"frame_{i+1:02d}.tif")
            tiff.imwrite(output_path, frame)

print('get file number ...')
# get get file in the ch0 folder and sort
input_path = "/groups/saalfeld/home/allierc/signaling/MDCK/MDCK gCAMP7b 2nd observation/10 sec wound region/ch0"
output_dir = "/groups/saalfeld/home/allierc/signaling/MDCK/MDCK gCAMP7b 2nd observation/10 sec wound region/ch1"
files = os.listdir(input_path)
# Filter the files to only include .tif files
tif_files = [f for f in files if f.endswith('.tif')]


for file in tif_files:
    print(file)
    # Get the individual frames for each channel
    frame_ch0 = tiff.imread(f"/groups/saalfeld/home/allierc/signaling/MDCK/MDCK gCAMP7b 2nd observation/10 sec wound region/ch0/"+file)  # Blue channel
    frame_ch1 = tiff.imread(f"/groups/saalfeld/home/allierc/signaling/MDCK/MDCK gCAMP7b 2nd observation/10 sec wound region/ch1/"+file)  # Green channel

    # Create a 3-channel (RGB) image by stacking the channels
    # Assuming the frames are 2D arrays, and we add a third (red) channel as zeros
    # print(frame_ch0.shape, frame_ch1.shape, len(frame_ch1.shape))
    if len(frame_ch1.shape) == 3:
        merged_frame = frame_ch1
    else:
        merged_frame = np.stack((frame_ch1, frame_ch0, np.zeros_like(frame_ch0)), axis=-1)
        # Define the output path for the merged RGB frame
        output_path = os.path.join(output_dir, file)
        # Save the merged RGB frame as a TIFF file
        tiff.imwrite(output_path, merged_frame)
