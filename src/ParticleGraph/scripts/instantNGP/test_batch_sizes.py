#!/usr/bin/env python3

import torch
import tinycudann as tcnn
import commentjson as json
import os
import gc
import sys

def test_batch_size(model, batch_size, device):
    """Test if a given batch size works without OOM"""
    try:
        # Clear cache before test
        torch.cuda.empty_cache()
        
        # Create test batch
        batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
        
        # Test forward pass
        with torch.no_grad():
            output = model(batch)
        
        # Test backward pass (more memory intensive)
        batch.requires_grad_(True)
        output = model(batch)
        loss = output.sum()
        loss.backward()
        
        del batch, output, loss
        torch.cuda.empty_cache()
        return True
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False
    except Exception as e:
        print(f"Unexpected error at batch size {batch_size}: {e}")
        torch.cuda.empty_cache()
        return False

def find_max_batch_size():
    device = torch.device("cuda")
    
    # Get script directory and load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config_hash.json")
    
    with open(config_path) as config_file:
        config = json.load(config_file)
    
    # Create model
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=2, 
        n_output_dims=3,  # RGB channels
        encoding_config=config["encoding"], 
        network_config=config["network"]
    ).to(device)
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Model: {model}")
    print()
    
    # Binary search for maximum batch size
    min_size = 2**10   # 1,024
    max_size = 2**28   # 268,435,456 (very large upper bound)
    
    print("Testing batch sizes to find maximum...")
    
    # First, find a working upper bound
    current = min_size
    while current <= max_size:
        print(f"Testing batch size: {current:,} (2^{int(torch.log2(torch.tensor(current)).item())})")
        
        if test_batch_size(model, current, device):
            print(f"✓ Batch size {current:,} works")
            min_size = current
            current *= 2
        else:
            print(f"✗ Batch size {current:,} failed (OOM)")
            max_size = current
            break
    
    # Binary search for exact maximum
    print(f"\nBinary search between {min_size:,} and {max_size:,}")
    
    while max_size - min_size > 1:
        mid = (min_size + max_size) // 2
        print(f"Testing batch size: {mid:,}")
        
        if test_batch_size(model, mid, device):
            print(f"✓ Batch size {mid:,} works")
            min_size = mid
        else:
            print(f"✗ Batch size {mid:,} failed (OOM)")
            max_size = mid
    
    optimal_batch_size = min_size
    
    print(f"\n" + "="*50)
    print(f"RESULTS:")
    print(f"Maximum batch size: {optimal_batch_size:,}")
    print(f"Power of 2: 2^{int(torch.log2(torch.tensor(optimal_batch_size)).item())}")
    print(f"Memory usage per sample: ~{torch.cuda.max_memory_allocated() / optimal_batch_size / 1024:.1f} KB")
    print(f"="*50)
    
    return optimal_batch_size

if __name__ == "__main__":
    max_batch_size = find_max_batch_size()
    
    print(f"\nTo use this batch size, change line 160 in mlp_learning_an_image_pytorch.py to:")
    print(f"batch_size = {max_batch_size}")
    
    # Find the closest power of 2
    power = int(torch.log2(torch.tensor(max_batch_size)).item())
    closest_power_of_2 = 2**power
    if max_batch_size != closest_power_of_2:
        print(f"Or use closest power of 2:")
        print(f"batch_size = 2**{power}  # {closest_power_of_2:,}")