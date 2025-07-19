#!/bin/bash

# Method 1: Using a loop (recommended)
echo "Starting batch processing..."

for i in {0..6}
do
    echo "Running: python GNN_particles_Ntype.py -o generate fly_N9_18_4_$i"
    python GNN_particles_Ntype.py -o generate fly_N9_18_4_$i
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully completed fly_N9_18_4_$i"
    else
        echo "Error occurred with fly_N9_18_4_$i"
        # Uncomment the next line if you want to stop on first error
        # exit 1
    fi
    echo "---"
done

echo "All commands completed!"

# Method 2: Explicit commands (alternative approach)
# Uncomment the lines below if you prefer explicit commands instead of the loop

# python GNN_particles_Ntype.py -o generate fly_N9_18_4_0
# python GNN_particles_Ntype.py -o generate fly_N9_18_4_1
# python GNN_particles_Ntype.py -o generate fly_N9_18_4_2
# python GNN_particles_Ntype.py -o generate fly_N9_18_4_3
# python GNN_particles_Ntype.py -o generate fly_N9_18_4_4
# python GNN_particles_Ntype.py -o generate fly_N9_18_4_5
# python GNN_particles_Ntype.py -o generate fly_N9_18_4_6
