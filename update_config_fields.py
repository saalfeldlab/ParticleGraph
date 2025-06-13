#!/usr/bin/env python3
import os
import yaml
import re

# Directory with config files
config_dir = '/groups/saalfeld/home/allierc/Py/ParticleGraph/config/signal'

# Count of files processed and modified
processed_files = 0
modified_files = 0

# Process each yaml file in the directory
for filename in os.listdir(config_dir):
    if filename.endswith('.yaml'):
        file_path = os.path.join(config_dir, filename)
        processed_files += 1

        # Read the file content
        with open(file_path, 'r') as f:
            content = f.read()

        # Create a backup of the original file
        backup_path = file_path + '.bak'
        with open(backup_path, 'w') as f:
            f.write(content)

        # Replace n_particles with n_neurons and n_particle_types with n_neuron_types
        original_content = content
        content = re.sub(r'n_particles:', r'n_neurons:', content)
        content = re.sub(r'n_particle_types:', r'n_neuron_types:', content)

        # Check if any changes were made
        if content != original_content:
            modified_files += 1
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Updated {filename}")
        else:
            # Remove backup if no changes were made
            os.remove(backup_path)

print(f"Processed {processed_files} files, modified {modified_files} files.")
