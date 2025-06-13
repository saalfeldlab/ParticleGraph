#!/usr/bin/env python3
import os
import glob

# Directory with config files
config_dir = '/groups/saalfeld/home/allierc/Py/ParticleGraph/config/signal'

# Find all .bak files
backup_files = glob.glob(os.path.join(config_dir, '*.yaml.bak'))

# Remove each backup file
removed_count = 0
for backup_file in backup_files:
    os.remove(backup_file)
    removed_count += 1
    print(f"Removed: {os.path.basename(backup_file)}")

print(f"Removed {removed_count} backup files.")
