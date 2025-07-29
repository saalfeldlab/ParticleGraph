#!/usr/bin/env python3
"""
Script to create a standalone MPM simulation repository
This script copies all necessary files for Material Point Method (MPM) simulations
from the current ParticleGraph project to a new './MPM' directory.
"""

import os
import shutil
import sys
from pathlib import Path

def create_directory_structure():
    """Create the basic directory structure for the MPM repo"""
    directories = [
        'MPM',
        'MPM/src',
        'MPM/src/ParticleGraph',
        'MPM/src/ParticleGraph/models',
        'MPM/src/ParticleGraph/generators',
        'MPM/src/ParticleGraph/taichi',
        'MPM/config',
        'MPM/config/multimaterial',
        'MPM/graphs_data',
        'MPM/log',
        'MPM/ParticleGraph',
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def copy_file_safe(src, dst):
    """Safely copy a file, creating directories if needed"""
    try:
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")
        return True
    except FileNotFoundError:
        print(f"Warning: Source file not found: {src}")
        return False
    except Exception as e:
        print(f"Error copying {src}: {e}")
        return False

def copy_directory_safe(src, dst):
    """Safely copy a directory"""
    try:
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"Copied directory: {src} -> {dst}")
            return True
        else:
            print(f"Warning: Source directory not found: {src}")
            return False
    except Exception as e:
        print(f"Error copying directory {src}: {e}")
        return False

def main():
    """Main function to create the MPM repository"""

    print("Creating MPM simulation repository...")
    print("=" * 50)

    # Create directory structure
    create_directory_structure()

    # Define file mappings: (source, destination)
    core_files = [
        # Core MPM implementation files
        ('src/ParticleGraph/models/Interaction_MPM.py', 'MPM/src/ParticleGraph/models/Interaction_MPM.py'),
        ('src/ParticleGraph/generators/MPM_step.py', 'MPM/src/ParticleGraph/generators/MPM_step.py'),
        ('src/ParticleGraph/generators/MPM_3D_step.py', 'MPM/src/ParticleGraph/generators/MPM_3D_step.py'),
        ('src/ParticleGraph/generators/MPM_P2G.py', 'MPM/src/ParticleGraph/generators/MPM_P2G.py'),
        ('src/ParticleGraph/generators/MPM_3D_P2G.py', 'MPM/src/ParticleGraph/generators/MPM_3D_P2G.py'),

        # Data generation and utilities
        ('src/ParticleGraph/generators/graph_data_generator.py', 'MPM/src/ParticleGraph/generators/graph_data_generator.py'),
        ('src/ParticleGraph/generators/utils.py', 'MPM/src/ParticleGraph/generators/utils.py'),

        # Model framework files
        ('src/ParticleGraph/models/utils.py', 'MPM/src/ParticleGraph/models/utils.py'),
        ('src/ParticleGraph/models/graph_trainer.py', 'MPM/src/ParticleGraph/models/graph_trainer.py'),
        ('src/ParticleGraph/models/__init__.py', 'MPM/src/ParticleGraph/models/__init__.py'),
        ('src/ParticleGraph/models/MLP.py', 'MPM/src/ParticleGraph/models/MLP.py'),
        ('src/ParticleGraph/models/Siren_Network.py', 'MPM/src/ParticleGraph/models/Siren_Network.py'),
        ('src/ParticleGraph/models/Gumbel.py', 'MPM/src/ParticleGraph/models/Gumbel.py'),

        # Taichi integration
        ('src/ParticleGraph/taichi/mpm3d.py', 'MPM/src/ParticleGraph/taichi/mpm3d.py'),
        ('src/ParticleGraph/taichi/mpm128.py', 'MPM/src/ParticleGraph/taichi/mpm128.py'),

        # Core utilities
        ('src/ParticleGraph/utils.py', 'MPM/src/ParticleGraph/utils.py'),
        ('src/ParticleGraph/__init__.py', 'MPM/src/ParticleGraph/__init__.py'),
        ('src/ParticleGraph/generators/__init__.py', 'MPM/src/ParticleGraph/generators/__init__.py'),
        ('ParticleGraph/plot_utils.py', 'MPM/ParticleGraph/plot_utils.py'),

        # Main execution scripts
        ('GNN_particles_Ntype.py', 'MPM/GNN_particles_Ntype.py'),
        ('GNN_particles_Ntype_pipeline.py', 'MPM/GNN_particles_Ntype_pipeline.py'),
        ('test.py', 'MPM/test.py'),

        # Project setup files
        ('setup.py', 'MPM/setup.py'),
        ('environment.yaml', 'MPM/environment.yaml'),
        ('env.yaml', 'MPM/env.yaml'),
        ('README.md', 'MPM/README.md'),
        ('LICENSE', 'MPM/LICENSE'),
    ]

    print("\nCopying core files...")
    print("-" * 30)

    # Copy core files
    for src, dst in core_files:
        copy_file_safe(src, dst)

    # Copy entire multimaterial config directory
    print("\nCopying configuration files...")
    print("-" * 30)
    copy_directory_safe('config/multimaterial', 'MPM/config/multimaterial')

    # Copy any additional config files that might be relevant
    additional_configs = [
        'config/test_smooth_particle.yaml',
    ]

    for config_file in additional_configs:
        if os.path.exists(config_file):
            copy_file_safe(config_file, f'MPM/{config_file}')

    # Create a simplified requirements.txt for MPM
    print("\nCreating MPM-specific files...")
    print("-" * 30)

    mpm_requirements = """# MPM Simulation Requirements
torch>=1.9.0
torch-geometric>=2.0.0
numpy
matplotlib
pyvista
taichi
PyYAML
scipy
tqdm
"""

    with open('MPM/requirements.txt', 'w') as f:
        f.write(mpm_requirements)
    print("Created: MPM/requirements.txt")

    # Create a simplified setup script for MPM
    mpm_setup = """#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="MPM_Simulation",
    version="1.0.0",
    description="Material Point Method (MPM) Simulation Framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torch-geometric>=2.0.0",
        "numpy",
        "matplotlib",
        "pyvista",
        "taichi",
        "PyYAML",
        "scipy",
        "tqdm",
    ],
    author="MPM Simulation Team",
    license="MIT",
)
"""

    with open('MPM/setup_mpm.py', 'w') as f:
        f.write(mpm_setup)
    print("Created: MPM/setup_mpm.py")

    # Create MPM-specific README
    mpm_readme = """# MPM Simulation Framework

A standalone Material Point Method (MPM) simulation framework extracted from ParticleGraph.

## Features

- 2D and 3D MPM simulations
- Multiple material types (liquid, jelly, snow)
- PyTorch and Taichi backends
- Visualization with PyVista
- Configurable simulation parameters

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # or
   pip install -e .
   ```
3. For Taichi GPU acceleration:
   ```bash
   pip install taichi[cuda]  # for NVIDIA GPUs
   pip install taichi[vulkan]  # for other GPUs
   ```

## Quick Start

Run a 2D MPM simulation:
```bash
python GNN_particles_Ntype.py --config config/multimaterial/multimaterial_1_1.yaml
```

Run a 3D MPM simulation:
```bash
python GNN_particles_Ntype.py --config config/multimaterial/multimaterial_2_1.yaml
```

## Configuration

Simulation parameters are configured through YAML files in the `config/multimaterial/` directory.

Key parameters:
- `n_particles`: Number of particles
- `n_grid`: Grid resolution
- `n_particle_types`: Number of material types
- `delta_t`: Time step
- `MPM_gravity`: Gravity force
- `MPM_friction`: Friction coefficient

## Visualization

The framework supports multiple visualization backends:
- PyVista for 3D rendering
- Matplotlib for 2D plots
- Real-time Taichi GUI

## Examples

See the `config/multimaterial/` directory for various simulation setups:
- Basic falling particles
- Multi-material interactions
- Different boundary conditions
"""

    with open('MPM/README_MPM.md', 'w') as f:
        f.write(mpm_readme)
    print("Created: MPM/README_MPM.md")

    # Create a simple run script
    run_script = """#!/bin/bash
# Simple script to run MPM simulations

echo "MPM Simulation Framework"
echo "======================="

# Check if config file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 config/multimaterial/multimaterial_1_1.yaml"
    exit 1
fi

CONFIG_FILE=$1

echo "Running MPM simulation with config: $CONFIG_FILE"
python GNN_particles_Ntype.py --config $CONFIG_FILE
"""

    with open('MPM/run_mpm.sh', 'w') as f:
        f.write(run_script)
    os.chmod('MPM/run_mpm.sh', 0o755)
    print("Created: MPM/run_mpm.sh")

    print("\n" + "=" * 50)
    print("MPM repository creation completed!")
    print(f"Location: {os.path.abspath('MPM')}")
    print("\nNext steps:")
    print("1. cd MPM")
    print("2. pip install -r requirements.txt")
    print("3. python setup_mpm.py install")
    print("4. ./run_mpm.sh config/multimaterial/multimaterial_1_1.yaml")
    print("\nFor 3D simulations, try:")
    print("./run_mpm.sh config/multimaterial/multimaterial_2_1.yaml")

if __name__ == "__main__":
    main()
