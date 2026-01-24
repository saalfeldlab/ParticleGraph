#!/usr/bin/env python3
"""
Standalone data generation script for subprocess execution.

This script is called by GNN_LLM.py as a subprocess to ensure that any code
modifications to PDE_Diffusiophoresis.py, PDE_D.py, or graph_data_generator.py
are reloaded for each iteration.

Usage:
    python generate_subprocess.py --config CONFIG_PATH --device DEVICE [--erase] [--log_file LOG_PATH]
"""

import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports

import argparse
import sys
import os
import traceback

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators.graph_data_generator import data_generate
from ParticleGraph.utils import set_device, add_pre_folder


def main():
    parser = argparse.ArgumentParser(description='Generate simulation data')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--erase', action='store_true', help='Erase existing data files')
    parser.add_argument('--log_file', type=str, default=None, help='Path to analysis log file')
    parser.add_argument('--config_file', type=str, default=None, help='Config file name for log directory')
    parser.add_argument('--error_log', type=str, default=None, help='Path to error log file')
    parser.add_argument('--visualize', action='store_true', default=True, help='Generate visualizations')
    parser.add_argument('--step', type=int, default=1, help='Visualization step')

    args = parser.parse_args()

    # Open error log file if specified
    error_log = None
    if args.error_log:
        try:
            error_log = open(args.error_log, 'w')
        except Exception as e:
            print(f"Warning: Could not open error log file: {e}", file=sys.stderr)

    try:
        # Load config
        config = ParticleGraphConfig.from_yaml(args.config)

        # Add folder prefix to dataset (matches GNN_Main.py behavior)
        # e.g., "diffusiophoresis_Claude" -> "diffusiophoresis/diffusiophoresis_Claude"
        config_basename = os.path.basename(args.config).replace('.yaml', '')
        _, pre_folder = add_pre_folder(config_basename)
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_basename

        # Set device
        device = set_device(args.device)

        # Open log file if specified
        log_file = None
        if args.log_file:
            log_file = open(args.log_file, 'w')

        try:
            # Run data generation - this will reload any modified code
            data_generate(
                config=config,
                visualize=args.visualize,
                run_vizualized=0,
                style='color',
                erase=args.erase,
                step=args.step,
                alpha=0.2,
                ratio=1,
                scenario='none',
                best_model=None,
                device=device,
                bSave=True,
            )

            print("\033[92mdata generation completed successfully\033[0m")

            # Force cleanup to prevent hanging from background threads
            import matplotlib.pyplot as plt
            plt.close('all')

            # Force exit - some imports (umap, torch_geometric) spawn threads
            import gc
            gc.collect()

        finally:
            if log_file:
                log_file.close()

    except Exception as e:
        # Capture full traceback for debugging
        error_msg = f"\n{'='*80}\n"
        error_msg += "GENERATION SUBPROCESS ERROR\n"
        error_msg += f"{'='*80}\n\n"
        error_msg += f"Error Type: {type(e).__name__}\n"
        error_msg += f"Error Message: {str(e)}\n\n"
        error_msg += "Full Traceback:\n"
        error_msg += traceback.format_exc()
        error_msg += f"\n{'='*80}\n"

        # Print to stderr
        print(error_msg, file=sys.stderr, flush=True)

        # Write to error log if available
        if error_log:
            error_log.write(error_msg)
            error_log.flush()

        # Exit with non-zero code
        sys.exit(1)

    finally:
        if error_log:
            error_log.close()

    # force immediate exit - bypasses thread cleanup that can hang
    # os._exit is needed because libraries like umap/torch_geometric spawn daemon threads
    print("\033[92mexiting subprocess...\033[0m", flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
