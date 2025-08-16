import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import datetime
from typing import List, Dict, Any


def setup_experiment_folders(base_path: str = './logs', experiment_name: str = None) -> Dict[str, str]:
    """
    Create organized folder structure for experiment logging

    Args:
        base_path: Base directory for all logs
        experiment_name: Custom experiment name (if None, uses timestamp)

    Returns:
        Dictionary with folder paths
    """

    # Create timestamp for experiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if experiment_name is None:
        experiment_name = f"fitzhugh_nagumo_{timestamp}"
    else:
        experiment_name = experiment_name

    # Define folder structure
    folders = {
        'base': os.path.join(base_path, experiment_name),
        'plots': os.path.join(base_path, experiment_name, 'plots'),
        'analysis': os.path.join(base_path, experiment_name, 'analysis'),
        'data': os.path.join(base_path, experiment_name, 'data'),
        'models': os.path.join(base_path, experiment_name, 'models'),
        'training_plots': os.path.join(base_path, experiment_name, 'training'),
    }

    # Create all folders
    for folder_name, folder_path in folders.items():
        os.makedirs(folder_path, exist_ok=True)

    return folders


def save_results_to_json(convergence_results: List[Dict], loss_data: Dict,
                         statistics: Dict, folders: Dict[str, str]) -> str:
    """
    Save all training results to JSON file

    Args:
        convergence_results: List of training run results
        loss_data: Loss progression data
        statistics: Calculated statistics
        folders: Dictionary with folder paths

    Returns:
        Path to saved JSON file
    """

    results_data = {
        'training_results': {
            'convergence_summary': convergence_results,
            'loss_progression': loss_data,
            'statistics': statistics,
            'best_model': {
                'run': statistics['best_run'],
                'total_mse': statistics['best_total_mse'],
                'v_mse': statistics['best_v_mse'],
                'w_mse': statistics['best_w_mse']
            }
        },
        'analysis_info': {
            'total_runs': len(convergence_results),
            'success_rate': statistics['success_rate'],
            'improvement_over_mean': f"{statistics['improvement']:.1f}%",
            'stability_cv': f"{statistics['total_mse_cv']:.1f}%"
        }
    }

    results_path = os.path.join(folders['data'], 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=4)

    print(f"Saved training results to: {results_path}")
    return results_path


