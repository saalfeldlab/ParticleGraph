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


def save_experiment_metadata(folders: Dict[str, str], system_params: Dict, training_params: Dict):
    """
    Save experiment configuration and metadata

    Args:
        folders: Dictionary with folder paths
        system_params: FitzHugh-Nagumo system parameters
        training_params: Training configuration parameters
    """

    metadata = {
        'experiment_info': {
            'timestamp': datetime.datetime.now().isoformat(),
            'experiment_type': 'FitzHugh-Nagumo Neural Dynamics',
            'description': 'Hybrid neural ODE training with derivative analysis'
        },
        'system_parameters': system_params,
        'training_parameters': training_params,
        'folder_structure': folders
    }

    metadata_path = os.path.join(folders['data'], 'experiment_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"saved experiment metadata to: {metadata_path}")
    return metadata_path


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


def analyze_training_results(convergence_results: List[Dict], loss_data: Dict = None,
                                    folders: Dict[str, str] = None):
    """
    Simple 2-panel training analysis: Loss + V Rollout RMSE
    """

    if folders is None:
        folders = setup_experiment_folders()

    plt.style.use('dark_background')

    runs = [r['run'] for r in convergence_results]
    v_mses = [r['v_mse'] for r in convergence_results]

    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('FitzHugh-Nagumo Training Analysis', fontsize=16, color='white')

    if loss_data:
        for i, (run_id, loss_progression) in enumerate(loss_data.items()):
            iterations = loss_progression['iterations']
            losses = loss_progression['losses']
            color = colors[i % len(colors)]
            ax1.plot(iterations, losses, color=color, linewidth=1,
                     label=f"Run {run_id}", alpha=0.8)

        ax1.set_xlabel('Iteration', fontsize=12, color='white')
        ax1.set_ylabel('Training Loss', fontsize=12, color='white')
        ax1.set_title('Training Loss Progression', fontsize=14, color='white')
        ax1.set_ylim([0, 6])
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.tick_params(colors='white')

    v_rmse = [np.sqrt(mse) for mse in v_mses]  # Convert MSE to RMSE
    bars = ax2.bar(runs, v_rmse, color=[colors[i % len(colors)] for i in range(len(runs))],
                   alpha=0.8, edgecolor='white', linewidth=1)

    ax2.set_xlabel('Run Number', fontsize=12, color='white')
    ax2.set_ylabel('V Rollout RMSE', fontsize=12, color='white')
    ax2.set_title('V Rollout RMSE by Run', fontsize=14, color='white')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(colors='white')
    ax2.set_ylim([0,0.1])

    plt.tight_layout()

    analysis_plot_path = os.path.join(folders['analysis'], 'comprehensive_training_analysis.png')
    plt.savefig(analysis_plot_path, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"Saved analysis plot: {analysis_plot_path}")
    plt.show()

    return folders


def create_experiment_summary(convergence_results: List[Dict], statistics: Dict, folders: Dict[str, str]):
    """Create a text summary of the experiment"""

    summary_text = f"""
FITZHUGH-NAGUMO EXPERIMENT SUMMARY
{'=' * 50}

Experiment Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Training Runs: {len(convergence_results)}

BEST MODEL PERFORMANCE:
- Run: {statistics['best_run']}
- Total MSE: {statistics['best_total_mse']:.6f}
- V MSE: {statistics['best_v_mse']:.6f}
- W MSE: {statistics['best_w_mse']:.6f}

OVERALL STATISTICS:
- Mean Total MSE: {statistics['total_mse_mean']:.6f} ± {statistics['total_mse_std']:.6f}
- Coefficient of Variation: {statistics['total_mse_cv']:.1f}%
- Success Rate: {statistics['success_rate']}
- Improvement over Mean: {statistics['improvement']:.1f}%

CONVERGENCE RANKING:
"""

    # Add ranking table
    sorted_runs = sorted(convergence_results, key=lambda x: x['total_mse'])
    for rank, run_data in enumerate(sorted_runs, 1):
        status = " ⭐ BEST" if rank == 1 else ""
        summary_text += f"{rank:2d}. Run {run_data['run']}: Total MSE = {run_data['total_mse']:.6f}{status}\n"

    summary_text += f"""
EXPERIMENT FILES:
- Training Results: data/training_results.json
- Comprehensive Analysis: analysis/comprehensive_training_analysis.png
- Loss Progression: plots/training/loss_progression.png
- MSE Analysis: analysis/mse_component_analysis.png
- Experiment Metadata: data/experiment_metadata.json

FOLDER STRUCTURE:
- Base Directory: {folders['base']}
- Plots: {folders['plots']}
- Analysis: {folders['analysis']}
- Data: {folders['data']}
- Models: {folders['models']}
"""

    summary_path = os.path.join(folders['base'], 'EXPERIMENT_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)

    # print(f"Saved experiment summary: {summary_path}")
    # print("\n" + summary_text)


def create_loss_data_from_experiment():
    """
    Create loss progression data structure from your experimental output
    You can modify this function to extract actual loss data from your training loop
    """

    # Example data structure based on your experimental output
    loss_data = {
        1: {
            'iterations': [1, 251, 501, 751, 1001, 1251, 1501, 1751, 2001, 2251, 2501, 2751, 3001, 3251, 3501, 3751,
                           4001, 4251, 4501, 4751],
            'losses': [5.348039, 4.669472, 4.684207, 4.229091, 3.879978, 2.048162, 0.928159, 0.693025, 0.536434,
                       0.494438, 0.538141, 0.494990, 0.540366, 0.585160, 0.468427, 0.425192, 0.452702, 0.602803,
                       0.526291, 0.523909]
        },
        2: {
            'iterations': [1, 251, 501, 751, 1001, 1251, 1501, 1751, 2001, 2251, 2501, 2751, 3001, 3251, 3501, 3751,
                           4001, 4251, 4501, 4751],
            'losses': [4.501005, 4.532033, 4.470904, 4.153150, 4.348590, 3.834573, 3.504427, 4.103400, 3.730602,
                       3.470953, 4.126611, 3.782760, 3.866946, 4.177008, 2.813051, 1.048107, 0.737052, 0.770687,
                       0.708162, 0.705012]
        },
        3: {
            'iterations': [1, 251, 501, 751, 1001, 1251, 1501, 1751, 2001, 2251, 2501, 2751, 3001, 3251, 3501, 3751,
                           4001, 4251, 4501, 4751],
            'losses': [4.807458, 4.470797, 4.456226, 4.021041, 3.239792, 1.095074, 0.910257, 0.727730, 0.654897,
                       0.636889, 0.567622, 0.540963, 0.517973, 0.589268, 0.624875, 0.509821, 0.489480, 0.507765,
                       0.531055, 0.528533]
        },
        4: {
            'iterations': [1, 251, 501, 751, 1001, 1251, 1501, 1751, 2001, 2251, 2501, 2751, 3001, 3251, 3501, 3751,
                           4001, 4251, 4501, 4751],
            'losses': [4.666375, 4.273243, 4.690498, 4.418748, 4.284633, 4.263946, 3.713566, 0.978556, 0.653239,
                       0.484920, 0.626218, 0.534410, 0.528272, 0.507886, 0.457159, 0.446877, 0.390366, 0.575282,
                       0.494907, 0.392585]
        },
        5: {
            'iterations': [1, 251, 501, 751, 1001, 1251, 1501, 1751, 2001, 2251, 2501, 2751, 3001, 3251, 3501, 3751,
                           4001, 4251, 4501, 4751],
            'losses': [4.693977, 4.473523, 4.259524, 4.274893, 4.565036, 4.079438, 4.075711, 4.146191, 3.656261,
                       1.077707, 0.935370, 0.634042, 0.728746, 0.595080, 0.518523, 0.541738, 0.620182, 0.496711,
                       0.597633, 0.513578]
        }
    }

    return loss_data


def run_training_analysis(convergence_results: List[Dict], loss_progression_data: Dict = None,
                          experiment_name: str = None, system_params: Dict = None,
                          training_params: Dict = None):
    """
    Complete training analysis with organized logging

    Args:
        convergence_results: Results from your training runs
        loss_progression_data: Loss data collected during training
        experiment_name: Custom name for this experiment
        system_params: FitzHugh-Nagumo system parameters
        training_params: Training configuration
    """

    # Set up organized folder structure
    print("Setting up experiment folders...")
    folders = setup_experiment_folders(experiment_name=experiment_name)

    # Default system parameters if not provided
    if system_params is None:
        system_params = {
            'a': 0.7,
            'b': 0.8,
            'epsilon': 0.18,
            'T': 1000.0,
            'dt': 0.1,
            'n_steps': 10000
        }

    # Default training parameters if not provided
    if training_params is None:
        training_params = {
            'n_iter': 5000,
            'lr': 0.0001,
            'test_runs': 5,
            'batch_size': 2000,
            'l1_lambda': 1.0E-3,
            'weight_decay': 1e-6,
            'recursive_loop': 3
        }

    # Save experiment metadata
    save_experiment_metadata(folders, system_params, training_params)

    # Run comprehensive analysis
    print("Running comprehensive training analysis...")
    analyze_training_results(convergence_results, loss_progression_data, folders)

    return folders
