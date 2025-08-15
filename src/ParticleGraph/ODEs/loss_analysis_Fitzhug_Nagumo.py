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
                             folders: Dict[str, str] = None, save_individual_plots: bool = True):
    """
    Comprehensive analysis of FitzHugh-Nagumo training results

    Args:
        convergence_results: List of dictionaries containing run results
        loss_data: Optional dictionary with iteration-wise loss data for each run
        folders: Dictionary with organized folder paths
        save_individual_plots: Whether to save individual component plots
    """

    if folders is None:
        folders = setup_experiment_folders()

    plt.style.use('dark_background')

    # Extract data from convergence results
    runs = [r['run'] for r in convergence_results]
    losses = [r['loss'] for r in convergence_results]
    v_mses = [r['v_mse'] for r in convergence_results]
    w_mses = [r['w_mse'] for r in convergence_results]
    total_mses = [r['total_mse'] for r in convergence_results]

    # Define consistent colors for each run
    colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6',
              '#06B6D4', '#EC4899', '#84CC16', '#F97316', '#6366F1',
              '#14B8A6', '#F59E0B', '#8B5A2B', '#22D3EE', '#A855F7']
    run_colors = {run: colors[i] for i, run in enumerate(runs)}

    # Calculate statistics
    statistics = calculate_statistics(convergence_results)

    # Save results to JSON
    save_results_to_json(convergence_results, loss_data, statistics, folders)

    # Create comprehensive analysis figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('FitzHugh-Nagumo Training Analysis', fontsize=20, color='white', y=0.98)

    # 1. Loss Progression Plot (if loss_data provided)
    if loss_data:
        ax1 = plt.subplot(2, 3, (1, 2))

        for run_id, loss_progression in loss_data.items():
            iterations = loss_progression['iterations']
            losses_iter = loss_progression['losses']
            label = f"Run {run_id}"
            if run_id == get_best_run(convergence_results):
                label += " (Best)"
                linewidth = 3
            else:
                linewidth = 2

            ax1.plot(iterations, losses_iter, color=run_colors[run_id],
                     linewidth=linewidth, label=label, alpha=0.8)

        ax1.set_xlabel('Iteration', fontsize=12, color='white')
        ax1.set_ylabel('Training Loss', fontsize=12, color='white')
        ax1.set_ylim([0, 6])
        ax1.set_title('Training Loss Progression (All Runs)', fontsize=14, color='white')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.tick_params(colors='white')

        # Save individual loss progression plot
        if save_individual_plots:
            fig_loss = plt.figure(figsize=(12, 8))
            for run_id, loss_progression in loss_data.items():
                iterations = loss_progression['iterations']
                losses_iter = loss_progression['losses']
                label = f"Run {run_id}"
                if run_id == get_best_run(convergence_results):
                    label += " (Best)"
                    linewidth = 3
                else:
                    linewidth = 2

                plt.plot(iterations, losses_iter, color=run_colors[run_id],
                         linewidth=linewidth, label=label, alpha=0.8)

            plt.xlabel('Iteration', fontsize=14, color='white')
            plt.ylabel('Training Loss', fontsize=14, color='white')
            plt.ylim([0, 6])
            plt.title('FitzHugh-Nagumo Training Loss Progression', fontsize=16, color='white')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            plt.tick_params(colors='white')

            loss_plot_path = os.path.join(folders['training_plots'], 'loss_progression.png')
            plt.savefig(loss_plot_path, dpi=200, bbox_inches='tight', facecolor='black')
            plt.close(fig_loss)
            print(f"Saved loss progression plot: {loss_plot_path}")

    # 2. MSE Component Analysis (Run numbers on x-axis)
    ax2 = plt.subplot(2, 3, 3)

    # Plot V MSE, W MSE, and Total MSE for each run
    ax2.scatter(runs, v_mses, c='#3B82F6', s=100, alpha=0.8, label='V MSE', edgecolors='white', linewidth=1)
    ax2.scatter(runs, w_mses, c='#EF4444', s=100, alpha=0.8, label='W MSE', edgecolors='white', linewidth=1)
    ax2.scatter(runs, total_mses, c='#10B981', s=150, alpha=0.8, label='Total MSE', edgecolors='white', linewidth=2)

    # Highlight best run
    best_run = get_best_run(convergence_results)
    best_idx = runs.index(best_run)
    ax2.scatter(best_run, v_mses[best_idx], c='#3B82F6', s=200, alpha=1.0,
                edgecolors='yellow', linewidth=3, marker='s')
    ax2.scatter(best_run, w_mses[best_idx], c='#EF4444', s=200, alpha=1.0,
                edgecolors='yellow', linewidth=3, marker='s')
    ax2.scatter(best_run, total_mses[best_idx], c='#10B981', s=250, alpha=1.0,
                edgecolors='yellow', linewidth=3, marker='s')

    ax2.set_xlabel('Run Number', fontsize=12, color='white')
    ax2.set_ylabel('MSE', fontsize=12, color='white')
    ax2.set_title('MSE Component Analysis', fontsize=14, color='white')
    ax2.set_xticks(runs)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.tick_params(colors='white')

    # Add text annotations for best run
    ax2.annotate(f'Best Run ({best_run})', xy=(best_run, total_mses[best_idx]),
                 xytext=(best_run + 0.3, total_mses[best_idx] + 0.05),
                 arrowprops=dict(arrowstyle='->', color='yellow', lw=2),
                 fontsize=10, color='yellow', weight='bold')

    # 3. Performance Summary Bar Chart
    ax3 = plt.subplot(2, 3, 4)

    x_pos = np.arange(len(runs))
    bars = ax3.bar(x_pos, total_mses, color=[run_colors[run] for run in runs], alpha=0.8)

    # Highlight best run
    best_run_idx = np.argmin(total_mses)
    bars[best_run_idx].set_edgecolor('yellow')
    bars[best_run_idx].set_linewidth(3)

    ax3.set_xlabel('Run', fontsize=12, color='white')
    ax3.set_ylabel('Total MSE', fontsize=12, color='white')
    ax3.set_title('Total MSE by Run', fontsize=14, color='white')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'Run {run}' for run in runs], rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(colors='white')

    # Add value labels on bars
    for bar, mse in zip(bars, total_mses):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{mse:.3f}', ha='center', va='bottom', color='white', fontsize=9)

    # 4. V vs W MSE Breakdown
    ax4 = plt.subplot(2, 3, 5)

    x_pos = np.arange(len(runs))
    width = 0.35

    bars1 = ax4.bar(x_pos - width / 2, v_mses, width, label='V MSE', color='#3B82F6', alpha=0.8)
    bars2 = ax4.bar(x_pos + width / 2, w_mses, width, label='W MSE', color='#EF4444', alpha=0.8)

    ax4.set_xlabel('Run', fontsize=12, color='white')
    ax4.set_ylabel('MSE', fontsize=12, color='white')
    ax4.set_title('V vs W MSE Components', fontsize=14, color='white')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'Run {run}' for run in runs])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(colors='white')

    # 5. Statistics Summary
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis('off')

    stats_text = f"""
TRAINING STATISTICS

Total MSE:
  Mean: {statistics['total_mse_mean']:.6f}
  Std:  {statistics['total_mse_std']:.6f}
  CV:   {statistics['total_mse_cv']:.1f}%

Training Loss:
  Mean: {statistics['loss_mean']:.6f}
  Std:  {statistics['loss_std']:.6f}
  CV:   {statistics['loss_cv']:.1f}%

BEST MODEL (Run {statistics['best_run']}):
  Total MSE: {statistics['best_total_mse']:.6f}
  V MSE:     {statistics['best_v_mse']:.6f}
  W MSE:     {statistics['best_w_mse']:.6f}

CONVERGENCE:
  Success Rate: {statistics['success_rate']}
  Improvement: {statistics['improvement']:.1f}%
    """

    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', color='white', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))

    plt.tight_layout()

    # Save comprehensive analysis plot
    analysis_plot_path = os.path.join(folders['analysis'], 'comprehensive_training_analysis.png')
    plt.savefig(analysis_plot_path, dpi=200, bbox_inches='tight', facecolor='black')
    print(f"Saved comprehensive analysis plot: {analysis_plot_path}")
    plt.show()

    # Create summary report
    create_experiment_summary(convergence_results, statistics, folders)

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


def get_best_run(convergence_results: List[Dict]) -> int:
    """Find the run with the lowest total MSE"""
    best_idx = np.argmin([r['total_mse'] for r in convergence_results])
    return convergence_results[best_idx]['run']


def calculate_statistics(convergence_results: List[Dict]) -> Dict:
    """Calculate comprehensive statistics from convergence results"""
    total_mses = [r['total_mse'] for r in convergence_results]
    losses = [r['loss'] for r in convergence_results]

    best_run_idx = np.argmin(total_mses)
    best_run = convergence_results[best_run_idx]

    stats = {
        'total_mse_mean': np.mean(total_mses),
        'total_mse_std': np.std(total_mses),
        'total_mse_cv': (np.std(total_mses) / np.mean(total_mses)) * 100,
        'loss_mean': np.mean(losses),
        'loss_std': np.std(losses),
        'loss_cv': (np.std(losses) / np.mean(losses)) * 100,
        'best_run': best_run['run'],
        'best_total_mse': best_run['total_mse'],
        'best_v_mse': best_run['v_mse'],
        'best_w_mse': best_run['w_mse'],
        'success_rate': f"{len(convergence_results)}/{len(convergence_results)}",
        'improvement': ((np.mean(total_mses) - best_run['total_mse']) / np.mean(total_mses)) * 100
    }

    return stats


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


# Example usage function to add to your main training script
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


# Modified integration example for your main training script
def integrate_with_training_script():
    """
    Example showing how to integrate the analysis into your main training script
    Add this to your Fitzhug_Nagumo.py file
    """

    # This goes in your main training section
    example_integration = '''

# Add this at the beginning of your script after imports
from training_analysis import (
    setup_experiment_folders, save_experiment_metadata, 
    analyze_training_results, run_training_analysis
)

# Modify your training loop to collect loss data
def main_training_with_analysis():

    device = set_device('auto')
    print(f'device  {device}')

    # System parameters
    system_params = {
        'a': 0.7,
        'b': 0.8,
        'epsilon': 0.18,
        'T': 1000.0,
        'dt': 0.1,
        'n_steps': 10000
    }

    # Training parameters
    training_params = {
        'n_iter': 5000,
        'lr': 1e-4,
        'test_runs': 5,
        'l1_lambda': 1.0E-3,
        'weight_decay': 1e-6,
        'recursive_loop': 3
    }

    # Set up experiment folders
    folders = setup_experiment_folders(experiment_name="fitzhugh_nagumo_experiment")
    save_experiment_metadata(folders, system_params, training_params)

    # Initialize data collection
    convergence_results = []
    loss_progression_data = {}

    # Training loop
    for run in range(training_params['test_runs']):
        print(f"training run {run+1}/{training_params['test_runs']}")

        # Initialize loss tracking for this run
        loss_progression_data[run+1] = {
            'iterations': [],
            'losses': []
        }

        model = model_duo(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_params['lr'])

        loss_list = []

        for iter in range(training_params['n_iter']):
            # ... your existing training code ...

            # Collect loss data every 250 iterations
            if iter % 250 == 0:
                loss_progression_data[run+1]['iterations'].append(iter+1)
                loss_progression_data[run+1]['losses'].append(loss.item())
                print(f"iteration {iter+1}/{training_params['n_iter']}, loss: {loss.item():.6f}")

            # Your training step code here...

        # After training completion for this run
        # Calculate final MSEs and add to convergence_results
        # ... your existing rollout analysis code ...

        convergence_results.append({
            'run': run+1,
            'iteration': iter,
            'loss': loss.item(),
            'v_mse': v_mse,
            'w_mse': w_mse,
            'total_mse': total_mse
        })

        # Save individual training run plots
        # Save model checkpoint
        model_path = os.path.join(folders['models'], f'model_run_{run+1}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'run': run+1,
            'iteration': iter,
            'loss': loss.item(),
            'v_mse': v_mse,
            'w_mse': w_mse,
            'total_mse': total_mse
        }, model_path)
        print(f"Saved model: {model_path}")

    # Run comprehensive analysis at the end
    print("\\n" + "="*60)
    print("RUNNING COMPREHENSIVE TRAINING ANALYSIS")
    print("="*60)

    final_folders = analyze_training_results(convergence_results, loss_progression_data, folders)

    # Print final summary
    print(f"\\nExperiment completed successfully!")
    print(f"Results saved in: {final_folders['base']}")
    print(f"View comprehensive analysis: {final_folders['analysis']}/comprehensive_training_analysis.png")

    return final_folders, convergence_results

if __name__ == '__main__':
    folders, results = main_training_with_analysis()
    '''

    return example_integration