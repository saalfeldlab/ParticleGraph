"""
LLM-Guided Diffusiophoresis Pattern Exploration

This script implements an LLM-guided exploration loop for finding
biologically interesting patterns in diffusiophoresis simulations.

Philosophy:
- Experiment = Generate simulation (movie)
- LLM = Claude rates pattern complexity 0-10
- Memory = Accumulates understanding of what produces interesting patterns

Usage:
    python GNN_LLM.py -o Claude diffusiophoresis iterations=64
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ParticleGraph.config import ParticleGraphConfig
# Note: data_generate is called via generate_subprocess.py for clean code reloading


def is_git_repo(root_dir: str) -> bool:
    """Check if directory is a git repository."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_modified_code_files(root_dir: str, tracked_files: list) -> list:
    """Check which tracked code files have been modified."""
    modified = []
    for file_path in tracked_files:
        try:
            result = subprocess.run(
                ['git', 'diff', '--quiet', file_path],
                cwd=root_dir,
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                modified.append(file_path)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return modified


def commit_code_modification(root_dir: str, file_path: str, iteration: int,
                              description: str = None) -> tuple:
    """Commit a code modification to git."""
    if not is_git_repo(root_dir):
        return False, "Not a git repository"

    if description is None:
        description = f"Code modification in {os.path.basename(file_path)}"

    try:
        # Stage the file
        subprocess.run(['git', 'add', file_path], cwd=root_dir, timeout=10)

        # Commit
        commit_msg = f"[Iter {iteration}] {description}\n\n[Automated commit by Claude]"
        result = subprocess.run(
            ['git', 'commit', '-m', commit_msg],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            if 'nothing to commit' in result.stdout.lower():
                return True, "No changes to commit"
            return False, f"Commit failed: {result.stderr}"

        return True, f"Committed: {os.path.basename(file_path)}"
    except subprocess.TimeoutExpired:
        return False, "Git timeout"
    except Exception as e:
        return False, str(e)


def track_code_modifications(root_dir: str, iteration: int) -> list:
    """Check for and commit any code modifications."""
    if not is_git_repo(root_dir):
        return []

    # Base files that Claude can modify
    code_files = [
        'src/ParticleGraph/generators/PDE_D.py',
        'src/ParticleGraph/generators/graph_data_generator.py',
        'src/ParticleGraph/generators/utils.py',
    ]

    # Add all PDE_Diffusiophoresis*.py files (base + variants)
    generators_dir = Path(root_dir) / 'src/ParticleGraph/generators'
    for pde_file in generators_dir.glob('PDE_Diffusiophoresis*.py'):
        rel_path = str(pde_file.relative_to(root_dir))
        if rel_path not in code_files:
            code_files.append(rel_path)

    # Add all PDE_D_*.py variant files (e.g., PDE_D_Boids.py, PDE_D_Chemotaxis.py)
    for pde_d_file in generators_dir.glob('PDE_D_*.py'):
        rel_path = str(pde_d_file.relative_to(root_dir))
        if rel_path not in code_files:
            code_files.append(rel_path)

    # Check for modified tracked files
    modified = get_modified_code_files(root_dir, code_files)

    # Also check for untracked (new) files in generators directory
    try:
        result = subprocess.run(
            ['git', 'ls-files', '--others', '--exclude-standard', 'src/ParticleGraph/generators/'],
            cwd=root_dir, capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line and line.endswith('.py') and line not in modified:
                    modified.append(line)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    results = []
    for file_path in modified:
        success, message = commit_code_modification(root_dir, file_path, iteration)
        results.append((file_path, success, message))

    return results


def create_frame_montage(fig_dir: str, output_path: str, n_frames: int = 10):
    """Create a montage of sample frames from the simulation."""
    # Find all PNG files
    png_files = sorted(Path(fig_dir).glob("Fig_0_*.png"))

    if not png_files:
        print(f"No PNG files found in {fig_dir}")
        return False

    # Select evenly spaced frames
    total_frames = len(png_files)
    if total_frames < n_frames:
        indices = list(range(total_frames))
    else:
        indices = [int(i * total_frames / n_frames) for i in range(n_frames)]

    selected_files = [png_files[i] for i in indices]

    # Load images
    images = [Image.open(f) for f in selected_files]

    # Create 2x5 montage
    cols, rows = 5, 2
    img_width, img_height = images[0].size

    # Scale down if too large
    max_width = 400
    if img_width > max_width:
        scale = max_width / img_width
        img_width = int(img_width * scale)
        img_height = int(img_height * scale)
        images = [img.resize((img_width, img_height), Image.LANCZOS) for img in images]

    montage = Image.new('RGB', (cols * img_width, rows * img_height))

    for idx, img in enumerate(images[:n_frames]):
        row = idx // cols
        col = idx % cols
        montage.paste(img, (col * img_width, row * img_height))

    montage.save(output_path)
    print(f"Created montage: {output_path}")
    return True


def compute_ucb_scores(analysis_path: str, ucb_path: str, c: float = 1.414,
                       current_log_path: str = None, current_iteration: int = None,
                       block_size: int = 8):
    """
    Parse analysis file, build exploration tree, compute UCB scores.
    For pattern exploration, we use the 0-10 score normalized to 0-1.
    """
    nodes = {}
    next_parent_map = {}

    # Parse previous iterations from analysis markdown file
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            content = f.read()

        current_node = None
        lines = content.split('\n')

        for line in lines:
            # Match iteration header
            iter_match = re.match(r'##+ Iter (\d+):', line)
            if iter_match:
                if current_node is not None and 'id' in current_node and 'score' in current_node:
                    nodes[current_node['id']] = current_node
                current_iter = int(iter_match.group(1))
                current_node = {'iter': current_iter}
                continue

            # Match Node line
            node_match = re.match(r'Node: id=(\d+), parent=(\d+|None|root)', line)
            if node_match and current_node is not None:
                current_node['id'] = int(node_match.group(1))
                parent_str = node_match.group(2)
                if parent_str in ('None', '0', 'root'):
                    current_node['parent'] = None
                else:
                    current_node['parent'] = int(parent_str)
                continue

            # Match Next line
            next_match = re.match(r'Next: parent=(\d+|root)', line)
            if next_match and current_node is not None:
                next_parent_str = next_match.group(1)
                if next_parent_str == 'root':
                    next_parent_map[current_node['iter']] = None
                else:
                    next_parent_map[current_node['iter']] = int(next_parent_str)
                continue

            # Match Score line (0-10 score)
            score_match = re.match(r'Score: (\d+)/10', line)
            if score_match and current_node is not None:
                current_node['score'] = int(score_match.group(1)) / 10.0  # Normalize to 0-1
                continue

            # Match Mutation line
            mutation_match = re.match(r'Mutation: (.+)', line)
            if mutation_match and current_node is not None:
                current_node['mutation'] = mutation_match.group(1).strip()
                continue

        # Save last node
        if current_node is not None and 'id' in current_node and 'score' in current_node:
            nodes[current_node['id']] = current_node

    # Apply next_parent_map
    for node_id, node in nodes.items():
        prev_iter = node_id - 1
        if prev_iter in next_parent_map:
            node['parent'] = next_parent_map[prev_iter]

    # Add current iteration from analysis.log if not yet in markdown
    if current_log_path and current_iteration and os.path.exists(current_log_path):
        with open(current_log_path, 'r') as f:
            log_content = f.read()

        # Parse score from analysis.log
        score_match = re.search(r'score[=:]\s*(\d+)', log_content)
        if score_match:
            score_value = int(score_match.group(1)) / 10.0

            if current_iteration in nodes:
                nodes[current_iteration]['score'] = score_value
            else:
                prev_iter = current_iteration - 1
                parent = next_parent_map.get(prev_iter, prev_iter if prev_iter in nodes else None)
                nodes[current_iteration] = {
                    'iter': current_iteration,
                    'id': current_iteration,
                    'parent': parent,
                    'score': score_value
                }

    if not nodes:
        return False

    # Filter to current block
    if block_size > 0 and current_iteration is not None:
        current_block = (current_iteration - 1) // block_size
        block_start = current_block * block_size + 1
        block_end = (current_block + 1) * block_size

        nodes = {nid: n for nid, n in nodes.items() if block_start <= nid <= block_end}

        for node_id, node in nodes.items():
            if node['parent'] is not None and node['parent'] not in nodes:
                node['parent'] = None

    if not nodes:
        return False

    # Build children map
    children = defaultdict(list)
    for node_id, node in nodes.items():
        if node['parent'] is not None:
            children[node['parent']].append(node_id)

    n_total = len(nodes)

    # Compute visits (PUCT backpropagation)
    visits = {node_id: 1 for node_id in nodes}
    for node_id in sorted(nodes.keys()):
        parent_id = nodes[node_id]['parent']
        while parent_id is not None and parent_id in nodes:
            visits[parent_id] += 1
            parent_id = nodes[parent_id]['parent']

    # Compute UCB scores
    ucb_scores = []
    for node_id, node in nodes.items():
        v = visits[node_id]
        reward = node.get('score', 0.0)
        exploration_term = c * math.sqrt(n_total) / (1 + v)
        ucb = reward + exploration_term

        ucb_scores.append({
            'id': node_id,
            'parent': node['parent'],
            'visits': v,
            'score': reward,
            'ucb': ucb,
            'mutation': node.get('mutation', ''),
            'is_current': node_id == current_iteration
        })

    ucb_scores.sort(key=lambda x: x['ucb'], reverse=True)

    # Write UCB scores
    with open(ucb_path, 'w') as f:
        if block_size > 0 and current_iteration is not None:
            current_block = (current_iteration - 1) // block_size
            block_start = current_block * block_size + 1
            block_end = (current_block + 1) * block_size
            f.write(f"=== UCB Scores (Block {current_block}, iters {block_start}-{block_end}, N={n_total}, c={c}) ===\n\n")
        else:
            f.write(f"=== UCB Scores (N_total={n_total}, c={c}) ===\n\n")

        for score in ucb_scores:
            parent_str = score['parent'] if score['parent'] is not None else 'root'
            mutation_str = score.get('mutation', '')
            line = (f"Node {score['id']}: UCB={score['ucb']:.3f}, "
                    f"parent={parent_str}, visits={score['visits']}, "
                    f"Score={score['score']*10:.0f}/10")
            if mutation_str:
                line += f", Mutation={mutation_str}"
            f.write(line + "\n")

    return True


def run_simulation(config_path: str, root_dir: str, config_name: str) -> tuple:
    """Run the diffusiophoresis simulation using dedicated subprocess script.

    Uses generate_subprocess.py to ensure code modifications are properly reloaded
    for each iteration (similar to NeuralGraph's train_signal_subprocess.py).

    Returns:
        tuple: (success: bool, error_traceback: str or None)
    """
    print("\033[93mRunning simulation...\033[0m")

    # Use the dedicated subprocess script for clean code reloading
    generate_script = os.path.join(root_dir, 'generate_subprocess.py')
    cmd = [
        sys.executable,
        '-u',  # Force unbuffered output for real-time streaming
        generate_script,
        '--config', config_path,
        '--device', 'cuda:1',
        '--erase',
        '--step', '50'
    ]

    print(f"\033[90mcommand: {' '.join(cmd)}\033[0m")
    print(f"\033[90mconfig: {config_path}\033[0m")

    # Stream output while capturing for error analysis
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    output_lines = []
    process = subprocess.Popen(
        cmd, cwd=root_dir, text=True, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='', flush=True)
        output_lines.append(line)

    process.wait()
    full_output = ''.join(output_lines)

    if process.returncode != 0:
        print(f"\033[91mSimulation failed with return code {process.returncode}\033[0m")
        return False, full_output

    print("\033[92mSimulation completed\033[0m")
    return True, None


def generate_mp4_video(fig_dir: str, output_path: str, framerate: int = 30) -> bool:
    """Generate MP4 video from PNG frames."""
    input_pattern = os.path.join(fig_dir, "Fig_0_%06d.png")

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",  # Suppress verbose output
        "-framerate", str(framerate),
        "-i", input_pattern,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"\033[92mGenerated video: {output_path}\033[0m")
            return True
        else:
            print(f"\033[91mVideo generation failed: {result.stderr}\033[0m")
            return False
    except subprocess.TimeoutExpired:
        print("\033[91mVideo generation timeout\033[0m")
        return False
    except FileNotFoundError:
        print("\033[93mffmpeg not found, skipping video generation\033[0m")
        return False


def setup_exploration_dirs(root_dir: str, instruction_name: str) -> dict:
    """Create exploration directory structure."""
    exploration_dir = f"{root_dir}/log/Claude_exploration/{instruction_name}"

    dirs = {
        'base': exploration_dir,
        'activity': f"{exploration_dir}/activity",
        'config': f"{exploration_dir}/config",
        'tree': f"{exploration_dir}/tree",
        'memory': f"{exploration_dir}/memory",
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return dirs


def save_exploration_artifacts(dirs: dict, iteration: int, block_number: int,
                                config_path: str, montage_path: str,
                                video_path: str = None) -> None:
    """Save iteration artifacts to exploration directories."""
    # Save config snapshot
    if os.path.exists(config_path):
        config_dst = f"{dirs['config']}/iter_{iteration:03d}_block_{block_number:02d}.yaml"
        shutil.copy2(config_path, config_dst)

    # Save montage to activity
    if os.path.exists(montage_path):
        montage_dst = f"{dirs['activity']}/montage_iter_{iteration:03d}.png"
        shutil.copy2(montage_path, montage_dst)

    # Save video to activity
    if video_path and os.path.exists(video_path):
        video_dst = f"{dirs['activity']}/video_iter_{iteration:03d}.mp4"
        shutil.copy2(video_path, video_dst)


def parse_ucb_scores_file(filepath: str) -> list:
    """Parse ucb_scores.txt into a list of node dicts."""
    nodes = []
    if not os.path.exists(filepath):
        return nodes

    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern: Node N: UCB=X.XXX, parent=P|root, visits=V, Score=X/10, Mutation=...
    pattern = r'Node (\d+): UCB=([\d.]+), parent=(\d+|root), visits=(\d+), Score=(\d+)/10(?:, Mutation=([^\n]+))?'

    for match in re.finditer(pattern, content):
        node_id = int(match.group(1))
        ucb = float(match.group(2))
        parent_str = match.group(3)
        parent = None if parent_str == 'root' else int(parent_str)
        visits = int(match.group(4))
        score = int(match.group(5)) / 10.0  # Normalize to 0-1
        mutation = match.group(6).strip() if match.group(6) else ""

        nodes.append({
            'id': node_id,
            'ucb': ucb,
            'parent': parent,
            'visits': visits,
            'score': score,
            'mutation': mutation
        })

    return nodes


def plot_ucb_tree(nodes: list, output_path: str, title: str = "UCB Exploration Tree",
                  simulation_info: str = None):
    """
    Plot the UCB exploration tree for pattern exploration.
    Uses 0-10 score instead of R².
    """
    if not nodes:
        print("No nodes to plot")
        return

    # Build tree structure
    children = defaultdict(list)
    node_map = {n['id']: n for n in nodes}

    for node in nodes:
        if node['parent'] is not None:
            children[node['parent']].append(node['id'])

    for parent_id in children:
        children[parent_id].sort()

    roots = [n['id'] for n in nodes if n['parent'] is None or n['parent'] not in node_map]

    # Compute layout
    depth_map = {}
    y_positions = {}

    def compute_depth(node_id, current_depth=0):
        depth_map[node_id] = current_depth
        for child_id in children.get(node_id, []):
            compute_depth(child_id, current_depth + 1)

    for root in roots:
        compute_depth(root, 0)

    leaf_counter = [0]

    def assign_y_dfs(node_id):
        child_list = children.get(node_id, [])
        if not child_list:
            y_positions[node_id] = leaf_counter[0]
            leaf_counter[0] += 1
        else:
            for child_id in child_list:
                assign_y_dfs(child_id)
            y_positions[node_id] = np.mean([y_positions[c] for c in child_list])

    for root in roots:
        assign_y_dfs(root)

    positions = {}
    for node_id in depth_map:
        if node_id in y_positions:
            positions[node_id] = (depth_map[node_id], y_positions[node_id])

    # Color based on score (0-10)
    def get_color(score):
        if score >= 0.7:  # 7-10
            return '#2ecc71'  # green
        elif score >= 0.4:  # 4-6
            return '#f39c12'  # orange
        else:  # 0-3
            return '#e74c3c'  # red

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Draw edges
    for node in nodes:
        if node['parent'] is not None and node['parent'] in positions and node['id'] in positions:
            x1, y1 = positions[node['parent']]
            x2, y2 = positions[node['id']]
            ax.plot([x1, x2], [y1, y2], color='#34495e', linestyle='-',
                   linewidth=1.5, alpha=0.6, zorder=1)

    # UCB range for size scaling
    ucb_values = [n['ucb'] for n in nodes]
    min_ucb = min(ucb_values)
    max_ucb = max(ucb_values)
    ucb_range = max_ucb - min_ucb if max_ucb > min_ucb else 1.0

    # Draw nodes
    for node in nodes:
        if node['id'] not in positions:
            continue

        x, y = positions[node['id']]
        color = get_color(node['score'])
        size = 400 + 200 * (node['ucb'] - min_ucb) / ucb_range

        is_leaf = len(children.get(node['id'], [])) == 0

        if is_leaf:
            ax.scatter(x, y, c=color, s=size, marker='x', linewidths=3, zorder=2)
        else:
            ax.scatter(x, y, c=color, s=size, marker='o',
                      edgecolors='black', linewidths=0.5, zorder=2)

        # Node ID label
        ax.annotate(str(node['id']), (x, y), ha='center', va='center',
                   fontsize=9, color='black', zorder=3)

        # Mutation above node
        if node['id'] > 1 and node['mutation']:
            mutation_text = re.sub(r'\s*\([^)]*\)\s*$', '', node['mutation']).strip()
            ax.annotate(mutation_text, (x, y), ha='left', va='bottom',
                       fontsize=6, xytext=(5, 14), textcoords='offset points',
                       color='#333333', zorder=3, rotation=45)

        # Score and UCB below node
        label_text = f"UCB={node['ucb']:.2f} V={node['visits']}\nScore={node['score']*10:.0f}/10"
        ax.annotate(label_text, (x, y), ha='center', va='top',
                   fontsize=8, xytext=(0, -14), textcoords='offset points',
                   color='#555555', zorder=3)

    # Simulation info
    if simulation_info:
        ax.text(0.02, 0.98, simulation_info, transform=ax.transAxes,
                fontsize=11, ha='left', va='top', color='#333333')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.axis('off')

    # Set axis limits
    if positions:
        x_vals = [p[0] for p in positions.values()]
        y_vals = [p[1] for p in positions.values()]
        ax.set_xlim(min(x_vals) - 0.5, max(x_vals) + 0.5)
        ax.set_ylim(min(y_vals) - 1, max(y_vals) + 1)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Score 7-10'),
        mpatches.Patch(color='#f39c12', label='Score 4-6'),
        mpatches.Patch(color='#e74c3c', label='Score 0-3'),
        plt.Line2D([0], [0], marker='o', color='gray', label='Internal node',
                   markerfacecolor='gray', markersize=8, linestyle='None'),
        plt.Line2D([0], [0], marker='x', color='gray', label='Leaf node',
                   markerfacecolor='gray', markersize=8, linestyle='None', markeredgewidth=2),
    ]
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                       facecolor='white', edgecolor='black', framealpha=1.0)
    for text in legend.get_texts():
        text.set_color('black')

    plt.title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved UCB tree: {output_path}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-guided pattern exploration")
    parser.add_argument("-o", "--option", nargs="+", help="task and config options")
    args = parser.parse_args()

    # Parse arguments
    if args.option:
        task = args.option[0]
        base_config_name = args.option[1] if len(args.option) > 1 else 'diffusiophoresis'
        task_params = {}
        for arg in args.option[2:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                task_params[key] = int(value) if value.isdigit() else value
    else:
        task = 'Claude'
        base_config_name = 'diffusiophoresis'
        task_params = {'iterations': 64}

    n_iterations = 1024
    llm_task_name = f'{base_config_name}_Claude'
    instruction_name = f'instruction_{base_config_name}'

    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_root = f"{root_dir}/config/diffusiophoresis"

    # Paths
    source_config = f"{config_root}/{base_config_name}.yaml"
    target_config = f"{config_root}/{llm_task_name}.yaml"
    instruction_path = f"{root_dir}/{instruction_name}.md"
    analysis_path = f"{root_dir}/{llm_task_name}_analysis.md"
    memory_path = f"{root_dir}/{llm_task_name}_memory.md"
    ucb_path = f"{root_dir}/{llm_task_name}_ucb_scores.txt"
    analysis_log_path = f"{root_dir}/{llm_task_name}_analysis.log"
    reasoning_path = f"{root_dir}/{llm_task_name}_reasoning.log"

    # Check instruction file exists
    if not os.path.exists(instruction_path):
        print(f"\033[91mInstruction file not found: {instruction_path}\033[0m")
        sys.exit(1)

    # Resume support: start_iteration parameter (default 1)
    start_iteration = 155

    if start_iteration > 1:
        print(f"\033[93mResuming from iteration {start_iteration}\033[0m")

    # Copy base config to Claude config (only on fresh start)
    if start_iteration == 1:
        if os.path.exists(source_config):
            shutil.copy2(source_config, target_config)
            print(f"\033[93mCopied {source_config} -> {target_config}\033[0m")

            # Add claude section to config
            with open(target_config, 'r') as f:
                config_data = yaml.safe_load(f)

            config_data['dataset'] = llm_task_name
            config_data['description'] = 'LLM-guided pattern exploration'
            config_data['claude'] = {
                'n_iter_block': 8,
                'ucb_c': 1.414
            }

            with open(target_config, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            print(f"\033[93mAdded claude section to {target_config}\033[0m")
        else:
            print(f"\033[91mBase config not found: {source_config}\033[0m")
            sys.exit(1)

        # Clear UCB scores
        if os.path.exists(ucb_path):
            os.remove(ucb_path)

        # Initialize analysis log
        with open(analysis_path, 'w') as f:
            f.write(f"# Pattern Exploration Log: {base_config_name}\n\n")

        # Initialize reasoning log
        open(reasoning_path, 'w').close()

        # Initialize memory with simulation config table
        # Load config to get model names and simulation params
        init_config = ParticleGraphConfig.from_yaml(target_config)
        mesh_model = getattr(init_config.graph_model, 'mesh_model_name', '') or 'N/A'
        particle_model = getattr(init_config.graph_model, 'particle_model_name', '') or 'N/A'
        n_types = getattr(init_config.simulation, 'n_particle_types', 'N/A')
        n_particles = getattr(init_config.simulation, 'n_particles', 'N/A')

        with open(memory_path, 'w') as f:
            f.write(f"# Working Memory: {base_config_name}\n\n")
            f.write("## Simulation Config\n\n")
            f.write("| Parameter        | Value                    |\n")
            f.write("| ---------------- | ------------------------ |\n")
            f.write(f"| mesh_model_name  | {mesh_model} |\n")
            f.write(f"| particle_model   | {particle_model} |\n")
            f.write(f"| n_particle_types | {n_types} |\n")
            f.write(f"| n_particles      | {n_particles} |\n\n")
            f.write("## Insights\n\n")
            f.write("| Category    | Finding                                              |\n")
            f.write("| ----------- | ---------------------------------------------------- |\n")
            f.write("| Patterns    | [key pattern observations]                           |\n")
            f.write("| Performance | [what configs work well]                             |\n")
            f.write("| Failures    | [what to avoid]                                      |\n\n")
            f.write("---\n\n")
            f.write("## Knowledge Base\n\n")
            f.write("### Pattern Principles\n\n")
            f.write("### Failed Configurations\n\n")
            f.write("### Code Insights\n\n")
            f.write("---\n\n")
            f.write("## Previous Block Summary\n\n")
            f.write("---\n\n")
            f.write("## Current Block (Block 1)\n\n")
            f.write("### Block Info\n\n")
            f.write("### Hypothesis\n\n")
            f.write("### Iterations This Block\n\n")
            f.write("### Emerging Observations\n\n")
    else:
        # Resuming - preserve existing files
        print(f"\033[93mPreserving {target_config} (resuming)\033[0m")
        print(f"\033[93mPreserving {analysis_path} (resuming)\033[0m")
        print(f"\033[93mPreserving {memory_path} (resuming)\033[0m")
        print(f"\033[93mPreserving {ucb_path} (resuming)\033[0m")

    # Load config to get n_iter_block
    config = ParticleGraphConfig.from_yaml(target_config)
    n_iter_block = config.claude.n_iter_block if config.claude else 8
    ucb_c = config.claude.ucb_c if config.claude else 1.414

    # Track code modifications
    code_changes_enabled = 'code' in task
    code_modified = False

    if code_changes_enabled:
        print("\033[93mCode modifications ENABLED\033[0m")

    # Setup exploration directories
    exploration_dirs = setup_exploration_dirs(root_dir, instruction_name)
    print(f"\033[93mExploration directory: {exploration_dirs['base']}\033[0m")

    # Main loop
    for iteration in range(start_iteration, n_iterations + 1):
        print(f"\n\n\033[94m=== Iteration {iteration}/{n_iterations} ===\033[0m")

        block_number = (iteration - 1) // n_iter_block + 1
        iter_in_block = (iteration - 1) % n_iter_block + 1
        is_block_end = iter_in_block == n_iter_block

        # Block boundary: clear UCB scores
        if iteration > 1 and (iteration - 1) % n_iter_block == 0:
            if os.path.exists(ucb_path):
                os.remove(ucb_path)
                print(f"\033[93mBlock boundary: cleared UCB scores\033[0m")

        # Reload config
        config = ParticleGraphConfig.from_yaml(target_config)
        # GNN_Main.py adds prefix folder to dataset path (e.g., "diffusiophoresis/diffusiophoresis_Claude")
        dataset_name = f"{base_config_name}/{config.dataset}"

        # 1. Run simulation with error recovery
        max_repair_attempts = 10
        success = False
        error_traceback = None

        for attempt in range(max_repair_attempts + 1):
            success, error_traceback = run_simulation(target_config, root_dir, llm_task_name)

            if success:
                break

            if attempt == 0:
                print(f"\033[91mSimulation failed at iteration {iteration}\033[0m")
                if error_traceback:
                    print(f"\033[91mError:\n{error_traceback[-2000:]}\033[0m")  # Last 2000 chars

            # Check if code was modified (only attempt repair for code errors)
            code_files = [
                'src/ParticleGraph/generators/PDE_Diffusiophoresis.py',
                'src/ParticleGraph/generators/PDE_D.py',
                'src/ParticleGraph/generators/graph_data_generator.py',
            ]
            modified_code = get_modified_code_files(root_dir, code_files) if is_git_repo(root_dir) else []

            if not modified_code and attempt == 0:
                print(f"\033[93mNo code modifications detected - skipping repair attempts\033[0m")
                break

            # Attempt repair only if code was modified
            if attempt < max_repair_attempts and modified_code:
                print(f"\033[93mAttempt {attempt + 1}/{max_repair_attempts}: Asking Claude to fix the code error...\033[0m")

                repair_prompt = f"""SIMULATION CRASHED - Please fix the code error.

Attempt {attempt + 1}/{max_repair_attempts}

Error traceback:
```
{error_traceback[-3000:] if error_traceback else 'No traceback available'}
```

Modified code files that may contain the bug:
{chr(10).join(f'- {root_dir}/{f}' for f in modified_code)}

Instructions:
1. Read the error traceback carefully
2. Identify the bug in the modified code
3. Fix the bug using the Edit tool
4. Do NOT make other changes, only fix the crash

If you cannot fix it, say "CANNOT_FIX" and explain why."""

                repair_cmd = [
                    'claude',
                    '-p', repair_prompt,
                    '--output-format', 'text',
                    '--max-turns', '10',
                    '--allowedTools', 'Read', 'Edit', 'Write'
                ]

                repair_result = subprocess.run(repair_cmd, cwd=root_dir, capture_output=True, text=True)
                repair_output = repair_result.stdout

                if 'CANNOT_FIX' in repair_output:
                    print(f"\033[91mClaude cannot fix the error\033[0m")
                    break

                print(f"\033[92mRepair attempt {attempt + 1} complete, retrying simulation...\033[0m")

        # If still failing after all attempts, rollback and log
        if not success:
            print(f"\033[91mAll repair attempts failed - rolling back code changes\033[0m")

            # Rollback modified files using git
            if is_git_repo(root_dir):
                for file_path in ['src/ParticleGraph/generators/PDE_Diffusiophoresis.py',
                                  'src/ParticleGraph/generators/PDE_D.py',
                                  'src/ParticleGraph/generators/graph_data_generator.py']:
                    try:
                        subprocess.run(['git', 'checkout', 'HEAD', '--', file_path],
                                      cwd=root_dir, capture_output=True, timeout=10)
                    except:
                        pass
                print(f"\033[93mRolled back code to last working state\033[0m")

            # Log failed modification to memory
            if os.path.exists(memory_path):
                with open(memory_path, 'a') as f:
                    f.write(f"\n### Failed Code Modification (Iter {iteration})\n")
                    f.write(f"Error: {error_traceback[-500:] if error_traceback else 'Unknown'}\n")
                    f.write(f"**DO NOT retry this modification**\n\n")

            continue  # Skip to next iteration

        # 2. Create frame montage
        fig_dir = f"{root_dir}/graphs_data/{dataset_name}/Fig"
        montage_path = f"{root_dir}/graphs_data/{dataset_name}/montage_iter_{iteration:03d}.png"

        if not create_frame_montage(fig_dir, montage_path, n_frames=10):
            print(f"\033[91mFailed to create montage\033[0m")
            continue

        # 3. Write analysis.log (basic metrics + simulation metrics if available)
        sim_log_path = f"{root_dir}/graphs_data/{dataset_name}/analysis.log"
        with open(analysis_log_path, 'w') as f:
            f.write(f"iteration: {iteration}\n")
            f.write(f"block: {block_number}\n")
            f.write(f"iter_in_block: {iter_in_block}\n")
            f.write(f"n_frames: {config.simulation.n_frames}\n")
            f.write(f"n_particles: {config.simulation.n_particles}\n")
            f.write(f"delta_t: {config.simulation.delta_t}\n")
            # Append simulation metrics if available
            if os.path.exists(sim_log_path):
                with open(sim_log_path, 'r') as sim_f:
                    f.write("\n# Simulation metrics:\n")
                    f.write(sim_f.read())

        # 4. Compute UCB scores
        compute_ucb_scores(analysis_path, ucb_path, c=ucb_c,
                          current_log_path=analysis_log_path,
                          current_iteration=iteration,
                          block_size=n_iter_block)

        # 5. Call Claude CLI
        print("\033[93mClaude analysis...\033[0m")

        claude_prompt = f"""Iteration {iteration}/{n_iterations}
Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block
{">>> BLOCK END <<<" if is_block_end else ""}

Instructions (follow all instructions): {instruction_path}
Working memory: {memory_path}
Full log (append only): {analysis_path}
Montage image (10 frames from simulation): {montage_path}
Metrics log: {analysis_log_path}
UCB scores: {ucb_path}
Current config: {target_config}"""

        # Add code file paths ONLY at block end (code changes between blocks)
        if code_changes_enabled and is_block_end:
            claude_prompt += f"""

Code files you can modify (BLOCK END only - for next block):
- {root_dir}/src/ParticleGraph/generators/PDE_Diffusiophoresis.py
- {root_dir}/src/ParticleGraph/generators/PDE_D.py
- {root_dir}/src/ParticleGraph/generators/graph_data_generator.py"""

        claude_cmd = [
            'claude',
            '-p', claude_prompt,
            '--output-format', 'text',
            '--max-turns', '100',
            '--allowedTools', 'Read', 'Edit', 'Write'
        ]

        # Run Claude
        output_lines = []
        process = subprocess.Popen(
            claude_cmd,
            cwd=root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end='', flush=True)
            output_lines.append(line)

        process.wait()

        # Save reasoning output
        output_text = ''.join(output_lines)
        if output_text.strip():
            with open(reasoning_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"=== Iteration {iteration} ===\n")
                f.write(f"{'='*60}\n")
                f.write(output_text.strip())
                f.write("\n\n")

        # 6. Track code modifications
        if code_changes_enabled:
            if is_git_repo(root_dir):
                results = track_code_modifications(root_dir, iteration)
                for file_path, success, message in results:
                    if success:
                        print(f"\033[92mGit: {message}\033[0m")
                        code_modified = True
                    else:
                        print(f"\033[93mGit: {message}\033[0m")

        # 7. Rename input video to iteration-specific name
        # The simulation already generates input_{dataset}.mp4 at framerate=8
        input_video = f"{root_dir}/graphs_data/{dataset_name}/input_{config.dataset}.mp4"
        video_path = f"{root_dir}/graphs_data/{dataset_name}/video_iter_{iteration:03d}.mp4"
        if os.path.exists(input_video):
            shutil.move(input_video, video_path)

        # 8. Save exploration artifacts
        save_exploration_artifacts(
            dirs=exploration_dirs,
            iteration=iteration,
            block_number=block_number,
            config_path=target_config,
            montage_path=montage_path,
            video_path=video_path
        )

        # 9. Recompute UCB and generate tree visualization
        compute_ucb_scores(analysis_path, ucb_path, c=ucb_c,
                          current_log_path=analysis_log_path,
                          current_iteration=iteration,
                          block_size=n_iter_block)

        # Generate UCB tree (for block 1: every iteration; block 2+: only at block end)
        should_save_tree = (block_number == 1) or is_block_end
        if should_save_tree and os.path.exists(ucb_path):
            nodes = parse_ucb_scores_file(ucb_path)
            if nodes:
                tree_path = f"{exploration_dirs['tree']}/ucb_tree_iter_{iteration:03d}.png"
                sim_info = f"n_particles={config.simulation.n_particles}, n_frames={config.simulation.n_frames}"
                sim_info += f", delta_t={config.simulation.delta_t}"
                plot_ucb_tree(nodes, tree_path,
                             title=f"UCB Tree - Iter {iteration}",
                             simulation_info=sim_info)

        # 10. Save memory at block end
        if is_block_end:
            memory_dst = f"{exploration_dirs['memory']}/block_{block_number:03d}_memory.md"
            if os.path.exists(memory_path):
                shutil.copy2(memory_path, memory_dst)
                print(f"\033[92mSaved memory snapshot: {memory_dst}\033[0m")

        print(f"\033[92mIteration {iteration} complete\033[0m")

    print(f"\n\033[92m=== Exploration complete ({n_iterations} iterations) ===\033[0m")

    # 
