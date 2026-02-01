"""
LLM-Guided Diffusiophoresis Pattern Exploration — Parallel Mode

Runs 4 simulations per batch, calls Claude once per batch to analyze
all 4 results and propose 4 new mutations.

Usage:
    python GNN_LLM_parallel.py                              # fresh start, local
    python GNN_LLM_parallel.py --resume                     # auto-resume
    python GNN_LLM_parallel.py -o Claude_cluster diffusiophoresis iterations=1024  # cluster mode
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


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def detect_last_iteration(analysis_path, config_save_dir, n_parallel):
    """Detect the last fully completed batch from saved artifacts.

    Scans two sources:
      1. analysis.md for ``## Iter N:`` entries
      2. config save dir for ``iter_NNN_slot_SS.yaml`` files

    Returns the start_iteration for the next batch (1-indexed), or 1 if nothing found.
    """
    found_iters = set()

    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            for line in f:
                match = re.match(r'^##+ Iter (\d+):', line)
                if match:
                    found_iters.add(int(match.group(1)))

    if os.path.isdir(config_save_dir):
        for fname in os.listdir(config_save_dir):
            match = re.match(r'iter_(\d+)_slot_\d+\.yaml', fname)
            if match:
                found_iters.add(int(match.group(1)))

    if not found_iters:
        return 1

    last_iter = max(found_iters)
    batch_start = ((last_iter - 1) // n_parallel) * n_parallel + 1
    batch_iters = set(range(batch_start, batch_start + n_parallel))

    if batch_iters.issubset(found_iters):
        return batch_start + n_parallel
    else:
        return batch_start


# ---------------------------------------------------------------------------
# Cluster helpers
# ---------------------------------------------------------------------------

CLUSTER_HOME = "/groups/saalfeld/home/allierc"
CLUSTER_ROOT_DIR = f"{CLUSTER_HOME}/Graph/ParticleGraph"


def submit_cluster_job(slot, config_path, log_dir, root_dir):
    """Submit a simulation job to the cluster WITHOUT -K (non-blocking).

    Returns the LSF job ID string, or None if submission failed.
    """
    cluster_script_path = f"{log_dir}/cluster_sim_{slot:02d}.sh"

    # Build cluster-side paths
    cluster_config_path = config_path.replace(root_dir, CLUSTER_ROOT_DIR)

    cluster_cmd = f"python generate_subprocess.py --config '{cluster_config_path}' --device cuda --erase --step 50"

    # Cluster-side log paths for capturing stdout/stderr
    cluster_log_dir = log_dir.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_stdout = f"{cluster_log_dir}/cluster_sim_{slot:02d}.out"
    cluster_stderr = f"{cluster_log_dir}/cluster_sim_{slot:02d}.err"

    with open(cluster_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {CLUSTER_ROOT_DIR}\n")
        f.write(f"conda run -n particle-graph {cluster_cmd}\n")
    os.chmod(cluster_script_path, 0o755)

    cluster_script = cluster_script_path.replace(root_dir, CLUSTER_ROOT_DIR)

    # Submit WITHOUT -K so it returns immediately; capture stdout/stderr to files
    ssh_cmd = (
        f"ssh allierc@login1 \"cd {CLUSTER_ROOT_DIR} && "
        f"bsub -n 8 -gpu 'num=1' -q gpu_a100 -W 6000 "
        f"-o '{cluster_stdout}' -e '{cluster_stderr}' "
        f"'bash {cluster_script}'\""
    )
    print(f"\033[96m  slot {slot}: submitting via SSH\033[0m")
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

    match = re.search(r'Job <(\d+)>', result.stdout)
    if match:
        job_id = match.group(1)
        print(f"\033[92m  slot {slot}: job {job_id} submitted\033[0m")
        return job_id
    else:
        print(f"\033[91m  slot {slot}: submission FAILED\033[0m")
        print(f"    stdout: {result.stdout.strip()}")
        print(f"    stderr: {result.stderr.strip()}")
        return None


def wait_for_cluster_jobs(job_ids, log_dir=None, poll_interval=60):
    """Poll bjobs via SSH until all jobs finish.

    Returns: dict {slot: bool} — True if DONE, False if EXIT/failed
    """
    pending = dict(job_ids)
    results = {}

    while pending:
        ids_str = ' '.join(pending.values())
        ssh_cmd = f'ssh allierc@login1 "bjobs {ids_str} 2>/dev/null"'
        out = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

        for slot, jid in list(pending.items()):
            for line in out.stdout.splitlines():
                if jid in line:
                    if 'DONE' in line:
                        results[slot] = True
                        del pending[slot]
                        print(f"\033[92m  slot {slot} (job {jid}): DONE\033[0m")
                    elif 'EXIT' in line:
                        results[slot] = False
                        del pending[slot]
                        print(f"\033[91m  slot {slot} (job {jid}): FAILED (EXIT)\033[0m")
                        # Try to read error log for diagnosis
                        if log_dir:
                            err_file = f"{log_dir}/cluster_sim_{slot:02d}.err"
                            if os.path.exists(err_file):
                                try:
                                    with open(err_file, 'r') as ef:
                                        err_content = ef.read().strip()
                                    if err_content:
                                        print(f"\033[91m  --- slot {slot} error log ---\033[0m")
                                        for eline in err_content.splitlines()[-30:]:
                                            print(f"\033[91m    {eline}\033[0m")
                                        print(f"\033[91m  --- end error log ---\033[0m")
                                except Exception:
                                    pass

            if slot in pending and jid not in out.stdout:
                results[slot] = True
                del pending[slot]
                print(f"\033[93m  slot {slot} (job {jid}): no longer in queue (assuming DONE)\033[0m")

        if pending:
            statuses = [f"slot {s}" for s in pending]
            print(f"\033[90m  ... waiting for {', '.join(statuses)} ({poll_interval}s)\033[0m")
            time.sleep(poll_interval)

    return results


# ---------------------------------------------------------------------------
# Helpers (reused from GNN_LLM.py)
# ---------------------------------------------------------------------------

def is_git_repo(root_dir: str) -> bool:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            cwd=root_dir, capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_modified_code_files(root_dir: str, tracked_files: list) -> list:
    modified = []
    for file_path in tracked_files:
        try:
            result = subprocess.run(
                ['git', 'diff', '--quiet', file_path],
                cwd=root_dir, capture_output=True, timeout=5
            )
            if result.returncode != 0:
                modified.append(file_path)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return modified


def commit_code_modification(root_dir: str, file_path: str, iteration: int,
                              description: str = None) -> tuple:
    if not is_git_repo(root_dir):
        return False, "Not a git repository"
    if description is None:
        description = f"Code modification in {os.path.basename(file_path)}"
    try:
        subprocess.run(['git', 'add', file_path], cwd=root_dir, timeout=10)
        commit_msg = f"[Iter {iteration}] {description}\n\n[Automated commit by Claude]"
        result = subprocess.run(
            ['git', 'commit', '-m', commit_msg],
            cwd=root_dir, capture_output=True, text=True, timeout=10
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
    if not is_git_repo(root_dir):
        return []
    code_files = [
        'src/ParticleGraph/generators/PDE_D.py',
        'src/ParticleGraph/generators/graph_data_generator.py',
        'src/ParticleGraph/generators/utils.py',
    ]
    generators_dir = Path(root_dir) / 'src/ParticleGraph/generators'
    for pde_file in generators_dir.glob('PDE_Diffusiophoresis*.py'):
        rel_path = str(pde_file.relative_to(root_dir))
        if rel_path not in code_files:
            code_files.append(rel_path)
    for pde_d_file in generators_dir.glob('PDE_D_*.py'):
        rel_path = str(pde_d_file.relative_to(root_dir))
        if rel_path not in code_files:
            code_files.append(rel_path)
    modified = get_modified_code_files(root_dir, code_files)
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
    png_files = sorted(Path(fig_dir).glob("Fig_0_*.png"))
    if not png_files:
        print(f"No PNG files found in {fig_dir}")
        return False
    total_frames = len(png_files)
    if total_frames < n_frames:
        indices = list(range(total_frames))
    else:
        indices = [int(i * total_frames / n_frames) for i in range(n_frames)]
    selected_files = [png_files[i] for i in indices]
    images = [Image.open(f) for f in selected_files]
    cols, rows = 5, 2
    img_width, img_height = images[0].size
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


def run_simulation(config_path: str, root_dir: str, config_name: str) -> tuple:
    """Run the diffusiophoresis simulation using dedicated subprocess script."""
    generate_script = os.path.join(root_dir, 'generate_subprocess.py')
    cmd = [
        sys.executable,
        '-u',
        generate_script,
        '--config', config_path,
        '--device', 'cuda:1',
        '--erase',
        '--step', '50'
    ]
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
        return False, full_output
    return True, None


def generate_mp4_video(fig_dir: str, output_path: str, framerate: int = 30) -> bool:
    input_pattern = os.path.join(fig_dir, "Fig_0_%06d.png")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(framerate),
        "-i", input_pattern,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
        output_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"\033[92mGenerated video: {output_path}\033[0m")
            return True
        else:
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_claude_cli(prompt, root_dir, max_turns=500):
    """Run Claude CLI with real-time output streaming. Returns output text."""
    claude_cmd = [
        'claude',
        '-p', prompt,
        '--output-format', 'text',
        '--max-turns', str(max_turns),
        '--allowedTools',
        'Read', 'Edit', 'Write'
    ]
    output_lines = []
    process = subprocess.Popen(
        claude_cmd, cwd=root_dir,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    for line in process.stdout:
        print(line, end='', flush=True)
        output_lines.append(line)
    process.wait()
    return ''.join(output_lines)


# ---------------------------------------------------------------------------
# UCB (reused from GNN_LLM.py — score-based 0-10)
# ---------------------------------------------------------------------------

def compute_ucb_scores(analysis_path, ucb_path, c=1.414,
                       current_log_path=None, current_iteration=None,
                       block_size=8):
    nodes = {}
    next_parent_map = {}

    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            content = f.read()

        current_node = None
        lines = content.split('\n')

        for line in lines:
            iter_match = re.match(r'##+ Iter (\d+):', line)
            if iter_match:
                if current_node is not None and 'id' in current_node and 'score' in current_node:
                    nodes[current_node['id']] = current_node
                current_iter = int(iter_match.group(1))
                current_node = {'iter': current_iter}
                continue

            node_match = re.match(r'Node: id=(\d+), parent=(\d+|None|root)', line)
            if node_match and current_node is not None:
                current_node['id'] = int(node_match.group(1))
                parent_str = node_match.group(2)
                if parent_str in ('None', '0', 'root'):
                    current_node['parent'] = None
                else:
                    current_node['parent'] = int(parent_str)
                continue

            next_match = re.match(r'Next: parent=(\d+|root)', line)
            if next_match and current_node is not None:
                next_parent_str = next_match.group(1)
                if next_parent_str == 'root':
                    next_parent_map[current_node['iter']] = None
                else:
                    next_parent_map[current_node['iter']] = int(next_parent_str)
                continue

            score_match = re.match(r'Score: (\d+)/10', line)
            if score_match and current_node is not None:
                current_node['score'] = int(score_match.group(1)) / 10.0
                continue

            mutation_match = re.match(r'Mutation: (.+)', line)
            if mutation_match and current_node is not None:
                current_node['mutation'] = mutation_match.group(1).strip()
                continue

        if current_node is not None and 'id' in current_node and 'score' in current_node:
            nodes[current_node['id']] = current_node

    for node_id, node in nodes.items():
        prev_iter = node_id - 1
        if prev_iter in next_parent_map:
            new_parent = next_parent_map[prev_iter]
            if new_parent == node_id:
                continue
            node['parent'] = new_parent

    if current_log_path and current_iteration and os.path.exists(current_log_path):
        with open(current_log_path, 'r') as f:
            log_content = f.read()
        score_match = re.search(r'score[=:]\s*(\d+)', log_content)
        if score_match:
            score_value = int(score_match.group(1)) / 10.0
            if current_iteration in nodes:
                nodes[current_iteration]['score'] = score_value
            else:
                prev_iter = current_iteration - 1
                parent = next_parent_map.get(prev_iter, prev_iter if prev_iter in nodes else None)
                nodes[current_iteration] = {
                    'iter': current_iteration, 'id': current_iteration,
                    'parent': parent, 'score': score_value
                }

    if not nodes:
        return False

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

    children = defaultdict(list)
    for node_id, node in nodes.items():
        if node['parent'] is not None:
            children[node['parent']].append(node_id)

    n_total = len(nodes)

    visits = {node_id: 1 for node_id in nodes}
    for node_id in sorted(nodes.keys()):
        parent_id = nodes[node_id]['parent']
        while parent_id is not None and parent_id in nodes:
            visits[parent_id] += 1
            parent_id = nodes[parent_id]['parent']

    ucb_scores = []
    for node_id, node in nodes.items():
        v = visits[node_id]
        reward = node.get('score', 0.0)
        exploration_term = c * math.sqrt(n_total) / (1 + v)
        ucb = reward + exploration_term
        ucb_scores.append({
            'id': node_id, 'parent': node['parent'], 'visits': v,
            'score': reward, 'ucb': ucb,
            'mutation': node.get('mutation', ''),
            'is_current': node_id == current_iteration
        })

    ucb_scores.sort(key=lambda x: x['ucb'], reverse=True)

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


def parse_ucb_scores_file(filepath):
    nodes = []
    if not os.path.exists(filepath):
        return nodes
    with open(filepath, 'r') as f:
        content = f.read()
    pattern = r'Node (\d+): UCB=([\d.]+), parent=(\d+|root), visits=(\d+), Score=(\d+)/10(?:, Mutation=([^\n]+))?'
    for match in re.finditer(pattern, content):
        nodes.append({
            'id': int(match.group(1)),
            'ucb': float(match.group(2)),
            'parent': None if match.group(3) == 'root' else int(match.group(3)),
            'visits': int(match.group(4)),
            'score': int(match.group(5)) / 10.0,
            'mutation': match.group(6).strip() if match.group(6) else ""
        })
    return nodes


def plot_ucb_tree(nodes, output_path, title="UCB Exploration Tree", simulation_info=None):
    if not nodes:
        return
    children = defaultdict(list)
    node_map = {n['id']: n for n in nodes}
    for node in nodes:
        if node['parent'] is not None:
            children[node['parent']].append(node['id'])
    for parent_id in children:
        children[parent_id].sort()
    roots = [n['id'] for n in nodes if n['parent'] is None or n['parent'] not in node_map]
    depth_map = {}
    y_positions = {}

    def compute_depth(node_id, d=0):
        depth_map[node_id] = d
        for child_id in children.get(node_id, []):
            compute_depth(child_id, d + 1)
    for root in roots:
        compute_depth(root, 0)

    leaf_counter = [0]
    def assign_y(node_id):
        child_list = children.get(node_id, [])
        if not child_list:
            y_positions[node_id] = leaf_counter[0]
            leaf_counter[0] += 1
        else:
            for c in child_list:
                assign_y(c)
            y_positions[node_id] = np.mean([y_positions[c] for c in child_list])
    for root in roots:
        assign_y(root)

    positions = {nid: (depth_map[nid], y_positions[nid])
                 for nid in depth_map if nid in y_positions}

    def get_color(score):
        if score >= 0.7: return '#2ecc71'
        elif score >= 0.4: return '#f39c12'
        else: return '#e74c3c'

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    for node in nodes:
        if node['parent'] is not None and node['parent'] in positions and node['id'] in positions:
            x1, y1 = positions[node['parent']]
            x2, y2 = positions[node['id']]
            ax.plot([x1, x2], [y1, y2], color='#34495e', linewidth=1.5, alpha=0.6, zorder=1)
    ucb_values = [n['ucb'] for n in nodes]
    ucb_range = max(ucb_values) - min(ucb_values) if max(ucb_values) > min(ucb_values) else 1.0
    min_ucb = min(ucb_values)
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
            ax.scatter(x, y, c=color, s=size, marker='o', edgecolors='black', linewidths=0.5, zorder=2)
        ax.annotate(str(node['id']), (x, y), ha='center', va='center', fontsize=9, color='black', zorder=3)
        if node['id'] > 1 and node['mutation']:
            mutation_text = re.sub(r'\s*\([^)]*\)\s*$', '', node['mutation']).strip()
            ax.annotate(mutation_text, (x, y), ha='left', va='bottom', fontsize=6,
                       xytext=(5, 14), textcoords='offset points', color='#333333', zorder=3, rotation=45)
        label_text = f"UCB={node['ucb']:.2f} V={node['visits']}\nScore={node['score']*10:.0f}/10"
        ax.annotate(label_text, (x, y), ha='center', va='top', fontsize=8,
                   xytext=(0, -14), textcoords='offset points', color='#555555', zorder=3)
    if simulation_info:
        ax.text(0.02, 0.98, simulation_info, transform=ax.transAxes, fontsize=11, ha='left', va='top', color='#333333')
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.axis('off')
    if positions:
        x_vals = [p[0] for p in positions.values()]
        y_vals = [p[1] for p in positions.values()]
        ax.set_xlim(min(x_vals) - 0.5, max(x_vals) + 0.5)
        ax.set_ylim(min(y_vals) - 1, max(y_vals) + 1)
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Score 7-10'),
        mpatches.Patch(color='#f39c12', label='Score 4-6'),
        mpatches.Patch(color='#e74c3c', label='Score 0-3'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, facecolor='white', edgecolor='black')
    plt.title(title)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved UCB tree: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-guided pattern exploration — Parallel Mode")
    parser.add_argument("-o", "--option", nargs="+", help="task and config options")
    parser.add_argument("--fresh", action="store_true", default=True, help="start from iteration 1 (default)")
    parser.add_argument("--resume", action="store_true", help="auto-resume from last completed batch")
    args = parser.parse_args()

    N_PARALLEL = 4

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
        task = 'Claude_code'
        base_config_name = 'diffusiophoresis'
        task_params = {'iterations': 1024}

    n_iterations = task_params.get('iterations', 1024)
    llm_task_name = f'{base_config_name}_Claude'
    instruction_name = f'instruction_{base_config_name}'

    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_root = f"{root_dir}/config/diffusiophoresis"

    # Shared file paths
    source_config = f"{config_root}/{base_config_name}.yaml"
    analysis_path = f"{root_dir}/{llm_task_name}_analysis.md"
    memory_path = f"{root_dir}/{llm_task_name}_memory.md"
    ucb_path = f"{root_dir}/{llm_task_name}_ucb_scores.txt"
    instruction_path = f"{root_dir}/{instruction_name}.md"
    parallel_instruction_path = f"{root_dir}/instruction_{base_config_name}_parallel.md"
    reasoning_path = f"{root_dir}/{llm_task_name}_reasoning.log"

    exploration_dir = f"{root_dir}/log/Claude_exploration/{instruction_name}_parallel"
    os.makedirs(exploration_dir, exist_ok=True)

    # Create exploration subdirectories
    for subdir in ['config', 'montage', 'video', 'figure', 'tree', 'memory', 'activity']:
        os.makedirs(f"{exploration_dir}/{subdir}", exist_ok=True)

    # Check instruction files
    if not os.path.exists(instruction_path):
        print(f"\033[91minstruction file not found: {instruction_path}\033[0m")
        sys.exit(1)
    if not os.path.exists(parallel_instruction_path):
        print(f"\033[93mwarning: parallel instruction file not found: {parallel_instruction_path}\033[0m")
        parallel_instruction_path = None

    # -----------------------------------------------------------------------
    # Fresh start (default) or auto-resume (--resume)
    # -----------------------------------------------------------------------
    if args.resume:
        config_save_dir = f"{exploration_dir}/config"
        start_iteration = detect_last_iteration(analysis_path, config_save_dir, N_PARALLEL)
        if start_iteration > 1:
            print(f"\033[93mAuto-resume: resuming from batch starting at {start_iteration}\033[0m")
        else:
            print(f"\033[93mNo previous iterations found, starting fresh\033[0m")
    else:
        start_iteration = 1
        if os.path.exists(analysis_path):
            print(f"\033[91mWARNING: Fresh start will erase existing results in:\033[0m")
            print(f"\033[91m  {analysis_path}\033[0m")
            print(f"\033[91m  {memory_path}\033[0m")
            answer = input("\033[91mContinue? (y/n): \033[0m").strip().lower()
            if answer != 'y':
                print("Aborted.")
                sys.exit(0)
        print(f"\033[93mFresh start\033[0m")

    # -----------------------------------------------------------------------
    # Initialize 4 slot configs
    # -----------------------------------------------------------------------
    config_paths = {}
    analysis_log_paths = {}
    slot_names = {}

    # Read source config for claude params
    with open(source_config, 'r') as f:
        source_data = yaml.safe_load(f)
    claude_cfg = source_data.get('claude', {})
    n_iter_block = claude_cfg.get('n_iter_block', 8)
    ucb_c = claude_cfg.get('ucb_c', 1.414)

    for slot in range(N_PARALLEL):
        slot_name = f"{llm_task_name}_{slot:02d}"
        slot_names[slot] = slot_name
        target = f"{config_root}/{slot_name}.yaml"
        config_paths[slot] = target
        analysis_log_paths[slot] = f"{root_dir}/{slot_name}_analysis.log"

        if start_iteration == 1 and not args.resume:
            shutil.copy2(source_config, target)
            with open(target, 'r') as f:
                config_data = yaml.safe_load(f)
            config_data['dataset'] = slot_name
            config_data['description'] = 'LLM-guided pattern exploration (parallel)'
            with open(target, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            print(f"\033[93m  slot {slot}: created {target} (dataset='{slot_name}')\033[0m")
        else:
            print(f"\033[93m  slot {slot}: preserving {target} (resuming)\033[0m")

    # -----------------------------------------------------------------------
    # Initialize shared files on fresh start
    # -----------------------------------------------------------------------
    if start_iteration == 1 and not args.resume:
        with open(analysis_path, 'w') as f:
            f.write(f"# Pattern Exploration Log: {base_config_name} (parallel)\n\n")
        open(reasoning_path, 'w').close()

        init_config = ParticleGraphConfig.from_yaml(config_paths[0])
        mesh_model = getattr(init_config.graph_model, 'mesh_model_name', '') or 'N/A'
        particle_model = getattr(init_config.graph_model, 'particle_model_name', '') or 'N/A'
        n_types = getattr(init_config.simulation, 'n_particle_types', 'N/A')
        n_particles = getattr(init_config.simulation, 'n_particles', 'N/A')

        with open(memory_path, 'w') as f:
            f.write(f"# Working Memory: {base_config_name} (parallel)\n\n")
            f.write("## Regime Comparison\n\n")
            f.write("| Regime | mesh_model | particle_model | n_types | n_particles | Best Score | Key Insight |\n")
            f.write("| ------ | ---------- | -------------- | ------- | ----------- | ---------- | ----------- |\n")
            f.write(f"| Base   | {mesh_model} | {particle_model} | {n_types} | {n_particles} | - | baseline |\n\n")
            f.write("## Knowledge Base\n\n")
            f.write("### Established Principles\n\n")
            f.write("### Open Questions\n\n")
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

        if os.path.exists(ucb_path):
            os.remove(ucb_path)

        print(f"\033[93mcleared shared files\033[0m")
    else:
        print(f"\033[93mpreserving shared files (resuming from iter {start_iteration})\033[0m")

    # Cluster mode enabled by default (use 'local' in task to disable)
    cluster_enabled = 'local' not in task

    # Code tracking
    code_changes_enabled = True

    print(f"\033[93m{instruction_name} PARALLEL (N={N_PARALLEL}, {n_iterations} iterations, starting at {start_iteration})\033[0m")

    # -----------------------------------------------------------------------
    # BATCH 0: Claude "start" call — initialize 4 config variations
    # -----------------------------------------------------------------------
    if start_iteration == 1 and not args.resume:
        print(f"\n\033[94m{'='*60}\033[0m")
        print(f"\033[94mBATCH 0: Claude initializing {N_PARALLEL} config variations\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        slot_list = "\n".join(
            f"  Slot {s}: {config_paths[s]}"
            for s in range(N_PARALLEL)
        )

        parallel_ref = f"\nParallel instructions: {parallel_instruction_path}" if parallel_instruction_path else ""

        start_prompt = f"""PARALLEL START: Initialize {N_PARALLEL} config variations for the first batch.

Instructions (follow all instructions): {instruction_path}{parallel_ref}
Working memory: {memory_path}
Full log (append only): {analysis_path}

Config files to edit (all {N_PARALLEL}):
{slot_list}

Read the instructions and the base config, then create {N_PARALLEL} diverse initial simulation
parameter variations. Each config already has a unique dataset name — do NOT change the
dataset field. Vary key PDE parameters (e.g. D_C1, D_C2, Da_c, chi, consumption/production,
n_particle_types, n_particles) across the {N_PARALLEL} slots to explore different starting points.

Write the planned mutations to the working memory file."""

        print("\033[93mClaude start call...\033[0m")
        output_text = run_claude_cli(start_prompt, root_dir, max_turns=100)

        if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
            print(f"\n\033[91mOAuth token expired during start call\033[0m")
            print("\033[93m  1. Run: claude /login\033[0m")
            print(f"\033[93m  2. Then re-run this script\033[0m")
            sys.exit(1)

        if output_text.strip():
            with open(reasoning_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"=== BATCH 0 (start call) ===\n")
                f.write(f"{'='*60}\n")
                f.write(output_text.strip())
                f.write("\n\n")

    # -----------------------------------------------------------------------
    # Main batch loop
    # -----------------------------------------------------------------------
    for batch_start in range(start_iteration, n_iterations + 1, N_PARALLEL):
        iterations = [batch_start + s for s in range(N_PARALLEL)
                      if batch_start + s <= n_iterations]

        batch_first = iterations[0]
        batch_last = iterations[-1]
        n_slots = len(iterations)

        block_number = (batch_first - 1) // n_iter_block + 1
        iter_in_block_first = (batch_first - 1) % n_iter_block + 1
        iter_in_block_last = (batch_last - 1) % n_iter_block + 1
        is_block_end = any((it - 1) % n_iter_block + 1 == n_iter_block for it in iterations)

        # Block boundary: erase UCB
        if batch_first > 1 and (batch_first - 1) % n_iter_block == 0:
            if os.path.exists(ucb_path):
                os.remove(ucb_path)
                print(f"\033[93mblock boundary: deleted {ucb_path}\033[0m")

        print(f"\n\n\033[94m{'='*60}\033[0m")
        print(f"\033[94mBATCH: iterations {batch_first}-{batch_last} / {n_iterations}  (block {block_number})\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        # -------------------------------------------------------------------
        # PHASE 1: Run 4 simulations (cluster parallel or local sequential)
        # -------------------------------------------------------------------
        job_results = {}
        configs = {}

        # Load configs for all slots
        for slot_idx, iteration in enumerate(iterations):
            config = ParticleGraphConfig.from_yaml(config_paths[slot_idx])
            configs[slot_idx] = config

        if cluster_enabled:
            # --- Cluster mode: submit 4 jobs in parallel ---
            print(f"\n\033[93mPHASE 1: Submitting {n_slots} simulation jobs to cluster\033[0m")

            job_ids = {}
            for slot_idx, iteration in enumerate(iterations):
                jid = submit_cluster_job(
                    slot=slot_idx,
                    config_path=config_paths[slot_idx],
                    log_dir=exploration_dir,
                    root_dir=root_dir
                )
                if jid:
                    job_ids[slot_idx] = jid
                else:
                    job_results[slot_idx] = False

            # Wait for all submitted jobs
            if job_ids:
                print(f"\n\033[93mPHASE 1b: Waiting for {len(job_ids)} cluster jobs\033[0m")
                cluster_results = wait_for_cluster_jobs(job_ids, log_dir=exploration_dir, poll_interval=60)
                job_results.update(cluster_results)

            # Check for generation errors — attempt auto-repair instead of stopping
            for slot_idx in range(n_slots):
                if job_results.get(slot_idx) == False:
                    err_file = f"{exploration_dir}/cluster_sim_{slot_idx:02d}.err"
                    if not os.path.exists(err_file):
                        continue
                    try:
                        with open(err_file, 'r') as ef:
                            err_content = ef.read()
                    except Exception:
                        continue
                    if 'GENERATION SUBPROCESS ERROR' not in err_content:
                        continue

                    print(f"\033[91m  slot {slot_idx}: GENERATION ERROR detected — attempting auto-repair\033[0m")

                    code_files = [
                        'src/ParticleGraph/generators/PDE_Diffusiophoresis.py',
                        'src/ParticleGraph/generators/PDE_D.py',
                        'src/ParticleGraph/generators/graph_data_generator.py',
                    ]
                    generators_dir = Path(root_dir) / 'src/ParticleGraph/generators'
                    for pde_d_file in generators_dir.glob('PDE_D_*.py'):
                        rel_path = str(pde_d_file.relative_to(root_dir))
                        if rel_path not in code_files:
                            code_files.append(rel_path)
                    code_files.append('src/ParticleGraph/generators/utils.py')
                    # Include per-slot config file (LLM always modifies configs)
                    slot_config_rel = os.path.relpath(config_paths[slot_idx], root_dir)
                    code_files.append(slot_config_rel)
                    modified_code = get_modified_code_files(root_dir, code_files) if is_git_repo(root_dir) else [slot_config_rel]

                    if not modified_code:
                        print(f"\033[93m  slot {slot_idx}: no modified files to repair — skipping\033[0m")
                        continue

                    max_repair_attempts = 3
                    repaired = False
                    for attempt in range(max_repair_attempts):
                        print(f"\033[93m  slot {slot_idx}: repair attempt {attempt + 1}/{max_repair_attempts}\033[0m")
                        repair_prompt = f"""SIMULATION CRASHED - Please fix the code error.

Error traceback:
```
{err_content[-3000:]}
```

Modified files: {chr(10).join(f'- {root_dir}/{f}' for f in modified_code)}

Fix the bug. Do NOT make other changes."""

                        repair_cmd = [
                            'claude', '-p', repair_prompt,
                            '--output-format', 'text', '--max-turns', '10',
                            '--allowedTools', 'Read', 'Edit', 'Write'
                        ]
                        repair_result = subprocess.run(repair_cmd, cwd=root_dir, capture_output=True, text=True)
                        if 'CANNOT_FIX' in repair_result.stdout:
                            print(f"\033[91m  slot {slot_idx}: Claude cannot fix — stopping repair\033[0m")
                            break

                        # Resubmit repaired slot to cluster
                        print(f"\033[96m  slot {slot_idx}: resubmitting after repair\033[0m")
                        jid = submit_cluster_job(
                            slot=slot_idx,
                            config_path=config_paths[slot_idx],
                            log_dir=exploration_dir,
                            root_dir=root_dir
                        )
                        if jid:
                            retry_results = wait_for_cluster_jobs(
                                {slot_idx: jid}, log_dir=exploration_dir, poll_interval=60
                            )
                            if retry_results.get(slot_idx):
                                job_results[slot_idx] = True
                                repaired = True
                                print(f"\033[92m  slot {slot_idx}: repair successful!\033[0m")
                                break
                            # Reload error for next attempt
                            try:
                                with open(err_file, 'r') as ef:
                                    err_content = ef.read()
                            except Exception:
                                pass

                    if not repaired:
                        print(f"\033[91m  slot {slot_idx}: repair failed after {max_repair_attempts} attempts — skipping\033[0m")
                        if is_git_repo(root_dir):
                            rollback_files = list(code_files)
                            for fp in rollback_files:
                                try:
                                    subprocess.run(['git', 'checkout', 'HEAD', '--', fp],
                                                  cwd=root_dir, capture_output=True, timeout=10)
                                except Exception:
                                    pass

        else:
            # --- Local mode: run 4 simulations sequentially with error recovery ---
            print(f"\n\033[93mPHASE 1: Running {n_slots} simulations locally\033[0m")

            for slot_idx, iteration in enumerate(iterations):
                config = configs[slot_idx]
                print(f"\n\033[90m  slot {slot_idx} (iter {iteration}): running simulation...\033[0m")

                max_repair_attempts = 3
                success = False
                error_traceback = None

                for attempt in range(max_repair_attempts + 1):
                    success, error_traceback = run_simulation(
                        config_paths[slot_idx], root_dir, slot_names[slot_idx]
                    )

                    if success:
                        break

                    if attempt == 0:
                        print(f"\033[91m  slot {slot_idx}: simulation failed\033[0m")

                    code_files = [
                        'src/ParticleGraph/generators/PDE_Diffusiophoresis.py',
                        'src/ParticleGraph/generators/PDE_D.py',
                        'src/ParticleGraph/generators/graph_data_generator.py',
                    ]
                    generators_dir = Path(root_dir) / 'src/ParticleGraph/generators'
                    for pde_d_file in generators_dir.glob('PDE_D_*.py'):
                        rel_path = str(pde_d_file.relative_to(root_dir))
                        if rel_path not in code_files:
                            code_files.append(rel_path)
                    # Include per-slot config file (LLM always modifies configs)
                    slot_config_rel = os.path.relpath(config_paths[slot_idx], root_dir)
                    code_files.append(slot_config_rel)
                    modified_code = get_modified_code_files(root_dir, code_files) if is_git_repo(root_dir) else [slot_config_rel]

                    if not modified_code and attempt == 0:
                        break

                    if attempt < max_repair_attempts and modified_code:
                        print(f"\033[93m  slot {slot_idx}: repair attempt {attempt + 1}/{max_repair_attempts}\033[0m")
                        repair_prompt = f"""SIMULATION CRASHED - Please fix the code error.

Error traceback:
```
{error_traceback[-3000:] if error_traceback else 'No traceback'}
```

Modified files: {chr(10).join(f'- {root_dir}/{f}' for f in modified_code)}

Fix the bug. Do NOT make other changes."""

                        repair_cmd = [
                            'claude', '-p', repair_prompt,
                            '--output-format', 'text', '--max-turns', '10',
                            '--allowedTools', 'Read', 'Edit', 'Write'
                        ]
                        repair_result = subprocess.run(repair_cmd, cwd=root_dir, capture_output=True, text=True)
                        if 'CANNOT_FIX' in repair_result.stdout:
                            break

                if not success:
                    if is_git_repo(root_dir):
                        rollback_files = list(code_files)
                        for fp in rollback_files:
                            try:
                                subprocess.run(['git', 'checkout', 'HEAD', '--', fp],
                                              cwd=root_dir, capture_output=True, timeout=10)
                            except:
                                pass

                job_results[slot_idx] = success

        # -------------------------------------------------------------------
        # PHASE 2: Create montages + analysis logs for successful slots
        # -------------------------------------------------------------------
        print(f"\n\033[93mPHASE 2: Post-processing successful slots\033[0m")

        montage_paths = {}
        fig_dirs = {}

        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            if not job_results.get(slot, False):
                print(f"\033[90m  slot {slot} (iter {iteration}): skipping (failed)\033[0m")
                continue

            config = configs[slot]
            dataset_name = f"{base_config_name}/{config.dataset}"
            fig_dir = f"{root_dir}/graphs_data/{dataset_name}/Fig"
            fig_dirs[slot] = fig_dir

            # Create montage
            montage_path = f"{root_dir}/graphs_data/{dataset_name}/montage_iter_{iteration:03d}.png"
            if create_frame_montage(fig_dir, montage_path, n_frames=10):
                montage_paths[slot] = montage_path

            # Write analysis.log
            sim_log_path = f"{root_dir}/graphs_data/{dataset_name}/analysis.log"
            with open(analysis_log_paths[slot], 'w') as f:
                f.write(f"iteration: {iteration}\n")
                f.write(f"slot: {slot}\n")
                f.write(f"block: {block_number}\n")
                f.write(f"n_frames: {config.simulation.n_frames}\n")
                f.write(f"n_particles: {config.simulation.n_particles}\n")
                f.write(f"delta_t: {config.simulation.delta_t}\n")
                if os.path.exists(sim_log_path):
                    with open(sim_log_path, 'r') as sim_f:
                        f.write("\n# Simulation metrics:\n")
                        f.write(sim_f.read())

            # Generate video
            input_video = f"{root_dir}/graphs_data/{dataset_name}/input_{config.dataset}.mp4"
            video_path = f"{root_dir}/graphs_data/{dataset_name}/video_iter_{iteration:03d}.mp4"
            if os.path.exists(input_video):
                shutil.move(input_video, video_path)

            # Save exploration artifacts
            if slot in montage_paths:
                montage_dst = f"{exploration_dir}/montage/montage_iter_{iteration:03d}.png"
                shutil.copy2(montage_paths[slot], montage_dst)
            if os.path.exists(video_path):
                video_dst = f"{exploration_dir}/video/video_iter_{iteration:03d}.mp4"
                shutil.copy2(video_path, video_dst)
            if fig_dir and os.path.exists(fig_dir):
                png_files = sorted(Path(fig_dir).glob("Fig_0_*.png"))
                if png_files:
                    figure_dst = f"{exploration_dir}/figure/figure_iter_{iteration:03d}.png"
                    shutil.copy2(str(png_files[-1]), figure_dst)

            # Save config snapshot
            config_dst = f"{exploration_dir}/config/iter_{iteration:03d}_slot_{slot:02d}.yaml"
            shutil.copy2(config_paths[slot], config_dst)

        # -------------------------------------------------------------------
        # PHASE 3: Batch UCB update
        # -------------------------------------------------------------------
        print(f"\n\033[93mPHASE 3: Computing UCB scores\033[0m")

        # Build temp analysis with stub entries for all successful slots
        existing_content = ""
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r') as f:
                existing_content = f.read()

        stub_entries = ""
        for slot_idx, iteration in enumerate(iterations):
            if not job_results.get(slot_idx, False):
                continue
            log_path = analysis_log_paths[slot_idx]
            if not os.path.exists(log_path):
                continue
            with open(log_path, 'r') as f:
                log_content = f.read()
            score_m = re.search(r'score[=:]\s*(\d+)', log_content)
            if score_m and f'## Iter {iteration}:' not in existing_content:
                score_val = score_m.group(1)
                stub_entries += (
                    f"\n## Iter {iteration}: pending\n"
                    f"Node: id={iteration}, parent=root\n"
                    f"Score: {score_val}/10\n"
                )

        tmp_analysis = analysis_path + '.tmp_ucb'
        with open(tmp_analysis, 'w') as f:
            f.write(existing_content + stub_entries)

        compute_ucb_scores(
            tmp_analysis, ucb_path, c=ucb_c,
            current_log_path=None,
            current_iteration=batch_last,
            block_size=n_iter_block
        )
        os.remove(tmp_analysis)
        print(f"\033[92mUCB scores computed (c={ucb_c}): {ucb_path}\033[0m")

        # -------------------------------------------------------------------
        # PHASE 4: Claude analyzes results + proposes next 4 mutations
        # -------------------------------------------------------------------
        print(f"\n\033[93mPHASE 4: Claude analysis + next mutations\033[0m")

        slot_info_lines = []
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            status = "COMPLETED" if job_results.get(slot, False) else "FAILED"
            m_path = montage_paths.get(slot, "N/A")
            slot_info_lines.append(
                f"Slot {slot} (iteration {iteration}) [{status}]:\n"
                f"  Metrics: {analysis_log_paths[slot]}\n"
                f"  Montage: {m_path}\n"
                f"  Config: {config_paths[slot]}"
            )
        slot_info = "\n\n".join(slot_info_lines)

        block_end_marker = "\n>>> BLOCK END <<<" if is_block_end else ""
        parallel_ref = f"\nParallel instructions: {parallel_instruction_path}" if parallel_instruction_path else ""

        claude_prompt = f"""Batch iterations {batch_first}-{batch_last} / {n_iterations}
Block info: block {block_number}, iterations {iter_in_block_first}-{iter_in_block_last}/{n_iter_block} within block{block_end_marker}

PARALLEL MODE: Analyze {n_slots} results, then propose next {N_PARALLEL} mutations.

Instructions (follow all instructions): {instruction_path}{parallel_ref}
Working memory: {memory_path}
Full log (append only): {analysis_path}
UCB scores: {ucb_path}

{slot_info}

Analyze all {n_slots} results. For each successful slot, write a separate iteration entry
(## Iter N: ...) to the full log and memory file. Then edit all {N_PARALLEL} config files
to set up the next batch of {N_PARALLEL} experiments.

IMPORTANT: Do NOT change the 'dataset' field in any config — it must stay as-is for each slot."""

        # Add code file paths at block end
        if code_changes_enabled and is_block_end:
            claude_prompt += f"""

Code files you can modify (BLOCK END only - for next block):
- {root_dir}/src/ParticleGraph/generators/PDE_Diffusiophoresis*.py (mesh model variants)
- {root_dir}/src/ParticleGraph/generators/PDE_D.py (base particle model - minimal, prefer creating variants)
- {root_dir}/src/ParticleGraph/generators/PDE_D_*.py (particle model variants - create new or modify existing)
- {root_dir}/src/ParticleGraph/generators/graph_data_generator.py"""

        print("\033[93mClaude analysis...\033[0m")
        output_text = run_claude_cli(claude_prompt, root_dir)

        # Check OAuth
        if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
            print(f"\n\033[91mOAuth token expired at batch {batch_first}-{batch_last}\033[0m")
            print("\033[93m  1. Run: claude /login\033[0m")
            print(f"\033[93m  2. Re-run with --resume\033[0m")
            sys.exit(1)

        # Save reasoning
        if output_text.strip():
            with open(reasoning_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"=== Batch {batch_first}-{batch_last} ===\n")
                f.write(f"{'='*60}\n")
                f.write(output_text.strip())
                f.write("\n\n")

        # Track code modifications
        if code_changes_enabled and is_git_repo(root_dir):
            results = track_code_modifications(root_dir, batch_last)
            for file_path, success, message in results:
                if success:
                    print(f"\033[92mGit: {message}\033[0m")

        # Recompute UCB after Claude writes entries
        compute_ucb_scores(analysis_path, ucb_path, c=ucb_c,
                           current_log_path=None,
                           current_iteration=batch_last,
                           block_size=n_iter_block)

        # UCB tree visualization
        should_save_tree = (block_number == 1) or is_block_end
        if should_save_tree and os.path.exists(ucb_path):
            nodes = parse_ucb_scores_file(ucb_path)
            if nodes:
                tree_path = f"{exploration_dir}/tree/ucb_tree_iter_{batch_last:03d}.png"
                config = configs[0]
                sim_info = f"n_particles={config.simulation.n_particles}, n_frames={config.simulation.n_frames}"
                sim_info += f", delta_t={config.simulation.delta_t}"
                plot_ucb_tree(nodes, tree_path,
                             title=f"UCB Tree - Batch {batch_first}-{batch_last}",
                             simulation_info=sim_info)

        # Save memory at block end
        if is_block_end:
            memory_dst = f"{exploration_dir}/memory/block_{block_number:03d}_memory.md"
            if os.path.exists(memory_path):
                shutil.copy2(memory_path, memory_dst)
                print(f"\033[92msaved memory snapshot: {memory_dst}\033[0m")

        # Batch summary
        n_success = sum(1 for v in job_results.values() if v)
        n_failed = sum(1 for v in job_results.values() if not v)
        print(f"\n\033[92mBatch {batch_first}-{batch_last} complete: {n_success} succeeded, {n_failed} failed\033[0m")

    print(f"\n\033[92m=== Exploration complete ({n_iterations} iterations) ===\033[0m")
