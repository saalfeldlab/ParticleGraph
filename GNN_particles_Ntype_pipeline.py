import sys
import os
import argparse
import time
from flyvis.utils.compute_cloud_utils import LSFManager, wait_for_single


def run_task(task: str, config: str, dry: bool = False):
    """Run a task on the cluster with job monitoring."""
    cluster_manager = LSFManager()
    cluster_command = cluster_manager.get_submit_command(
        job_name="particle-graph",
        n_cpus=4,
        output_file=f"particle-graph-{task}-{int(time.time())}.out",
        gpu="num=1",
        queue="gpu_h100",
    )
    script_path = os.path.abspath("GNN_particles_Ntype.py")
    script_command = cluster_manager.get_script_part(
        f"{sys.executable} {script_path} --option {task} {config}"
    )
    print(f"Running {task} with config {config}")
    if not dry:
        job_id = cluster_manager.run_job(cluster_command + script_command)
    else:
        job_id = "dry"
        print(f"Dry run: {cluster_command + script_command}")
    wait_for_single(job_id, dry=dry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ParticleGraph")
    parser.add_argument(
        "-t",
        "--tasks",
        help="Tasks to run",
        choices=["generate", "train", "test"],
        default=["generate", "train", "test"],
        nargs="+",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Config file",
        default="fly_N9_18_4_1_alt_prtrnd_model",
    )
    parser.add_argument("--dry", action="store_true", default=False)
    args = parser.parse_args()
    for t in args.tasks:
        run_task(t, args.config, dry=args.dry)
