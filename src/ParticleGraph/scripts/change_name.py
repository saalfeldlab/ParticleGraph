import os
import shutil
from pathlib import Path


def rename_adjacency_files(base_paths):
    """
    Rename adjacency.pt to connectivity.pt in all subfolders of given base paths,
    while keeping a backup as adjacency.pt_
    """

    for base_path in base_paths:
        base_path = Path(base_path)

        if not base_path.exists():
            print(f"Warning: Base path does not exist: {base_path}")
            continue

        print(f"Processing base path: {base_path}")

        # Walk through all subdirectories
        for root, dirs, files in os.walk(base_path):
            root_path = Path(root)
            adjacency_file = root_path / "adjacency.pt"

            if adjacency_file.exists():
                backup_file = root_path / "adjacency.pt_"
                connectivity_file = root_path / "connectivity.pt"

                try:
                    # Create backup
                    shutil.copy2(adjacency_file, backup_file)
                    print(f"Created backup: {backup_file}")

                    # Rename to connectivity.pt
                    adjacency_file.rename(connectivity_file)
                    print(f"Renamed: {adjacency_file} -> {connectivity_file}")

                except Exception as e:
                    print(f"Error processing {adjacency_file}: {e}")

    print("Renaming process completed!")


if __name__ == "__main__":
    # Define the base paths
    base_paths = [
        "/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/signal",
        "/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/CElegans"
    ]

    # Confirm before proceeding
    print("This script will:")
    print("1. Find all 'adjacency.pt' files in subfolders of the specified directories")
    print("2. Create a backup copy named 'adjacency.pt_'")
    print("3. Rename the original file to 'connectivity.pt'")
    print()
    print("Base paths to process:")
    for path in base_paths:
        print(f"  - {path}")
    print()

    response = input("Do you want to proceed? (y/N): ")
    if response.lower() == 'y':
        rename_adjacency_files(base_paths)
    else:
        print("Operation cancelled.")