import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def to_numpy(x):
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)

def scan_and_plot_signal_activity(root_dir):
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        print(f"Analyzing folder: {folder}")
        x_file = os.path.join(folder_path, 'x_list_0.npy')
        if not os.path.exists(x_file):
            continue
        try:
            x = np.load(x_file)
        except Exception as e:
            print(f"Could not load {x_file}: {e}")
            continue
        # Always extract signal as activity = x[:, :, 6:7].squeeze()
        try:
            activity = x[:, :, 6:7].squeeze()
        except Exception as e:
            print(f"Error extracting activity from {x_file}: {e}")
            continue

        n_neurons = activity.shape[0]
        n_time = activity.shape[1]
        # Compute mean and std for activity
        mean_activity = np.mean(activity)
        std_activity = np.std(activity)
        print(f"Mean: {mean_activity:.3f}, Std: {std_activity:.3f}")
        n = np.random.permutation(n_neurons)
        try:
            import matplotlib
            from matplotlib import rc
            plt.rcParams['text.usetex'] = True
            rc('font', **{'family': 'serif', 'serif': ['Palatino']})
        except Exception:
            plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif", "serif"]
            plt.rcParams["font.family"] = "serif"
        fig, ax = plt.subplots(figsize=(15, 10))
        for i in range(10):
            trace = activity[n[i].astype(int), :] + i * std_activity * 4 - mean_activity
            ax.plot(trace, linewidth=2)
        ax.set_xlabel(r'time', fontsize=64)
        ax.set_ylabel(r'$x_{i}$', fontsize=64)
        ax.set_xticks([0, n_time-1])
        ax.set_yticks([0, 10])
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
        plt.tight_layout()
        save_path = os.path.join(folder_path, 'firing_rate.tif')
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    scan_and_plot_signal_activity("/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/signal")
