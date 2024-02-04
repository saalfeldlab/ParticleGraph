import numpy as np
import torch


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to convert.

    Returns:
        np.ndarray: The NumPy array.
    """
    return tensor.detach().cpu().numpy()


class cc:

     """
    Convert index to color.
    """

    def __init__(self, model_config):
        self.model_config = model_config
        self.model = model_config['model']
        if model_config['cmap'] == 'tab10':
            self.nmap = 8
        else:
            self.nmap = model_config['nparticle_types']

        self.bMesh = 'Mesh' in model_config['model']

    def color(self, index):

        if self.model == 'ElecParticles':

            if index == 0:
                index = (0, 0, 1)
            elif index == 1:
                index = (0, 0.5, 0.75)
            elif index == 2:
                index = (1, 0, 0)
            elif index == 3:
                index = (0.75, 0, 0)
            else
                index = (0, 0, 0)
            return (index)
        elif self.bMesh:
            if index == 0:
                index = (0, 0, 0)
            else:
                color_map = plt.colormaps.get_cmap(self.model_config['cmap'])
                index = color_map(index / self.nmap)

        else:
            # color_map = plt.cm.get_cmap(self.model_config['cmap'])
            color_map = plt.colormaps.get_cmap(self.model_config['cmap'])
            index = color_map(index / self.nmap)

        return index
