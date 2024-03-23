import torch
from torch_geometric.nn import MessagePassing


class PDE_Z(MessagePassing):
    """
    Dummy class that returns 0 as the acceleration of the particles.

    Returns
    -------
    pred : float
        an array of zeros (dimension 2)
    """

    def __init__(self):
        super(PDE_Z, self).__init__(aggr='add')

    def forward(self, data):

        pred = torch.zeros_like(data.x)

        return pred[:,0:2]
