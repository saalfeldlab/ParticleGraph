import torch
import torch.nn as nn
import tinycudann as tcnn


class FusedMLP(nn.Module):
    """
    Fully fused MLP using tinycudann.Network.
    No fallbacks, assumes tinycudann is available.

    Args:
        in_dim (int): input dimension
        hidden_dim (int): hidden layer width
        out_dim (int): output dimension
        n_hidden (int): number of hidden layers (e.g., 2 => 3 total linear layers)
        activation (str): 'ReLU', 'SiLU', 'GELU', etc.
        output_activation (str|None): e.g. 'None', 'ReLU'
        use_fp16 (bool): cast inputs to float16 for maximum throughput
        device (torch.device | str | None): device to place the module on
    """

    def __init__(self, in_dim, hidden_dim, out_dim, n_hidden=2,
                 activation='ReLU', output_activation=None,
                 use_fp16=True, device=None):
        super().__init__()
        self.use_fp16 = use_fp16

        self.net = tcnn.Network(
            n_input_dims=in_dim,
            n_output_dims=out_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": activation,
                "output_activation": output_activation or "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": n_hidden,
            },
        )

        # Move to device if specified
        if device is not None:
            self.to(device)

    def forward(self, x):
        # tinycudann prefers contiguous float16/float32 CUDA tensors
        if x.is_cuda and self.use_fp16:
            x = x.to(torch.float16)
        else:
            x = x.to(torch.float32)
        return self.net(x)
