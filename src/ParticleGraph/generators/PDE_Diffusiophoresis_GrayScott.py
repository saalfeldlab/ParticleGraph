import torch
import torch_geometric as pyg
from ParticleGraph.utils import to_numpy

class PDE_Diffusiophoresis_GrayScott(pyg.nn.MessagePassing):
    """
    Gray-Scott reaction-diffusion model for diffusiophoresis assembly simulation.

    Literature: Pearson (1993) "Complex Patterns in a Simple System", Science 261:189-192

    The Gray-Scott model describes two chemicals U (substrate) and V (autocatalyst):
    - Reaction: U + 2V → 3V (autocatalytic conversion)
    - U is continuously fed, V is continuously removed (kill rate)

    Pattern types (from Pearson 1993):
    - α (alpha): Spots - F≈0.02, k≈0.05
    - β (beta): Spots that replicate - F≈0.025, k≈0.05
    - γ (gamma): Worms/stripes - F≈0.035, k≈0.06
    - δ (delta): Mitosis (spot splitting) - F≈0.03, k≈0.055
    - ε (epsilon): Chaos - F≈0.02, k≈0.055
    - λ (lambda): Stripes/labyrinths - F≈0.04, k≈0.065

    Equations:
        dU/dt = Du * ∇²U - U*V² + F*(1-U)
        dV/dt = Dv * ∇²V + U*V² - (F+k)*V

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    increment : torch.Tensor
        The first derivative of two scalar fields U and V (mapped to C₁ and C₂)
    """

    # PARAMS_DOC: Self-documenting parameter structure for LLM-guided exploration
    PARAMS_DOC = {
        "model_name": "Gray-Scott",
        "description": "Two-component autocatalytic reaction-diffusion system (Pearson 1993)",
        "literature": "Pearson (1993) Science 261:189-192 'Complex Patterns in a Simple System'",
        "equations": {
            "dU/dt": "Du * ∇²U - U*V² + F*(1-U)",
            "dV/dt": "Dv * ∇²V + U*V² - (F+k)*V"
        },
        "params_mesh": [
            {
                "row": 0,
                "description": "U field (substrate) parameters",
                "slots": [
                    {"index": 0, "name": "Du", "description": "Diffusion coefficient for U", "typical_range": [0.16, 0.24]},
                    {"index": 1, "name": "F", "description": "Feed rate (substrate replenishment)", "typical_range": [0.01, 0.08]},
                    {"index": 2, "name": "k", "description": "Kill rate (autocatalyst removal)", "typical_range": [0.04, 0.07]},
                    {"index": 3, "name": "time_scale", "description": "Time scaling factor for dynamics", "typical_range": [1.0, 100.0]},
                    {"index": 4, "name": "unused_4", "description": "Reserved slot", "typical_range": [0, 0]},
                    {"index": 5, "name": "unused_5", "description": "Reserved slot", "typical_range": [0, 0]}
                ]
            },
            {
                "row": 1,
                "description": "V field (autocatalyst) parameters",
                "slots": [
                    {"index": 0, "name": "Dv", "description": "Diffusion coefficient for V", "typical_range": [0.04, 0.12]},
                    {"index": 1, "name": "unused_1", "description": "Reserved slot", "typical_range": [0, 0]},
                    {"index": 2, "name": "unused_2", "description": "Reserved slot", "typical_range": [0, 0]},
                    {"index": 3, "name": "unused_3", "description": "Reserved slot", "typical_range": [0, 0]},
                    {"index": 4, "name": "unused_4", "description": "Reserved slot", "typical_range": [0, 0]},
                    {"index": 5, "name": "unused_5", "description": "Reserved slot", "typical_range": [0, 0]}
                ]
            }
        ],
        "pattern_regimes": {
            "alpha_spots": "F=0.010-0.022, k=0.045-0.052 → isolated spots",
            "beta_replicating": "F=0.022-0.030, k=0.048-0.054 → spot replication",
            "gamma_worms": "F=0.030-0.040, k=0.057-0.063 → worm-like structures",
            "delta_mitosis": "F=0.028-0.035, k=0.053-0.058 → spot splitting",
            "epsilon_chaos": "F=0.018-0.025, k=0.052-0.058 → chaotic dynamics",
            "lambda_stripes": "F=0.038-0.050, k=0.060-0.068 → stripes/labyrinths"
        }
    }

    def __init__(self, aggr_type='add', bc_dpos=None, p=None):
        super(PDE_Diffusiophoresis_GrayScott, self).__init__(aggr='add')

        self.bc_dpos = bc_dpos

        # U field parameters (row 0)
        self.Du = p[0, 0]       # Diffusion coefficient for U
        self.F = p[0, 1]        # Feed rate
        self.k = p[0, 2]        # Kill rate

        # Time scaling factor - Gray-Scott dynamics are slower than Brusselator
        # Need to scale up to observe patterns in similar number of frames
        if p[0].size(0) > 3:
            self.time_scale = p[0, 3]
        else:
            self.time_scale = torch.tensor(1.0, device=p.device)

        # V field parameters (row 1)
        self.Dv = p[1, 0]       # Diffusion coefficient for V

        # Store coefficient for later use
        self.coeff = p

        # Print initialized parameters for verification
        print(f"Initialized PDE_Diffusiophoresis_GrayScott with parameters:")
        print(f"U: Du={self.Du.item():.4f}, V: Dv={self.Dv.item():.4f}")
        print(f"F={self.F.item():.4f}, k={self.k.item():.4f}, time_scale={self.time_scale.item():.1f}")

        # Identify pattern regime based on F, k values
        F_val, k_val = self.F.item(), self.k.item()
        if F_val < 0.022 and k_val < 0.052:
            regime = "alpha (spots)"
        elif F_val < 0.030 and k_val < 0.054:
            regime = "beta (replicating spots)"
        elif F_val < 0.040 and k_val < 0.063:
            regime = "gamma (worms/stripes)"
        elif F_val > 0.038 and k_val > 0.060:
            regime = "lambda (stripes/labyrinths)"
        else:
            regime = "mixed/transition"
        print(f"Expected pattern regime: {regime}")

        # Initial condition values for compatibility with graph_data_generator
        # For Gray-Scott: U starts near 1 (fed state), V starts near 0
        # These are named A and B for compatibility with the Brusselator interface
        self.A = torch.tensor(1.0, device=p.device)  # Initial U value
        self.B = torch.tensor(0.0, device=p.device)  # Initial V value

    def forward(self, data):
        """
        Update the concentration fields using the Gray-Scott reaction-diffusion model

        Parameters
        ----------
        data : torch_geometric.data.Data
            Contains:
            - x: Node features [n_nodes, n_features]
              Field values are at indices 6 and 7 (U and V mapped to C₁ and C₂)
            - edge_index: Connectivity [2, n_edges]
            - edge_attr: Edge attributes (Laplacian coefficients) [n_edges]

        Returns
        -------
        d_C : torch.Tensor
            The derivatives of U and V [n_nodes, 2]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Extract field values (U = C1, V = C2)
        U = x[:, 6:7]  # U (substrate) at index 6
        V = x[:, 7:8]  # V (autocatalyst) at index 7

        # Compute Laplacians for diffusion
        laplacian_U = self.propagate(edge_index, u=U, edge_attr=edge_attr, D_ratio=None)
        laplacian_V = self.propagate(edge_index, u=V, edge_attr=edge_attr, D_ratio=None)

        # Store Laplacians for potential use elsewhere
        self.laplacian_C1 = laplacian_U
        self.laplacian_C2 = laplacian_V

        # Diffusion terms
        diff_U = self.Du * laplacian_U
        diff_V = self.Dv * laplacian_V

        # Gray-Scott reaction terms (Pearson 1993)
        # dU/dt = Du*∇²U - U*V² + F*(1-U)
        # dV/dt = Dv*∇²V + U*V² - (F+k)*V
        UV2 = U * V * V  # Autocatalytic reaction term

        R_U = -UV2 + self.F * (1.0 - U)     # U reaction: consumed by reaction, fed from reservoir
        R_V = UV2 - (self.F + self.k) * V   # V reaction: produced by reaction, removed by kill rate

        # Combine diffusion and reaction with time scaling
        dU = self.time_scale * (diff_U + R_U)
        dV = self.time_scale * (diff_V + R_V)

        # Combine derivatives
        d_C = torch.cat([dU, dV], dim=1)

        return d_C


    def message(self, u_j, edge_attr, D_ratio=None):
        """
        Message function for Laplacian computation.
        When D_ratio is provided, applies directional weighting for anisotropic diffusion.
        """
        if D_ratio is not None:
            # Anisotropic case: weight by directional diffusion ratio
            L = edge_attr[:, None] * D_ratio[:, None] * u_j
        else:
            # Isotropic case: standard Laplacian
            L = edge_attr[:, None] * u_j
        return L
