import torch
import torch_geometric as pyg
from ParticleGraph.utils import to_numpy

class PDE_Diffusiophoresis_GM(pyg.nn.MessagePassing):
    """
    Gierer-Meinhardt reaction-diffusion model for diffusiophoresis assembly simulation.

    Literature:
    - Gierer, A. & Meinhardt, H. (1972) "A theory of biological pattern formation",
      Kybernetik 12:30-39
    - Meinhardt, H. (1982) "Models of Biological Pattern Formation", Academic Press
    - Koch, A.J. & Meinhardt, H. (1994) "Biological pattern formation: from basic mechanisms
      to complex structures", Reviews of Modern Physics 66(4):1481-1507

    The Gierer-Meinhardt model is a classical activator-inhibitor system with
    ratio-dependent activation:
    - a: Activator (short-range autocatalysis)
    - h: Inhibitor (long-range suppression)

    Key difference from Brusselator: The nonlinearity is a^2/h (ratio) rather than
    a^2*b (product). This ratio dependence creates:
    - Self-regulating pattern amplitude (inhibitor prevents unbounded growth)
    - Sharp spike-like activator peaks (vs smooth Brusselator spots)
    - Different bifurcation structure from all other models tested

    Equations:
        da/dt = Da * nabla^2 a + rho * a^2 / (h * (1 + kappa * a^2)) - mu_a * a + sigma_a
        dh/dt = Dh * nabla^2 h + rho * a^2 - mu_h * h + sigma_h

    Pattern types depend on parameters:
    - Da/Dh ratio: Controls pattern type (spots vs stripes)
    - rho: Production rate (higher = stronger patterns)
    - mu_a, mu_h: Decay rates (ratio controls steady state)
    - sigma_a: Activator source (background production, prevents complete decay)
    - kappa: Saturation coefficient (limits maximum activator concentration)

    Turing instability requires: Da << Dh (short-range activation, long-range inhibition)

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    increment : torch.Tensor
        The first derivative of two scalar fields a and h (mapped to C1 and C2)
    """

    # PARAMS_DOC: Self-documenting parameter structure for LLM-guided exploration
    PARAMS_DOC = {
        "model_name": "Gierer-Meinhardt",
        "description": "Two-component activator-inhibitor system with ratio-dependent activation",
        "literature": "Gierer & Meinhardt (1972) Kybernetik 12:30-39; Koch & Meinhardt (1994) Rev Mod Phys 66:1481",
        "equations": {
            "da/dt": "Da * nabla^2 a + rho * a^2 / (h * (1 + kappa * a^2)) - mu_a * a + sigma_a",
            "dh/dt": "Dh * nabla^2 h + rho * a^2 - mu_h * h + sigma_h"
        },
        "params_mesh": [
            {
                "row": 0,
                "description": "Activator (a) field parameters",
                "slots": [
                    {"index": 0, "name": "Da", "description": "Diffusion coefficient for activator", "typical_range": [0.005, 0.1]},
                    {"index": 1, "name": "rho", "description": "Production rate (autocatalysis strength)", "typical_range": [0.01, 1.0]},
                    {"index": 2, "name": "mu_a", "description": "Activator decay rate", "typical_range": [0.01, 0.1]},
                    {"index": 3, "name": "sigma_a", "description": "Activator source (background production)", "typical_range": [0.001, 0.05]},
                    {"index": 4, "name": "kappa", "description": "Saturation coefficient (limits peak height)", "typical_range": [0.0, 0.5]},
                    {"index": 5, "name": "time_scale", "description": "Overall time scaling factor", "typical_range": [1.0, 100.0]}
                ]
            },
            {
                "row": 1,
                "description": "Inhibitor (h) field parameters",
                "slots": [
                    {"index": 0, "name": "Dh", "description": "Diffusion coefficient for inhibitor", "typical_range": [0.1, 1.0]},
                    {"index": 1, "name": "mu_h", "description": "Inhibitor decay rate", "typical_range": [0.01, 0.1]},
                    {"index": 2, "name": "sigma_h", "description": "Inhibitor source (background)", "typical_range": [0.0, 0.01]},
                    {"index": 3, "name": "unused_3", "description": "Reserved slot", "typical_range": [0, 0]},
                    {"index": 4, "name": "unused_4", "description": "Reserved slot", "typical_range": [0, 0]},
                    {"index": 5, "name": "unused_5", "description": "Reserved slot", "typical_range": [0, 0]}
                ]
            }
        ],
        "pattern_regimes": {
            "spots": "Da << Dh (e.g. Da=0.01, Dh=0.5), high rho, low kappa -> sharp isolated spots",
            "stripes": "Da/Dh closer to 1 (e.g. Da=0.05, Dh=0.2), moderate rho -> stripe/labyrinth",
            "weak_patterns": "low rho or high sigma_a -> weak modulation over uniform background",
            "spike_splitting": "very high rho, low kappa -> spike insertion/splitting dynamics"
        }
    }

    def __init__(self, aggr_type='add', bc_dpos=None, p=None):
        super(PDE_Diffusiophoresis_GM, self).__init__(aggr='add')

        self.bc_dpos = bc_dpos

        # Activator parameters (row 0)
        self.Da = p[0, 0]            # Diffusion coefficient for activator
        self.rho = p[0, 1]           # Production rate
        self.mu_a = p[0, 2]          # Activator decay rate

        # Source term sigma_a
        if p[0].size(0) > 3:
            self.sigma_a = p[0, 3]
        else:
            self.sigma_a = torch.tensor(0.01, device=p.device)

        # Saturation coefficient kappa
        if p[0].size(0) > 4:
            self.kappa = p[0, 4]
        else:
            self.kappa = torch.tensor(0.0, device=p.device)

        # Time scaling factor
        if p[0].size(0) > 5:
            self.time_scale = p[0, 5]
        else:
            self.time_scale = torch.tensor(1.0, device=p.device)

        # Inhibitor parameters (row 1)
        self.Dh = p[1, 0]            # Diffusion coefficient for inhibitor

        # Inhibitor decay rate
        if p[1].size(0) > 1:
            self.mu_h = p[1, 1]
        else:
            self.mu_h = torch.tensor(0.02, device=p.device)

        # Inhibitor source term
        if p[1].size(0) > 2:
            self.sigma_h = p[1, 2]
        else:
            self.sigma_h = torch.tensor(0.0, device=p.device)

        # Store coefficient for later use
        self.coeff = p

        # Required for compatibility with base class expectations
        # graph_data_generator.py uses model.A and model.B for initial conditions
        # For GM: a starts near sigma_a/mu_a (steady state without diffusion)
        # h starts near rho * (sigma_a/mu_a)^2 / mu_h
        a_ss = self.sigma_a / self.mu_a if self.mu_a.item() > 0 else torch.tensor(1.0, device=p.device)
        self.A = torch.clamp(a_ss, min=0.1, max=10.0)  # Initial activator value
        self.B = torch.tensor(0.5, device=p.device)  # Initial inhibitor scale

        # Print initialized parameters for verification
        print(f"initialized PDE_Diffusiophoresis_GM with parameters:")
        print(f"a: Da={self.Da.item():.4f}, h: Dh={self.Dh.item():.4f}")
        print(f"rho={self.rho.item():.4f}, mu_a={self.mu_a.item():.4f}, mu_h={self.mu_h.item():.4f}")
        print(f"sigma_a={self.sigma_a.item():.4f}, sigma_h={self.sigma_h.item():.4f}, kappa={self.kappa.item():.4f}")
        print(f"time_scale={self.time_scale.item():.1f}")
        print(f"Da/Dh ratio={self.Da.item()/max(self.Dh.item(), 1e-6):.4f} (<<1 needed for Turing)")

    def forward(self, data):
        """
        Update the concentration fields using the Gierer-Meinhardt reaction-diffusion model

        Parameters
        ----------
        data : torch_geometric.data.Data
            Contains:
            - x: Node features [n_nodes, n_features]
              Field values are at indices 6 and 7 (a and h mapped to C1 and C2)
            - edge_index: Connectivity [2, n_edges]
            - edge_attr: Edge attributes (Laplacian coefficients) [n_edges]

        Returns
        -------
        d_C : torch.Tensor
            The derivatives of a and h [n_nodes, 2]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Extract field values (a = C1, h = C2)
        a = x[:, 6:7]  # Activator at index 6
        h = x[:, 7:8]  # Inhibitor at index 7

        # Compute Laplacians for diffusion
        laplacian_a = self.propagate(edge_index, u=a, edge_attr=edge_attr, D_ratio=None)
        laplacian_h = self.propagate(edge_index, u=h, edge_attr=edge_attr, D_ratio=None)

        # Store Laplacians for potential use elsewhere
        self.laplacian_C1 = laplacian_a
        self.laplacian_C2 = laplacian_h

        # Diffusion terms
        diff_a = self.Da * laplacian_a
        diff_h = self.Dh * laplacian_h

        # Clamp fields to positive values (GM requires a>0, h>0)
        a_safe = torch.clamp(a, min=1e-4)
        h_safe = torch.clamp(h, min=1e-4)

        # Gierer-Meinhardt reaction terms
        # Activator: autocatalytic production a^2/h, linear decay, background source
        # With optional saturation: a^2 / (h * (1 + kappa * a^2))
        a_squared = a_safe * a_safe
        if self.kappa.item() > 0:
            activation = self.rho * a_squared / (h_safe * (1.0 + self.kappa * a_squared))
        else:
            activation = self.rho * a_squared / h_safe

        R_a = activation - self.mu_a * a_safe + self.sigma_a

        # Inhibitor: produced by activator, linear decay, background source
        R_h = self.rho * a_squared - self.mu_h * h_safe + self.sigma_h

        # Combine diffusion and reaction with time scaling
        da = self.time_scale * (diff_a + R_a)
        dh = self.time_scale * (diff_h + R_h)

        # Combine derivatives
        d_C = torch.cat([da, dh], dim=1)

        return d_C


    def message(self, u_j, edge_attr, D_ratio=None):
        """
        Message function for Laplacian computation.
        """
        if D_ratio is not None:
            L = edge_attr[:, None] * D_ratio[:, None] * u_j
        else:
            L = edge_attr[:, None] * u_j
        return L
