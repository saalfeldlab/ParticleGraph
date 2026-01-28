import torch
import torch_geometric as pyg
from ParticleGraph.utils import to_numpy

class PDE_Diffusiophoresis(pyg.nn.MessagePassing):
    """
    Reaction-diffusion model for diffusiophoresis assembly simulation.
    Implements Brusselator model for two concentration fields (C₁, C₂).

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    increment : torch.Tensor
        The first derivative of two scalar fields C₁ and C₂
    """

    # PARAMS_DOC: Self-documenting parameter structure for LLM-guided exploration
    # This class attribute enables the LLM to understand and modify params_mesh correctly
    PARAMS_DOC = {
        "model_name": "Brusselator",
        "description": "Two-component reaction-diffusion system with cubic autocatalysis",
        "equations": {
            "dC1/dt": "D1 * ∇²C₁ + Da_c * (A - (B+1)*C₁ + C₁²*C₂) + χ * ∇²C₂ + noise - damping*(C₁-A)",
            "dC2/dt": "D2 * ∇²C₂ + Da_c * (B*C₁ - C₁²*C₂) - damping*(C₂-B/A)"
        },
        "params_mesh": [
            {
                "row": 0,
                "description": "C₁ field parameters",
                "slots": [
                    {"index": 0, "name": "D1", "description": "Diffusion coefficient for C₁", "typical_range": [0.01, 0.5]},
                    {"index": 1, "name": "Da_c", "description": "Damköhler number (reaction rate)", "typical_range": [1.0, 50.0]},
                    {"index": 2, "name": "A", "description": "Brusselator parameter A (feed concentration)", "typical_range": [0.5, 5.0]},
                    {"index": 3, "name": "B", "description": "Brusselator parameter B (reaction strength)", "typical_range": [1.0, 10.0]},
                    {"index": 4, "name": "mu", "description": "Morphological parameter (unused)", "typical_range": [0.01, 0.1]},
                    {"index": 5, "name": "chi", "description": "Cross-diffusion coefficient", "typical_range": [-0.1, 0.1]}
                ]
            },
            {
                "row": 1,
                "description": "C₂ field parameters",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coefficient for C₂", "typical_range": [0.1, 1.0]},
                    {"index": 5, "name": "noise_amplitude", "description": "Stochastic noise for symmetry breaking", "typical_range": [0.0, 0.01]}
                ]
            }
        ],
        "pattern_regimes": {
            "spots": "B/A > 1 + A², small D1/D2 ratio",
            "stripes": "B/A ≈ 1 + A², moderate D1/D2",
            "labyrinth": "B/A < 1 + A², large domain"
        }
    }

    def __init__(self, aggr_type='add', bc_dpos=None, p=None):
        super(PDE_Diffusiophoresis, self).__init__(aggr='add')  # "add" aggregation

        self.bc_dpos = bc_dpos

        # Initialize parameters directly from p tensor
        # C1 parameters
        self.D1 = p[0, 0]       # Diffusion coefficient for C₁ (or D1_x if aniso active)
        self.Da_c = p[0, 1]     # Damköhler number
        self.A = p[0, 2]        # Brusselator parameter A
        self.B = p[0, 3]        # Brusselator parameter B

        # Add morphological parameter μ if available, otherwise use default
        if p[0].size(0) > 4:
            self.mu = p[0, 4]   # Morphological parameter (unused in current model)
        else:
            self.mu = torch.tensor(0.04, device=p.device)  # Default for hexagonal patterns

        # Cross-diffusion coefficient χ (chi) - C1 follows C2 gradients
        # Positive χ: C1 diffuses toward high C2; Negative χ: C1 diffuses away from high C2
        # Uses slot 5 in params_mesh[0]
        if p[0].size(0) > 5:
            self.chi = p[0, 5]   # Cross-diffusion coefficient
        else:
            self.chi = torch.tensor(0.0, device=p.device)  # Default: no cross-diffusion

        # STOCHASTIC NOISE AMPLITUDE for symmetry breaking
        # noise_amplitude controls strength of random perturbations to break eigenmode lock
        # Literature: García-Ojalvo et al. 1993 - noise-induced pattern transitions in Turing systems
        # Uses slot 5 in params_mesh[1] (repurposed from aniso to noise)
        if p[1].size(0) > 5:
            self.noise_amplitude = p[1, 5]  # Noise amplitude for C1 field
        else:
            self.noise_amplitude = torch.tensor(0.0, device=p.device)  # Default: no noise

        # C2 parameters
        self.D2 = p[1, 0]       # Diffusion coefficient for C₂

        # Store coefficient for later use
        self.coeff = p

        # Print initialized parameters for verification
        print(f"initialized PDE_Diffusiophoresis with parameters:")
        print(f"C₁: D={self.D1.item():.3f}, C₂: D={self.D2.item():.3f}, Da_c={self.Da_c.item():.3f}, A={self.A.item():.3f}, B={self.B.item():.3f}, μ={self.mu.item():.3f}, χ={self.chi.item():.3f}, noise={self.noise_amplitude.item():.4f}")
    
    def forward(self, data):
        """
        Update the concentration fields using the Brusselator reaction-diffusion model

        Parameters
        ----------
        data : torch_geometric.data.Data
            Contains:
            - x: Node features [n_nodes, n_features]
              Field values are at indices 6 and 7 (C₁ and C₂)
            - edge_index: Connectivity [2, n_edges]
            - edge_attr: Edge attributes (Laplacian coefficients) [n_edges]

        Returns
        -------
        d_C : torch.Tensor
            The derivatives of C₁ and C₂ [n_nodes, 2]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Extract field values
        C1 = x[:, 6:7]  # C₁ at index 6
        C2 = x[:, 7:8]  # C₂ at index 7

        # Isotropic diffusion - standard Laplacian
        laplacian_C1 = self.propagate(edge_index, u=C1, edge_attr=edge_attr, D_ratio=None)
        diff_C1 = self.D1 * laplacian_C1

        # C2 diffusion
        laplacian_C2 = self.propagate(edge_index, u=C2, edge_attr=edge_attr, D_ratio=None)
        diff_C2 = self.D2 * laplacian_C2

        # Store Laplacians for potential use elsewhere
        self.laplacian_C1 = laplacian_C1
        self.laplacian_C2 = laplacian_C2

        # Compute reaction terms (Brusselator model)
        R1 = self.Da_c * (self.A - (self.B+1)*C1 + C1*C1*C2)
        R2 = self.Da_c * (self.B*C1 - C1*C1*C2)

        # Cross-diffusion: C1 diffuses in response to C2 Laplacian
        # Positive χ: C1 accumulates where C2 is high (chemotaxis-like)
        # This can break labyrinthine symmetry and select stripes
        cross_diff_C1 = self.chi * laplacian_C2

        # STOCHASTIC NOISE for eigenmode symmetry breaking
        # Noise amplitude is scaled by sqrt(delta_t) for proper Langevin dynamics
        # Literature: García-Ojalvo et al. 1993 - noise can destabilize degenerate eigenmodes
        if self.noise_amplitude.item() > 0:
            noise_C1 = self.noise_amplitude * torch.randn_like(C1)
        else:
            noise_C1 = 0.0

        damping = 0.005
        dC1 = diff_C1 + R1 + cross_diff_C1 + noise_C1 - damping * (C1 - self.A)
        dC2 = diff_C2 + R2 - damping * (C2 - self.B/self.A)

        # Combine derivatives
        d_C = torch.cat([dC1, dC2], dim=1)

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