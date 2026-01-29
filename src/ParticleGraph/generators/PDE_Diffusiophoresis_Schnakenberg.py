import torch
import torch_geometric as pyg
from ParticleGraph.utils import to_numpy

class PDE_Diffusiophoresis_Schnakenberg(pyg.nn.MessagePassing):
    """
    Schnakenberg reaction-diffusion model for diffusiophoresis assembly simulation.

    Literature:
    - Schnakenberg (1979) "Simple chemical reaction systems with limit cycle behaviour",
      Journal of Theoretical Biology, 81(3):389-400
    - Murray (2003) "Mathematical Biology II: Spatial Models and Biomedical Applications",
      Springer, Chapter 2 (Turing instability analysis)

    The Schnakenberg model is a minimal Turing system with quadratic autocatalysis:
    - u: Activator (autocatalytic species)
    - v: Substrate (consumed by autocatalysis)

    Equations:
        du/dt = Du * nabla^2 u + gamma * (a - u + u^2 * v)
        dv/dt = Dv * nabla^2 v + gamma * (b - u^2 * v)

    Steady state: u* = a + b, v* = b / (a+b)^2

    Turing instability condition (Murray 2003):
    - Requires Du << Dv (short-range activation, long-range inhibition)
    - d = Dv/Du > 1 (diffusion ratio)
    - The system becomes Turing unstable when:
      f_u * g_v - f_v * g_u > 0 (trace condition)
      d * f_u + g_v > 2 * sqrt(d * (f_u * g_v - f_v * g_u)) (diffusion-driven instability)

    Key difference from Brusselator: simpler nonlinearity (u^2*v only), different
    pattern selection properties. Schnakenberg tends to produce more regular spots
    in the deeply unstable regime and can transition to stripes at the boundary.

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    increment : torch.Tensor
        The first derivative of two scalar fields u and v (mapped to C1 and C2)
    """

    PARAMS_DOC = {
        "model_name": "Schnakenberg",
        "description": "Two-component activator-substrate system with quadratic autocatalysis",
        "literature": "Schnakenberg (1979) J Theor Biol 81:389-400; Murray (2003) Math Bio II",
        "equations": {
            "du/dt": "Du * nabla^2 u + gamma * (a - u + u^2 * v)",
            "dv/dt": "Dv * nabla^2 v + gamma * (b - u^2 * v)"
        },
        "params_mesh": [
            {
                "row": 0,
                "description": "u field (activator) parameters",
                "slots": [
                    {"index": 0, "name": "Du", "description": "Diffusion coefficient for u (activator)", "typical_range": [0.01, 1.0]},
                    {"index": 1, "name": "gamma", "description": "Reaction rate scaling (like Damkohler number)", "typical_range": [10.0, 1000.0]},
                    {"index": 2, "name": "a", "description": "Source rate for u (feed term)", "typical_range": [0.05, 0.2]},
                    {"index": 3, "name": "b", "description": "Source rate for v (substrate supply)", "typical_range": [0.5, 2.0]},
                    {"index": 4, "name": "mu", "description": "Reserved (unused)", "typical_range": [0, 0]},
                    {"index": 5, "name": "chi", "description": "Cross-diffusion coefficient", "typical_range": [-0.1, 0.1]}
                ]
            },
            {
                "row": 1,
                "description": "v field (substrate) parameters",
                "slots": [
                    {"index": 0, "name": "Dv", "description": "Diffusion coefficient for v (substrate)", "typical_range": [1.0, 50.0]},
                    {"index": 1, "name": "unused_1", "description": "Reserved slot", "typical_range": [0, 0]},
                    {"index": 2, "name": "unused_2", "description": "Reserved slot", "typical_range": [0, 0]},
                    {"index": 3, "name": "unused_3", "description": "Reserved slot", "typical_range": [0, 0]},
                    {"index": 4, "name": "unused_4", "description": "Reserved slot", "typical_range": [0, 0]},
                    {"index": 5, "name": "unused_5", "description": "Reserved slot", "typical_range": [0, 0]}
                ]
            }
        ],
        "pattern_regimes": {
            "spots": "gamma=100-500, a=0.1, b=0.9, Du=0.05, Dv=1.0 -> hexagonal spots",
            "stripes": "gamma=500-1000, a=0.1, b=0.9, Du=0.05, Dv=1.0 -> stripe/labyrinth",
            "mixed": "Near Turing boundary -> coexistence of spots and stripes"
        }
    }

    def __init__(self, aggr_type='add', bc_dpos=None, p=None):
        super(PDE_Diffusiophoresis_Schnakenberg, self).__init__(aggr='add')

        self.bc_dpos = bc_dpos

        # u field parameters (row 0)
        self.Du = p[0, 0]       # Diffusion coefficient for u
        self.gamma = p[0, 1]    # Reaction rate scaling
        self.a = p[0, 2]        # Source rate for u
        self.b = p[0, 3]        # Source rate for v (substrate supply)

        # Cross-diffusion coefficient
        if p[0].size(0) > 5:
            self.chi = p[0, 5]
        else:
            self.chi = torch.tensor(0.0, device=p.device)

        # v field parameters (row 1)
        self.Dv = p[1, 0]       # Diffusion coefficient for v

        # Store coefficient for later use
        self.coeff = p

        # Compute steady state
        u_star = self.a + self.b
        v_star = self.b / (u_star * u_star)

        # Required for compatibility with base class expectations
        # graph_data_generator.py computes: C1_0 = model.A, C2_0 = model.B / model.A
        # For Schnakenberg: u* = a + b, v* = b / (a+b)^2
        # Set A = u*, B = v* * u* so that B/A = v*
        self.A = u_star.clone() if hasattr(u_star, 'clone') else torch.tensor(u_star, device=p.device)
        self.B = (v_star * u_star).clone() if hasattr(v_star, 'clone') else torch.tensor(v_star * u_star, device=p.device)

        # Print initialized parameters
        print(f"initialized PDE_Diffusiophoresis_Schnakenberg with parameters:")
        print(f"u: Du={self.Du.item():.4f}, v: Dv={self.Dv.item():.4f}")
        print(f"gamma={self.gamma.item():.1f}, a={self.a.item():.4f}, b={self.b.item():.4f}")
        print(f"chi={self.chi.item():.4f}")
        print(f"Steady state: u*={u_star.item():.4f}, v*={v_star.item():.4f}")
        print(f"Diffusion ratio Dv/Du={self.Dv.item()/self.Du.item():.1f}")

    def forward(self, data):
        """
        Update the concentration fields using the Schnakenberg reaction-diffusion model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Contains:
            - x: Node features [n_nodes, n_features]
              Field values at indices 6 and 7 (u and v mapped to C1 and C2)
            - edge_index: Connectivity [2, n_edges]
            - edge_attr: Edge attributes (Laplacian coefficients) [n_edges]

        Returns
        -------
        d_C : torch.Tensor
            The derivatives of u and v [n_nodes, 2]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Extract field values (u = C1, v = C2)
        u = x[:, 6:7]  # u (activator) at index 6
        v = x[:, 7:8]  # v (substrate) at index 7

        # Compute Laplacians for diffusion
        laplacian_u = self.propagate(edge_index, u=u, edge_attr=edge_attr, D_ratio=None)
        laplacian_v = self.propagate(edge_index, u=v, edge_attr=edge_attr, D_ratio=None)

        # Store Laplacians for potential use elsewhere
        self.laplacian_C1 = laplacian_u
        self.laplacian_C2 = laplacian_v

        # Diffusion terms
        diff_u = self.Du * laplacian_u
        diff_v = self.Dv * laplacian_v

        # Cross-diffusion: u diffuses in response to v Laplacian
        cross_diff_u = self.chi * laplacian_v

        # Schnakenberg reaction terms
        # du/dt = gamma * (a - u + u^2 * v)
        # dv/dt = gamma * (b - u^2 * v)
        u_sq_v = u * u * v
        R_u = self.gamma * (self.a - u + u_sq_v)
        R_v = self.gamma * (self.b - u_sq_v)

        # Light damping toward steady state for numerical stability
        u_star = self.a + self.b
        v_star = self.b / (u_star * u_star)
        damping = 0.005
        damp_u = -damping * (u - u_star)
        damp_v = -damping * (v - v_star)

        # Combine
        du = diff_u + R_u + cross_diff_u + damp_u
        dv = diff_v + R_v + damp_v

        # Combine derivatives
        d_C = torch.cat([du, dv], dim=1)

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
