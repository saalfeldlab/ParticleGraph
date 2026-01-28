import torch
import torch_geometric as pyg
from ParticleGraph.utils import to_numpy

class PDE_Diffusiophoresis_FHN(pyg.nn.MessagePassing):
    """
    FitzHugh-Nagumo reaction-diffusion model for diffusiophoresis assembly simulation.

    Literature:
    - FitzHugh (1961) "Impulses and Physiological States in Theoretical Models of Nerve Membrane", Biophys J 1:445-466
    - Nagumo et al. (1962) "An Active Pulse Transmission Line Simulating Nerve Axon", Proc IRE 50:2061-2070
    - Ermakova et al. (2009) "On propagation of excitation waves in moving media: The FitzHugh-Nagumo model", PLoS ONE 4:e4454

    The FitzHugh-Nagumo model is a simplified excitable system:
    - u: Fast activator (voltage-like)
    - v: Slow inhibitor (recovery variable)

    This model can produce:
    - Traveling waves and pulses
    - Spiral waves
    - Target patterns
    - Stripe patterns (in oscillatory regime)
    - Labyrinthine patterns

    The key difference from Brusselator/Gray-Scott is that FHN is an EXCITABLE system,
    not a pure Turing system. This can produce fundamentally different dynamics.

    Equations:
        du/dt = Du * nabla^2 u + u - u^3/3 - v + I
        dv/dt = Dv * nabla^2 v + epsilon * (u + a - b*v)

    Pattern types depend on parameters:
    - a, b: Shape of nullclines (determines excitability vs oscillatory behavior)
    - epsilon: Time scale separation (smaller = sharper waves)
    - I: External current (shifts operating point)
    - Du/Dv ratio: Determines spatial pattern type

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    increment : torch.Tensor
        The first derivative of two scalar fields u and v (mapped to C1 and C2)
    """

    # PARAMS_DOC: Self-documenting parameter structure for LLM-guided exploration
    PARAMS_DOC = {
        "model_name": "FitzHugh-Nagumo",
        "description": "Two-component excitable/oscillatory reaction-diffusion system",
        "literature": "FitzHugh (1961) Biophys J 1:445-466; Nagumo et al. (1962) Proc IRE 50:2061-2070",
        "equations": {
            "du/dt": "Du * nabla^2 u + u - u^3/3 - v + I",
            "dv/dt": "Dv * nabla^2 v + epsilon * (u + a - b*v)"
        },
        "params_mesh": [
            {
                "row": 0,
                "description": "u field (activator) parameters",
                "slots": [
                    {"index": 0, "name": "Du", "description": "Diffusion coefficient for u", "typical_range": [0.01, 1.0]},
                    {"index": 1, "name": "a", "description": "Nullcline parameter a (shifts v-nullcline)", "typical_range": [0.5, 1.0]},
                    {"index": 2, "name": "b", "description": "Nullcline parameter b (slope of v-nullcline)", "typical_range": [0.5, 2.0]},
                    {"index": 3, "name": "epsilon", "description": "Time scale ratio (slow/fast)", "typical_range": [0.01, 0.5]},
                    {"index": 4, "name": "I", "description": "External stimulus current", "typical_range": [-0.5, 0.5]},
                    {"index": 5, "name": "time_scale", "description": "Overall time scaling factor", "typical_range": [1.0, 100.0]}
                ]
            },
            {
                "row": 1,
                "description": "v field (inhibitor) parameters",
                "slots": [
                    {"index": 0, "name": "Dv", "description": "Diffusion coefficient for v", "typical_range": [0.0, 0.5]},
                    {"index": 1, "name": "unused_1", "description": "Reserved slot", "typical_range": [0, 0]},
                    {"index": 2, "name": "unused_2", "description": "Reserved slot", "typical_range": [0, 0]},
                    {"index": 3, "name": "unused_3", "description": "Reserved slot", "typical_range": [0, 0]},
                    {"index": 4, "name": "unused_4", "description": "Reserved slot", "typical_range": [0, 0]},
                    {"index": 5, "name": "unused_5", "description": "Reserved slot", "typical_range": [0, 0]}
                ]
            }
        ],
        "pattern_regimes": {
            "excitable": "a=0.7, b=0.8, epsilon=0.08, I=0 -> traveling waves, spirals",
            "oscillatory": "a=0.7, b=0.8, epsilon=0.08, I=0.4 -> bulk oscillations, target patterns",
            "turing_stripes": "Du >> Dv, a=0.75, b=1.0, epsilon=0.1 -> stripe patterns",
            "bistable": "a=0.5, b=0.5, epsilon=0.1 -> front propagation"
        }
    }

    def __init__(self, aggr_type='add', bc_dpos=None, p=None):
        super(PDE_Diffusiophoresis_FHN, self).__init__(aggr='add')

        self.bc_dpos = bc_dpos

        # u field parameters (row 0)
        self.Du = p[0, 0]           # Diffusion coefficient for u
        self.a = p[0, 1]            # Nullcline parameter a
        self.b = p[0, 2]            # Nullcline parameter b
        self.epsilon = p[0, 3]      # Time scale separation

        # External current I (shifts excitability)
        if p[0].size(0) > 4:
            self.I = p[0, 4]
        else:
            self.I = torch.tensor(0.0, device=p.device)

        # Time scaling factor
        if p[0].size(0) > 5:
            self.time_scale = p[0, 5]
        else:
            self.time_scale = torch.tensor(1.0, device=p.device)

        # v field parameters (row 1)
        self.Dv = p[1, 0]           # Diffusion coefficient for v

        # Store coefficient for later use
        self.coeff = p

        # Print initialized parameters for verification
        print(f"initialized PDE_Diffusiophoresis_FHN with parameters:")
        print(f"u: Du={self.Du.item():.4f}, v: Dv={self.Dv.item():.4f}")
        print(f"a={self.a.item():.3f}, b={self.b.item():.3f}, epsilon={self.epsilon.item():.4f}, I={self.I.item():.3f}")
        print(f"time_scale={self.time_scale.item():.1f}")

        # Identify regime based on parameters
        a_val = self.a.item()
        b_val = self.b.item()
        I_val = self.I.item()

        # Check if oscillatory (based on nullcline intersection)
        # FHN is oscillatory when the fixed point is unstable
        # Roughly: oscillatory when I is large enough to cross the cubic nullcline peak
        if abs(I_val) > 0.3:
            regime = "oscillatory (target patterns, bulk oscillations)"
        elif self.Du.item() > 10 * self.Dv.item():
            regime = "Turing-like (potential stripes)"
        else:
            regime = "excitable (traveling waves, spirals)"
        print(f"Expected pattern regime: {regime}")

        # Initial condition values for compatibility with graph_data_generator
        # graph_data_generator.py:883-884 computes: C1_0 = model.A, C2_0 = model.B / model.A
        # For FHN: u (C1) starts near resting state, v (C2) near v*=a/b
        # CRITICAL: self.A MUST be non-zero to avoid C2_0 = 0/0 = NaN
        # Using self.A=1.0 and self.B=a gives C2_0 = a (near equilibrium v)
        self.A = torch.tensor(1.0, device=p.device)  # Initial u value (must be non-zero!)
        self.B = self.a.clone()  # B/A = a, so C2_0 = a (FHN v-equilibrium ≈ (u+a)/b)

    def forward(self, data):
        """
        Update the concentration fields using the FitzHugh-Nagumo reaction-diffusion model

        Parameters
        ----------
        data : torch_geometric.data.Data
            Contains:
            - x: Node features [n_nodes, n_features]
              Field values are at indices 6 and 7 (u and v mapped to C1 and C2)
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
        v = x[:, 7:8]  # v (inhibitor) at index 7

        # Compute Laplacians for diffusion
        laplacian_u = self.propagate(edge_index, u=u, edge_attr=edge_attr, D_ratio=None)
        laplacian_v = self.propagate(edge_index, u=v, edge_attr=edge_attr, D_ratio=None)

        # Store Laplacians for potential use elsewhere
        self.laplacian_C1 = laplacian_u
        self.laplacian_C2 = laplacian_v

        # Diffusion terms
        diff_u = self.Du * laplacian_u
        diff_v = self.Dv * laplacian_v

        # FitzHugh-Nagumo reaction terms
        # du/dt = u - u^3/3 - v + I
        # dv/dt = epsilon * (u + a - b*v)

        # STABILITY FIX (Block 14): Clamp u to prevent cubic runaway
        # The cubic u - u³/3 has stable fixed points in [-2, 2].
        # Outside this range, u grows unboundedly causing NaN.
        # This is a standard numerical stabilization for FHN.
        u_clamped = torch.clamp(u, -2.0, 2.0)

        # Activator dynamics: cubic nonlinearity with inhibitor coupling
        # Use clamped u for the cubic term to prevent explosion
        R_u = u_clamped - (u_clamped * u_clamped * u_clamped) / 3.0 - v + self.I

        # Inhibitor dynamics: slow recovery variable
        R_v = self.epsilon * (u_clamped + self.a - self.b * v)

        # Combine diffusion and reaction with time scaling
        du = self.time_scale * (diff_u + R_u)
        dv = self.time_scale * (diff_v + R_v)

        # Combine derivatives
        d_C = torch.cat([du, dv], dim=1)

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
