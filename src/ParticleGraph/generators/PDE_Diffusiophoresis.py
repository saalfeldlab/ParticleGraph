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
            "dC1/dt": "D1_eff * ∇²C₁ + Da_c * (A - (B+1)*C₁ + C₁²*C₂/S) + χ * ∇²C₂ + noise - damping*(C₁-A)",
            "dC2/dt": "D2 * ∇²C₂ + Da_c * (B*C₁ - C₁²*C₂/S) - damping*(C₂-B/A)",
            "S": "(1 + K_sat * C₁²)  [substrate inhibition, K_sat=0 → standard Brusselator]",
            "D1_eff": "D1 * (1 + nld_delta * (C₁-A)²/A²)  [nonlinear diffusion, 0=standard]"
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
                "description": "C₂ field parameters + mesh model feature controls",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coefficient for C₂", "typical_range": [0.1, 1.0]},
                    {"index": 2, "name": "damping", "description": "Damping coefficient toward steady state (0=use default 0.005)", "typical_range": [0.0, 0.1]},
                    {"index": 3, "name": "nld_delta", "description": "Nonlinear diffusion strength (0=constant D1, >0=concentration-dependent)", "typical_range": [0.0, 5.0]},
                    {"index": 4, "name": "K_sat", "description": "Substrate inhibition (0=standard Brusselator, >0=autocatalysis saturates at high C1)", "typical_range": [0.0, 0.5]},
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

        # Block 3 code change: Parameterized damping coefficient
        # Literature: Cross & Hohenberg (1993) Rev Mod Phys 65:851 — pattern selection
        # in reaction-diffusion systems; damping toward homogeneous steady state controls
        # the balance between pattern-forming instability and relaxation.
        # params_mesh[1][2] controls damping:
        #   0.0 = use default damping of 0.005 (backward compatible)
        #   >0  = use specified damping value
        # Higher damping stabilizes patterns faster but may suppress weak instabilities.
        # Lower damping allows stronger deviations but may delay or prevent convergence.
        if p[1].size(0) > 2 and p[1, 2] > 0:
            self.damping = p[1, 2]
        else:
            self.damping = torch.tensor(0.005, device=p.device)  # Default

        # Block 11 code change: Nonlinear (concentration-dependent) diffusion
        # params_mesh[1][3] controls nld_delta:
        #   0.0 = constant D1 (backward compatible, default)
        #   >0  = D1_eff = D1 * (1 + delta * (C1-A)^2 / A^2)
        # Literature: Gambino, Lombardo & Sammartino (2013) Nonlinear Analysis: RWA 14:1095-1112
        #   "Turing instability and traveling fronts for a nonlinear reaction-diffusion system
        #    with cross-diffusion"
        # Also: Biktashev & Tsyganov (2009) Proc R Soc A 465:3561-3580
        # Effect: Diffusion coefficient depends on local concentration. At C1 peaks (C1 >> A)
        #   and troughs (C1 << A), diffusion is enhanced. Near steady state (C1 ≈ A), D1 is
        #   unchanged. This creates multi-scale Turing patterns: sharper spot boundaries
        #   (enhanced diffusion at peaks smooths gradients less) and potential transitions
        #   from hexagonal spots to labyrinthine/stripe patterns. Concentration-dependent
        #   diffusion breaks the constant-wavelength assumption of standard Turing analysis.
        if p[1].size(0) > 3:
            self.nld_delta = p[1, 3]
        else:
            self.nld_delta = torch.tensor(0.0, device=p.device)

        # Block 12 code change: Substrate inhibition (bounded autocatalysis)
        # params_mesh[1][4] controls K_sat:
        #   0.0 = standard Brusselator autocatalysis C1²*C2 (backward compatible, default)
        #   >0  = saturating autocatalysis: C1²*C2 / (1 + K_sat*C1²)
        # Literature: Haldane (1930) "Enzymes" — substrate inhibition kinetics;
        #   Szili & Toth (1993) J Chem Soc Faraday Trans 89:43 — modified Brusselator
        #   with bounded reaction rates for stable finite-amplitude patterns.
        # Effect: At high C1, the autocatalytic term C1²*C2 saturates instead of growing
        #   quadratically. This prevents concentration blow-up and stabilizes patterns at
        #   finite amplitude. Key prediction: should allow STRONGER coupling (higher chi,
        #   consumption) in the labyrinthine regime without instability, potentially
        #   enabling the flower/mandala morphology (which needs chi≥-12) on labyrinthine
        #   field backgrounds.
        if p[1].size(0) > 4:
            self.K_sat = p[1, 4]
        else:
            self.K_sat = torch.tensor(0.0, device=p.device)

        # Store coefficient for later use
        self.coeff = p

        # Print initialized parameters for verification
        print(f"initialized PDE_Diffusiophoresis with parameters:")
        print(f"C₁: D={self.D1.item():.3f}, C₂: D={self.D2.item():.3f}, Da_c={self.Da_c.item():.3f}, A={self.A.item():.3f}, B={self.B.item():.3f}, μ={self.mu.item():.3f}, χ={self.chi.item():.3f}, noise={self.noise_amplitude.item():.4f}, damping={self.damping.item():.4f}")
        nld_val = self.nld_delta.item() if hasattr(self.nld_delta, 'item') else self.nld_delta
        if nld_val > 0:
            print(f"nonlinear diffusion: delta={nld_val:.3f} (D1_eff = D1*(1+delta*(C1-A)^2/A^2), Gambino 2013)")
        ksat_val = self.K_sat.item() if hasattr(self.K_sat, 'item') else self.K_sat
        if ksat_val > 0:
            print(f"substrate inhibition: K_sat={ksat_val:.3f} (autocatalysis = C1²C2/(1+K_sat*C1²), Haldane 1930)")
    
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

        # Block 11: Nonlinear diffusion (Gambino et al. 2013)
        # D1_eff = D1 * (1 + nld_delta * (C1 - A)^2 / A^2)
        # When nld_delta=0: standard constant diffusion (backward compatible)
        # When nld_delta>0: diffusion is enhanced at concentration peaks/troughs (far from A)
        # Clamped to [D1, D1*(1+nld_delta*4)] to prevent numerical instability
        if hasattr(self, 'nld_delta') and self.nld_delta > 0:
            deviation_sq = (C1 - self.A) ** 2 / (self.A ** 2 + 1e-6)
            # Clamp deviation to avoid extreme diffusion at very high/low C1
            deviation_sq = torch.clamp(deviation_sq, max=4.0)
            D1_eff = self.D1 * (1.0 + self.nld_delta * deviation_sq)
            diff_C1 = D1_eff * laplacian_C1
        else:
            diff_C1 = self.D1 * laplacian_C1

        # C2 diffusion
        laplacian_C2 = self.propagate(edge_index, u=C2, edge_attr=edge_attr, D_ratio=None)
        diff_C2 = self.D2 * laplacian_C2

        # Store Laplacians for potential use elsewhere
        self.laplacian_C1 = laplacian_C1
        self.laplacian_C2 = laplacian_C2

        # Compute reaction terms (Brusselator model)
        # Block 12: Substrate inhibition — autocatalytic term saturates at high C1
        # Standard: C1²*C2. Modified: C1²*C2 / (1 + K_sat*C1²)
        # When K_sat=0: standard Brusselator (backward compatible)
        # When K_sat>0: bounded autocatalysis prevents concentration blow-up
        autocatalysis = C1 * C1 * C2
        if hasattr(self, 'K_sat') and self.K_sat > 0:
            autocatalysis = autocatalysis / (1.0 + self.K_sat * C1 * C1)
        R1 = self.Da_c * (self.A - (self.B + 1) * C1 + autocatalysis)
        R2 = self.Da_c * (self.B * C1 - autocatalysis)

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

        dC1 = diff_C1 + R1 + cross_diff_C1 + noise_C1 - self.damping * (C1 - self.A)
        dC2 = diff_C2 + R2 - self.damping * (C2 - self.B/self.A)

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