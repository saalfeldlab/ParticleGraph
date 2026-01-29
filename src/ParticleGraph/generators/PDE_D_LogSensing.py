import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_LogSensing(pyg.nn.MessagePassing):
    """
    Logarithmic gradient sensing particle model for diffusiophoresis.

    Implements Weber-Fechner law for chemotactic response: particles sense
    relative (logarithmic) concentration gradients rather than absolute gradients.

    Literature:
    - Kalinin et al. (2009) Biophysical Journal 96:2439-2448
      "Logarithmic sensing in Escherichia coli bacterial chemotaxis"
    - Mesibov et al. (1973) Journal of General Physiology 62:203-223
      "The range of attractant concentrations for bacterial chemotaxis"
    - Berg & Purcell (1977) Biophysical Journal 20:193-219
      "Physics of chemoreception"

    Physics:
    In standard diffusiophoresis: v = M * ∇C (linear response)
    In log-sensing:              v = M * ∇C / (C + C0) ≈ M * ∇(log C)

    Key differences from linear PDE_D:
    1. Self-limiting aggregation: At high C, effective mobility → M/C (small)
    2. Enhanced sensitivity at low C: Particles in depleted regions respond strongly
    3. Concentration-dependent mobility: Creates adaptive behavior

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
      Same as base PDE_D for compatibility.
    """

    PARAMS_DOC = {
        "model_name": "LogSensing",
        "literature": "Kalinin et al. (2009) Biophysical Journal 96:2439-2448",
        "description": "Logarithmic gradient sensing (Weber-Fechner law) for chemotactic particles",
        "equations": {
            "field_to_particle": "v = M1 * ∇C1/(C1+C0) + M2 * ∇C2/(C2+C0)",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = (p1*exp(-d^(2p2)/(2σ²)) - p3*exp(-d^(2p4)/(2σ²))) * dir"
        },
        "params": [
            {"index": 0, "name": "M1", "description": "Log-mobility for C1 gradient", "typical_range": [-8, 8]},
            {"index": 1, "name": "M2", "description": "Log-mobility for C2 gradient", "typical_range": [-8, 8]},
            {"index": 2, "name": "consumption", "description": "C1 consumption rate", "typical_range": [0, 200]},
            {"index": 3, "name": "production", "description": "C2 production rate", "typical_range": [-200, 0]},
            {"index": 4, "name": "ar_p1", "description": "Attraction strength", "typical_range": [0.5, 3.0]},
            {"index": 5, "name": "ar_p2", "description": "Attraction exponent", "typical_range": [0.5, 2.0]},
            {"index": 6, "name": "ar_p3", "description": "Repulsion strength", "typical_range": [0.5, 3.0]},
            {"index": 7, "name": "ar_p4", "description": "Repulsion exponent", "typical_range": [0.5, 2.0]}
        ],
        "log_sensing_params": {
            "C0": "Regularization constant to prevent log(0). Controls transition between linear (C<<C0) and log (C>>C0) regimes. Default=0.5",
            "note": "Effective mobility = M / (C + C0). When C >> C0, sensing is logarithmic. When C << C0, sensing approaches linear (M/C0 * ∇C)."
        }
    }

    def __init__(self, aggr_type='mean', p=None, particle_params=None, bc_dpos=None, dimension=2, sigma=0.005):
        super(PDE_D_LogSensing, self).__init__(aggr=aggr_type)

        self.p = p
        self.particle_params = particle_params
        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.sigma = sigma

        # Global parameters from mesh (used as fallback when particle_params=None)
        self.M1 = p[0, 5]
        self.M2 = p[1, 1]

        # Particle effects on fields
        self.consumption_rate = p[2, 1]
        self.production_rate = p[2, 2]
        self.influence_radius = p[2, 3]

        # Peclet number
        self.Pe = p[2, 0]

        # Particle-particle repulsion parameters (same as base PDE_D)
        self.repulsion_strength = 50
        self.repulsion_range = 0.04

        # Log-sensing regularization constant C0
        # Controls the crossover between linear (C << C0) and log (C >> C0) sensing
        # For Brusselator with A=4.5: steady state C1=A=4.5, so C0=0.5 means
        # effective mobility is ~M/5 at steady state (moderate attenuation)
        self.C0 = 0.5

        # Report configuration
        print(f"initialized PDE_D_LogSensing with parameters:")
        print(f"mobility: M₁={self.M1.item()}, M₂={self.M2.item()}")
        print(f"log-sensing C0={self.C0} (crossover concentration)")
        print(f"Pe={self.Pe.item():.3f}, sigma={self.sigma}")
        print(f"particle→Field: consumption={self.consumption_rate.item()}, production={self.production_rate.item()}, influence_radius={self.influence_radius.item():.3f}")
        if particle_params is not None:
            print(f"multi-type support: {particle_params.shape[0]} particle types")
            print(f"per-type params: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]")

    def forward(self, data, direction='fp'):
        """
        Compute interactions based on direction.
        Same interface as base PDE_D for compatibility.
        """
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # Extract per-type parameters if available
        if self.particle_params is not None:
            particle_type = x[:, 1 + 2*self.dimension].long()
            parameters = self.particle_params[to_numpy(particle_type), :]
        else:
            parameters = None

        if direction == 'interpolate':
            result = self.propagate(edge_index, x=x, mode='interpolate', parameters=parameters)
            pos = x[:, 1:self.dimension+1]
            in_box = ((pos >= 0) & (pos <= 1)).all(dim=1, keepdim=True)
            result = result * in_box.float()
            return result
        elif direction == 'fp':
            result = self.propagate(edge_index, x=x, mode='fp', parameters=parameters)
            pos = x[:, 1:self.dimension+1]
            in_box = ((pos >= 0) & (pos <= 1)).all(dim=1, keepdim=True)
            result = result * in_box.float()
            return result
        elif direction == 'pf':
            result = self.propagate(edge_index, x=x, mode='pf', parameters=parameters)
            return result
        else:  # direction == 'pp'
            result = self.propagate(edge_index, x=x, mode='pp', parameters=parameters)
            return result

    def message(self, edge_index_i, edge_index_j, x_i, x_j, mode=None, parameters_i=None):
        """
        Compute messages based on mode.

        Key difference from base PDE_D: In 'fp' mode, gradients are divided by
        local concentration (+ C0) to implement logarithmic sensing.
        """
        # Get positions
        pos_i = x_i[:, 1:self.dimension+1]
        pos_j = x_j[:, 1:self.dimension+1]

        # Calculate displacement vectors with boundary conditions
        d_pos = self.bc_dpos(pos_j - pos_i)
        dist = torch.sqrt(torch.sum(d_pos**2, dim=1))
        dist_safe = torch.clamp(dist, min=1e-6)

        if mode == 'interpolate':
            # Same as base PDE_D — field interpolation is unaffected
            C1_mesh = x_j[:, 6:7]
            C2_mesh = x_j[:, 7:8]
            weight = torch.exp(-dist / 0.01).unsqueeze(1)
            return torch.cat([C1_mesh * weight, C2_mesh * weight, weight], dim=1)

        elif mode == 'fp':
            # LOG-SENSING DIFFUSIOPHORESIS
            # v = M * ∇C / (C + C0) instead of v = M * ∇C
            fields_i = x_i[:, 6:8]  # Particle fields [C1, C2]
            fields_j = x_j[:, 6:8]  # Mesh fields [C1, C2]

            dC1 = fields_j[:, 0:1] - fields_i[:, 0:1]
            dC2 = fields_j[:, 1:2] - fields_i[:, 1:2]

            # Local concentration at particle position (for log-sensing denominator)
            C1_local = fields_i[:, 0:1]
            C2_local = fields_i[:, 1:2]

            # Log-sensing denominator: C + C0 (prevents division by zero)
            # When C >> C0: sensing is logarithmic (∇C/C ≈ ∇(logC))
            # When C << C0: sensing approaches linear (∇C/C0)
            denom_C1 = torch.abs(C1_local) + self.C0
            denom_C2 = torch.abs(C2_local) + self.C0

            # Smoothing kernel (same as base PDE_D)
            kernel = torch.exp(-dist / 0.05)

            # Direction vector
            dir_norm = d_pos / dist_safe.unsqueeze(1)

            # Gradient estimation with log-sensing normalization
            domain_scale = 32.0
            grad_C1_log = (dC1 * kernel.unsqueeze(1)) / (dist_safe.unsqueeze(1) * domain_scale * denom_C1)
            grad_C2_log = (dC2 * kernel.unsqueeze(1)) / (dist_safe.unsqueeze(1) * domain_scale * denom_C2)

            # Get mobility coefficients (per-type or global)
            if parameters_i is not None:
                M1 = parameters_i[:, 0:1]
                M2 = parameters_i[:, 1:2]
            else:
                M1 = self.M1
                M2 = self.M2

            # Log-sensing diffusiophoretic velocity
            velocities = (M1 * grad_C1_log + M2 * grad_C2_log) * dir_norm

            return velocities

        elif mode == 'pf':
            # Particle → Field: same as base PDE_D
            weights = torch.exp(-dist**2 / (2 * (self.influence_radius/3)**2))

            if parameters_i is not None:
                consumption = parameters_i[:, 2]
                production = parameters_i[:, 3]
            else:
                consumption = self.consumption_rate
                production = self.production_rate

            field_updates = torch.zeros((pos_i.size(0), 2), device=pos_i.device)
            field_updates[:, 0] = -consumption * weights
            field_updates[:, 1] = production * weights

            return field_updates

        else:  # mode == 'pp'
            # Particle → Particle: same as base PDE_D
            if parameters_i is not None:
                p1 = parameters_i[:, 4]
                p2 = parameters_i[:, 5]
                p3 = parameters_i[:, 6]
                p4 = parameters_i[:, 7]

                f = (p1 * torch.exp(-dist ** (2 * p2) / (2 * self.sigma ** 2))
                     - p3 * torch.exp(-dist ** (2 * p4) / (2 * self.sigma ** 2)))

                forces = f[:, None] * d_pos / dist_safe.unsqueeze(1)
            else:
                forces = torch.zeros_like(pos_i)
                in_range = dist < self.repulsion_range
                if in_range.any():
                    dir_norm = d_pos / dist_safe.unsqueeze(1)
                    repulsion_mag = self.repulsion_strength * torch.exp(
                        -5.0 * dist[in_range] / self.repulsion_range
                    )
                    forces[in_range] = -dir_norm[in_range] * repulsion_mag.unsqueeze(1)

            return forces

    def update(self, aggr_out, mode=None):
        """
        Process aggregated messages — same as base PDE_D.
        """
        if mode == 'interpolate':
            C1_weighted = aggr_out[:, 0:1]
            C2_weighted = aggr_out[:, 1:2]
            weight_sum = aggr_out[:, 2:3]
            weight_sum = torch.clamp(weight_sum, min=1e-10)
            return torch.cat([C1_weighted / weight_sum, C2_weighted / weight_sum], dim=1)
        else:
            return aggr_out
