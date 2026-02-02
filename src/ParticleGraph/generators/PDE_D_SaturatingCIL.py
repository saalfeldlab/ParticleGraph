import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_SaturatingCIL(pyg.nn.MessagePassing):
    """
    Contact Inhibition of Locomotion with Michaelis-Menten saturating
    particle-field coupling.

    Extends DensityDependent (CIL) by replacing linear pf consumption
    with Michaelis-Menten kinetics:
        consumption_eff = consumption * C1 / (K_m + C1)
    where K_m is the half-saturation constant. At high C1 (cluster centers),
    consumption saturates, preventing local field over-depletion. This creates
    a stabilizing negative feedback: dense regions consume less per particle
    relative to their concentration, allowing the Turing field to partially
    recover underneath clusters.

    Physical motivation: Enzyme kinetics — biological consumption of a
    chemical signal saturates when substrate is abundant, following
    Michaelis-Menten kinetics. This is universal in biochemistry.

    Literature:
    - Michaelis, L. & Menten, M. L. (1913) Biochem Z 49:333-369
      "Die Kinetik der Invertinwirkung" (enzyme kinetics)
    - Mayor, R. & Carmona-Fontaine, C. (2010) Trends Cell Biol 20:319-328
      "Keeping in touch with contact inhibition of locomotion"
    - Cates, M. E. & Tailleur, J. (2015) ARCMP 6:219-244
      "Motility-induced phase separation"

    Physics:
    1. fp: Density-dependent linear diffusiophoresis (CIL)
       v = M * f(rho) * nabla_C, f(rho) = 1/(1+(rho/rho_0)^n)
    2. pf: Michaelis-Menten saturating consumption/production
       dC1 = -consumption * C1/(K_m + C1) * w(r)
       dC2 = production * w(r)   (production remains linear)
    3. pp: Standard attraction-repulsion

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
    """

    PARAMS_DOC = {
        "model_name": "SaturatingCIL",
        "literature": "Michaelis & Menten (1913) Biochem Z 49:333; Mayor & Carmona-Fontaine (2010) TCB 20:319; Cates & Tailleur (2015) ARCMP 6:219",
        "description": "CIL with Michaelis-Menten saturating consumption — prevents over-depletion at cluster centers",
        "equations": {
            "field_to_particle": "v = M * f(rho) * nabla_C, f(rho) = 1/(1+(rho/rho_0)^n)",
            "density_function": "f(rho) = 1 / (1 + (rho/rho_0)^n), Hill function",
            "particle_to_field": "dC1 = -consumption * C1/(K_m+C1) * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = (p1*exp(-d^(2p2)/(2sigma^2)) - p3*exp(-d^(2p4)/(2sigma^2))) * dir"
        },
        "params_mesh": [
            {
                "row": 0, "description": "C1 field parameters",
                "slots": [
                    {"index": 0, "name": "D1", "description": "Diffusion coeff for C1 (mesh model)", "typical_range": [0.01, 0.5]},
                    {"index": 1, "name": "Da_c", "description": "Damkohler number (mesh model)", "typical_range": [1.0, 50.0]},
                    {"index": 2, "name": "A", "description": "Brusselator param A", "typical_range": [0.5, 5.0]},
                    {"index": 3, "name": "B", "description": "Brusselator param B", "typical_range": [1.0, 10.0]},
                    {"index": 4, "name": "mu", "description": "Morphological parameter", "typical_range": [0.01, 0.1]},
                    {"index": 5, "name": "M1", "description": "Mobility coefficient for C1 gradients", "typical_range": [-16, 16]},
                    {"index": 6, "name": "unused_0", "description": "Unused (pad)", "typical_range": [0.0, 0.0]},
                    {"index": 7, "name": "K_m", "description": "Michaelis-Menten half-saturation constant (units of C1)", "typical_range": [1.0, 10.0]}
                ]
            },
            {
                "row": 1, "description": "C2 field parameters",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coeff for C2 (mesh model)", "typical_range": [0.1, 1.0]},
                    {"index": 1, "name": "M2", "description": "Mobility for C2 gradients", "typical_range": [-16, 16]},
                    {"index": 2, "name": "unused_1", "description": "Unused (pad)", "typical_range": [0.0, 0.0]},
                    {"index": 3, "name": "unused_2", "description": "Unused (pad)", "typical_range": [0.0, 0.0]}
                ]
            },
            {
                "row": 2, "description": "Particle-field coupling + CIL parameters",
                "slots": [
                    {"index": 0, "name": "Pe", "description": "Peclet number", "typical_range": [0.5, 2.0]},
                    {"index": 1, "name": "consumption", "description": "Max consumption rate of C1 (V_max in MM)", "typical_range": [10, 200]},
                    {"index": 2, "name": "production", "description": "Production rate of C2 (linear)", "typical_range": [-200, -10]},
                    {"index": 3, "name": "influence_radius", "description": "Gaussian pf influence radius", "typical_range": [0.01, 0.1]},
                    {"index": 4, "name": "rho_0", "description": "CIL critical density threshold", "typical_range": [10, 60]},
                    {"index": 5, "name": "hill_n", "description": "CIL Hill coefficient", "typical_range": [1.0, 4.0]}
                ]
            }
        ],
        "width_constraint": "ALL rows of params_mesh MUST have same number of columns (8). Pad shorter rows.",
        "particle_params": {
            "description": "Per-type params from simulation.params (one row per n_particle_types)",
            "slots": [
                {"index": 0, "name": "M1", "description": "Per-type mobility for C1"},
                {"index": 1, "name": "M2", "description": "Per-type mobility for C2"},
                {"index": 2, "name": "consumption", "description": "Per-type V_max consumption rate"},
                {"index": 3, "name": "production", "description": "Per-type production rate"},
                {"index": 4, "name": "ar_p1", "description": "Attraction strength"},
                {"index": 5, "name": "ar_p2", "description": "Attraction exponent"},
                {"index": 6, "name": "ar_p3", "description": "Repulsion strength"},
                {"index": 7, "name": "ar_p4", "description": "Repulsion exponent"}
            ]
        }
    }

    def __init__(self, aggr_type='mean', p=None, particle_params=None, bc_dpos=None, dimension=2, sigma=0.005):
        super(PDE_D_SaturatingCIL, self).__init__(aggr=aggr_type)

        self.p = p
        self.particle_params = particle_params
        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.sigma = sigma

        # Global parameters from mesh
        self.M1 = p[0, 5]
        self.M2 = p[1, 1]
        self.consumption_rate = p[2, 1]
        self.production_rate = p[2, 2]
        self.influence_radius = p[2, 3]
        self.Pe = p[2, 0]

        # pp repulsion
        self.repulsion_strength = 50
        self.repulsion_range = 0.04

        # CIL density-dependent parameters
        self.rho_0 = p[2, 4] if p.shape[1] > 4 and p[2, 4] != 0 else 15.0
        self.hill_n = p[2, 5] if p.shape[1] > 5 and p[2, 5] != 0 else 2.0
        self.sensing_radius = 0.05

        # Michaelis-Menten half-saturation constant
        # K_m in units of C1 concentration. When C1 = K_m, consumption = V_max/2.
        # Brusselator equilibrium C1 = A = 4.5, so K_m ~ A is a natural scale.
        self.K_m = p[0, 7] if p.shape[1] > 7 and p[0, 7] != 0 else 4.5

        # Convert to tensors
        if not isinstance(self.rho_0, torch.Tensor):
            self.rho_0 = torch.tensor(float(self.rho_0), device=p.device)
        if not isinstance(self.hill_n, torch.Tensor):
            self.hill_n = torch.tensor(float(self.hill_n), device=p.device)
        if not isinstance(self.K_m, torch.Tensor):
            self.K_m = torch.tensor(float(self.K_m), device=p.device)

        # Storage for local density
        self.local_density = None

        # Report configuration
        rho0_val = self.rho_0.item() if hasattr(self.rho_0, 'item') else self.rho_0
        hill_val = self.hill_n.item() if hasattr(self.hill_n, 'item') else self.hill_n
        km_val = self.K_m.item() if hasattr(self.K_m, 'item') else self.K_m
        print(f"initialized PDE_D_SaturatingCIL with parameters:")
        print(f"  mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        print(f"  CIL density-dependent: rho_0={rho0_val}, hill_n={hill_val}, sensing_radius={self.sensing_radius}")
        print(f"  Michaelis-Menten: K_m={km_val} (half-saturation, Michaelis & Menten 1913)")
        print(f"  Pe={self.Pe.item():.3f}, sigma={self.sigma}")
        print(f"  particle->field: consumption(V_max)={self.consumption_rate.item()}, production={self.production_rate.item()}, influence_radius={self.influence_radius.item():.3f}")
        if particle_params is not None:
            print(f"  multi-type support: {particle_params.shape[0]} particle types")

    def forward(self, data, direction='fp'):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        if self.particle_params is not None:
            particle_type = x[:, 1 + 2*self.dimension].long()
            max_type = particle_type.max().item()
            n_param_rows = self.particle_params.shape[0]
            if max_type >= n_param_rows:
                raise ValueError(
                    f"PDE_D_SaturatingCIL: particle_params has {n_param_rows} rows but found "
                    f"particle type {max_type}. Need {max_type + 1} rows in simulation.params."
                )
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

            # Apply CIL density-dependent modulation
            if self.local_density is not None:
                n_total = x.size(0)
                n_particles = self.local_density.size(0)
                n_nodes = n_total - n_particles

                ratio = self.local_density / self.rho_0
                modulation = 1.0 / (1.0 + ratio ** self.hill_n)

                mod_full = torch.ones(n_total, 1, device=x.device)
                mod_full[n_nodes:, 0] = modulation
                result = result * mod_full

            pos = x[:, 1:self.dimension+1]
            in_box = ((pos >= 0) & (pos <= 1)).all(dim=1, keepdim=True)
            result = result * in_box.float()
            return result
        elif direction == 'pf':
            result = self.propagate(edge_index, x=x, mode='pf', parameters=parameters)
            return result
        else:  # pp
            self._compute_local_density(x, edge_index)
            result = self.propagate(edge_index, x=x, mode='pp', parameters=parameters)
            return result

    def _compute_local_density(self, x, edge_index):
        """Count neighbors within sensing_radius for CIL modulation."""
        n_particles = x.size(0)
        target_nodes = edge_index[1]

        pos_i = x[edge_index[1], 1:self.dimension+1]
        pos_j = x[edge_index[0], 1:self.dimension+1]
        d_pos = self.bc_dpos(pos_j - pos_i)
        dist = torch.sqrt(torch.sum(d_pos**2, dim=1))

        within_radius = dist < self.sensing_radius
        counts = torch.zeros(n_particles, device=x.device)
        counts.scatter_add_(0, target_nodes[within_radius],
                           torch.ones(within_radius.sum(), device=x.device))

        self.local_density = counts

    def message(self, edge_index_i, edge_index_j, x_i, x_j, mode=None, parameters_i=None):
        pos_i = x_i[:, 1:self.dimension+1]
        pos_j = x_j[:, 1:self.dimension+1]

        d_pos = self.bc_dpos(pos_j - pos_i)
        dist = torch.sqrt(torch.sum(d_pos**2, dim=1))
        dist_safe = torch.clamp(dist, min=1e-6)

        if mode == 'interpolate':
            C1_mesh = x_j[:, 6:7]
            C2_mesh = x_j[:, 7:8]
            weight = torch.exp(-dist / 0.01).unsqueeze(1)
            return torch.cat([C1_mesh * weight, C2_mesh * weight, weight], dim=1)

        elif mode == 'fp':
            fields_i = x_i[:, 6:8]
            fields_j = x_j[:, 6:8]

            dC1 = fields_j[:, 0:1] - fields_i[:, 0:1]
            dC2 = fields_j[:, 1:2] - fields_i[:, 1:2]

            kernel = torch.exp(-dist / 0.05)
            dir_norm = d_pos / dist_safe.unsqueeze(1)
            domain_scale = 32.0
            grad_C1 = (dC1 * kernel.unsqueeze(1)) / (dist_safe.unsqueeze(1) * domain_scale)
            grad_C2 = (dC2 * kernel.unsqueeze(1)) / (dist_safe.unsqueeze(1) * domain_scale)

            if parameters_i is not None:
                M1 = parameters_i[:, 0:1]
                M2 = parameters_i[:, 1:2]
            else:
                M1 = self.M1
                M2 = self.M2

            velocities = (M1 * grad_C1 + M2 * grad_C2) * dir_norm
            return velocities

        elif mode == 'pf':
            # Michaelis-Menten saturating consumption (Michaelis & Menten 1913)
            # consumption_eff = V_max * C1 / (K_m + C1)
            # At low C1: consumption ~ V_max * C1/K_m (linear)
            # At high C1: consumption ~ V_max (saturated)
            weights = torch.exp(-dist**2 / (2 * (self.influence_radius/3)**2))

            if parameters_i is not None:
                consumption = parameters_i[:, 2]
                production = parameters_i[:, 3]
            else:
                consumption = self.consumption_rate
                production = self.production_rate

            # Get local C1 concentration at particle position
            C1_local = x_i[:, 6]  # C1 field value at particle
            C1_safe = torch.clamp(C1_local, min=0.0)  # Ensure non-negative

            # Michaelis-Menten saturation factor: C1 / (K_m + C1)
            mm_factor = C1_safe / (self.K_m + C1_safe)

            field_updates = torch.zeros((pos_i.size(0), 2), device=pos_i.device)
            field_updates[:, 0] = -consumption * mm_factor * weights  # Saturating consumption
            field_updates[:, 1] = production * weights  # Linear production (unchanged)
            return field_updates

        else:  # pp
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
        if mode == 'interpolate':
            C1_weighted = aggr_out[:, 0:1]
            C2_weighted = aggr_out[:, 1:2]
            weight_sum = aggr_out[:, 2:3]
            weight_sum = torch.clamp(weight_sum, min=1e-10)
            return torch.cat([C1_weighted / weight_sum, C2_weighted / weight_sum], dim=1)
        else:
            return aggr_out
