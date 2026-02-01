import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_CILDurotaxis(pyg.nn.MessagePassing):
    """
    CIL + Durotaxis hybrid particle model for diffusiophoresis.

    Combines two complementary mechanisms WITHOUT threshold coupling (CTC):
    1. Contact Inhibition of Locomotion (CIL): density-dependent mobility
       suppression. Particles in dense regions slow down, creating self-limiting
       aggregation and convergence.
    2. Durotaxis: gradient-amplified mobility. Particles move faster at steep
       concentration gradients, enhancing morphological structure.

    The combination is designed to achieve convergence (via CIL) with richer
    morphology (via durotaxis), particularly for 1-type and 2-type systems
    where CTC-based models fail.

    Literature:
    - Mayor & Carmona-Fontaine (2010) Trends in Cell Biology 20:319-328
      "Keeping in touch with contact inhibition of locomotion"
    - Lo, C. M. et al. (2000) Biophysical Journal 79:144-152
      "Cell movement is guided by the rigidity of the substrate"
    - Cates & Tailleur (2015) Annual Review of Condensed Matter Physics 6:219-244
      "Motility-induced phase separation"

    Physics:
    v = M * f(rho) * (1 + alpha * |grad_C1|) * grad_C
    where f(rho) = 1 / (1 + (rho/rho_0)^n) is the CIL Hill function

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
    """

    PARAMS_DOC = {
        "model_name": "CILDurotaxis",
        "literature": "Mayor (2010) TCB + Lo (2000) Biophys J + Cates (2015) ARCMP",
        "description": "CIL density-dependent mobility + durotaxis gradient amplification. No CTC.",
        "equations": {
            "field_to_particle": "v = M * f(rho) * (1 + alpha * clamp(|grad_C1|, max=1.0)) * grad_C",
            "density_function": "f(rho) = 1 / (1 + (rho/rho_0)^n), Hill function",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = (p1*exp(-d^(2p2)/(2sigma^2)) - p3*exp(-d^(2p4)/(2sigma^2))) * dir"
        },
        "params_mesh": [
            {
                "row": 0, "description": "C1 field parameters",
                "slots": [
                    {"index": 0, "name": "D1", "description": "Diffusion coeff for C1"},
                    {"index": 1, "name": "Da_c", "description": "Damkohler number"},
                    {"index": 2, "name": "A", "description": "Brusselator param A"},
                    {"index": 3, "name": "B", "description": "Brusselator param B"},
                    {"index": 4, "name": "mu", "description": "Morphological parameter"},
                    {"index": 5, "name": "M1", "description": "Global mobility for C1"},
                    {"index": 6, "name": "grad_amp_alpha", "description": "Durotaxis gradient amplification factor (0=off)"},
                    {"index": 7, "name": "unused", "description": "Padding"}
                ]
            },
            {
                "row": 1, "description": "C2 field parameters",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coeff for C2"},
                    {"index": 1, "name": "M2", "description": "Global mobility for C2"}
                ]
            },
            {
                "row": 2, "description": "Particle-field coupling + CIL parameters",
                "slots": [
                    {"index": 0, "name": "Pe", "description": "Peclet number"},
                    {"index": 1, "name": "consumption", "description": "C1 consumption rate"},
                    {"index": 2, "name": "production", "description": "C2 production rate"},
                    {"index": 3, "name": "influence_radius", "description": "Gaussian influence radius"},
                    {"index": 4, "name": "rho_0", "description": "CIL critical density threshold (default 15)"},
                    {"index": 5, "name": "hill_n", "description": "CIL Hill coefficient (default 2)"}
                ]
            }
        ]
    }

    def __init__(self, aggr_type='mean', p=None, particle_params=None, bc_dpos=None, dimension=2, sigma=0.005):
        super(PDE_D_CILDurotaxis, self).__init__(aggr=aggr_type)

        self.p = p
        self.particle_params = particle_params
        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.sigma = sigma

        # Global mobility parameters
        self.M1 = p[0, 5]
        self.M2 = p[1, 1]

        # Particle effects on fields
        self.consumption_rate = p[2, 1]
        self.production_rate = p[2, 2]
        self.influence_radius = p[2, 3]

        # Peclet number
        self.Pe = p[2, 0]

        # Particle-particle repulsion parameters
        self.repulsion_strength = 50
        self.repulsion_range = 0.04

        # Durotaxis gradient amplification factor
        self.grad_amp_alpha = p[0, 6] if p.shape[1] > 6 else 0.0

        # CIL density-dependent mobility parameters
        self.rho_0 = p[2, 4] if p.shape[1] > 4 and p[2, 4] != 0 else 15.0
        self.hill_n = p[2, 5] if p.shape[1] > 5 and p[2, 5] != 0 else 2.0
        self.sensing_radius = 0.05

        # Convert to proper tensors
        if not isinstance(self.rho_0, torch.Tensor):
            self.rho_0 = torch.tensor(float(self.rho_0), device=p.device)
        if not isinstance(self.hill_n, torch.Tensor):
            self.hill_n = torch.tensor(float(self.hill_n), device=p.device)

        # Storage for local density (computed in pp pass, used in fp pass)
        self.local_density = None

        # Report configuration
        rho0_val = self.rho_0.item() if hasattr(self.rho_0, 'item') else self.rho_0
        hill_val = self.hill_n.item() if hasattr(self.hill_n, 'item') else self.hill_n
        ga_val = self.grad_amp_alpha.item() if hasattr(self.grad_amp_alpha, 'item') else self.grad_amp_alpha
        print(f"initialized PDE_D_CILDurotaxis with parameters:")
        print(f"  mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        print(f"  CIL: rho_0={rho0_val}, hill_n={hill_val}, sensing_radius={self.sensing_radius}")
        print(f"  durotaxis: grad_amp_alpha={ga_val:.3f}")
        print(f"  Pe={self.Pe.item():.3f}, sigma={self.sigma}")
        print(f"  particle->field: consumption={self.consumption_rate.item()}, production={self.production_rate.item()}, influence_radius={self.influence_radius.item():.3f}")
        if particle_params is not None:
            print(f"  multi-type support: {particle_params.shape[0]} particle types")

    def forward(self, data, direction='fp'):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # Extract per-type parameters if available
        if self.particle_params is not None:
            particle_type = x[:, 1 + 2*self.dimension].long()
            max_type = particle_type.max().item()
            n_param_rows = self.particle_params.shape[0]
            if max_type >= n_param_rows:
                raise ValueError(
                    f"PDE_D_CILDurotaxis: particle_params has {n_param_rows} rows but found "
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

            # Apply CIL density-dependent modulation to particle velocities
            if self.local_density is not None:
                n_total = x.size(0)
                n_particles = self.local_density.size(0)
                n_nodes = n_total - n_particles

                # Hill function modulation
                ratio = self.local_density / self.rho_0
                modulation = 1.0 / (1.0 + ratio ** self.hill_n)

                # Apply to particle portion only
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
        else:  # direction == 'pp'
            # Compute local density for CIL (used by fp pass)
            self._compute_local_density(x, edge_index)
            # Compute standard pp forces
            result = self.propagate(edge_index, x=x, mode='pp', parameters=parameters)
            return result

    def _compute_local_density(self, x, edge_index):
        """Count particle neighbors within sensing_radius for CIL."""
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

            velocity_raw = (M1 * grad_C1 + M2 * grad_C2) * dir_norm

            # Durotaxis: amplify at steep gradients (CIL applied in forward())
            if self.grad_amp_alpha > 0:
                grad_mag = torch.abs(grad_C1)
                grad_mag_clamped = torch.clamp(grad_mag, max=1.0)
                amp_factor = 1.0 + self.grad_amp_alpha * grad_mag_clamped
                velocity_raw = velocity_raw * amp_factor

            return velocity_raw

        elif mode == 'pf':
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
