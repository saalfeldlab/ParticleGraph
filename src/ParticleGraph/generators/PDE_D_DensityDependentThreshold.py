import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_DensityDependentThreshold(pyg.nn.MessagePassing):
    """
    Hybrid particle model combining density-dependent mobility (contact inhibition
    of locomotion) with concentration-threshold bistable coupling.

    This model combines two mechanisms:
    1. Density-dependent mobility (CIL): Particles reduce mobility at high local
       density via a Hill function: M_eff = M * 1/(1 + (rho/rho_0)^n)
       This creates self-limiting aggregation and natural containment.
    2. ThresholdCoupling (Wolpert 1969): Bistable response reverses mobility
       sign at concentration threshold T:
       sign_factor = -tanh(steepness * (C1 - T) / A)

    The combination provides:
    - CIL containment prevents particle escape (density regulation)
    - Bistable coupling drives convergence to steady state (threshold switching)
    - Together they may enable convergence across all particle type counts

    Literature:
    - Mayor & Carmona-Fontaine (2010) Trends in Cell Biology 20:319-328
      "Keeping in touch with contact inhibition of locomotion"
    - Cates & Tailleur (2015) Annual Review of Condensed Matter Physics 6:219-244
      "Motility-induced phase separation" (density-dependent motility)
    - Wolpert, L. (1969) J Theor Biol 25:1-47
      "Positional information and the spatial pattern of cellular differentiation"

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
    """

    PARAMS_DOC = {
        "model_name": "DensityDependentThreshold",
        "literature": "Mayor (2010) TCB 20:319; Cates & Tailleur (2015) ARCMP 6:219; Wolpert (1969) JTB 25:1",
        "description": "Density-dependent mobility (CIL) + bistable concentration-threshold coupling",
        "equations": {
            "field_to_particle": "v = M * f(rho) * (-tanh(3*(C1-T)/A)) * (grad_C1 + grad_C2) * dir",
            "density_function": "f(rho) = 1 / (1 + (rho/rho_0)^n), Hill function",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = (p1*exp(-d^(2p2)/(2sigma^2)) - p3*exp(-d^(2p4)/(2sigma^2))) * dir"
        },
        "params_mesh": [
            {
                "row": 0, "description": "C1 field parameters + CTC threshold",
                "slots": [
                    {"index": 0, "name": "D1", "description": "Diffusion coeff for C1 (mesh model)"},
                    {"index": 1, "name": "Da_c", "description": "Damkohler number (mesh model)"},
                    {"index": 2, "name": "A", "description": "Brusselator param A (also CTC reference)"},
                    {"index": 3, "name": "B", "description": "Brusselator param B"},
                    {"index": 4, "name": "mu", "description": "Morphological parameter"},
                    {"index": 5, "name": "M1", "description": "Mobility coefficient for C1 gradients"},
                    {"index": 6, "name": "ctc_threshold", "description": "CTC threshold factor (T=ctc*A)"},
                    {"index": 7, "name": "rho_0", "description": "Critical density for CIL (neighbor count)"}
                ]
            },
            {
                "row": 1, "description": "C2 field parameters",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coeff for C2"},
                    {"index": 1, "name": "M2", "description": "Mobility coefficient for C2 gradients"}
                ]
            },
            {
                "row": 2, "description": "Particle-field coupling + CIL params",
                "slots": [
                    {"index": 0, "name": "Pe", "description": "Peclet number"},
                    {"index": 1, "name": "consumption", "description": "Particle consumption rate of C1"},
                    {"index": 2, "name": "production", "description": "Particle production rate of C2"},
                    {"index": 3, "name": "influence_radius", "description": "Gaussian influence radius for pf coupling"},
                    {"index": 4, "name": "hill_n", "description": "Hill coefficient for CIL (cooperativity, default=2)"},
                    {"index": 5, "name": "cross_type_factor", "description": "Per-type CTC threshold spread (0=same, 0.3=+-30%)"}
                ]
            }
        ],
        "particle_params": {
            "description": "Per-type params from simulation.params (one row per n_particle_types)",
            "slots": [
                {"index": 0, "name": "M1", "description": "Per-type mobility for C1"},
                {"index": 1, "name": "M2", "description": "Per-type mobility for C2"},
                {"index": 2, "name": "consumption", "description": "Per-type consumption rate"},
                {"index": 3, "name": "production", "description": "Per-type production rate"},
                {"index": 4, "name": "ar_p1", "description": "Attraction strength"},
                {"index": 5, "name": "ar_p2", "description": "Attraction exponent"},
                {"index": 6, "name": "ar_p3", "description": "Repulsion strength"},
                {"index": 7, "name": "ar_p4", "description": "Repulsion exponent"}
            ]
        }
    }

    def __init__(self, aggr_type='mean', p=None, particle_params=None, bc_dpos=None, dimension=2, sigma=0.005):
        super(PDE_D_DensityDependentThreshold, self).__init__(aggr=aggr_type)

        self.p = p
        self.particle_params = particle_params
        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.sigma = sigma

        self.M1 = p[0, 5]
        self.M2 = p[1, 1]
        self.consumption_rate = p[2, 1]
        self.production_rate = p[2, 2]
        self.influence_radius = p[2, 3]
        self.Pe = p[2, 0]
        self.repulsion_strength = 50
        self.repulsion_range = 0.04

        # CTC threshold parameters (same convention as DurotaxisThreshold)
        self.ctc_threshold = p[0, 6] if p.shape[1] > 6 else 0.0
        self.A_ref = p[0, 2]

        # Density-dependent mobility parameters (CIL)
        self.rho_0 = p[0, 7] if p.shape[1] > 7 else 15.0
        self.hill_n = p[2, 4] if p.shape[1] > 4 and p[2, 4] != 0 else 2.0
        self.sensing_radius = 0.05

        # Per-type threshold spread
        self.cross_type_factor = p[2, 5] if p.shape[1] > 5 else 0.0

        # Convert to tensors
        if not isinstance(self.rho_0, torch.Tensor):
            self.rho_0 = torch.tensor(float(self.rho_0), device=p.device)
        if not isinstance(self.hill_n, torch.Tensor):
            self.hill_n = torch.tensor(float(self.hill_n), device=p.device)

        # Storage for local density (computed in pp pass, used in fp pass)
        self.local_density = None

        # Report configuration
        rho0_val = self.rho_0.item() if hasattr(self.rho_0, 'item') else self.rho_0
        hill_val = self.hill_n.item() if hasattr(self.hill_n, 'item') else self.hill_n
        ctc_val = self.ctc_threshold.item() if hasattr(self.ctc_threshold, 'item') else self.ctc_threshold
        T_val = ctc_val * self.A_ref.item()
        ctf_val = self.cross_type_factor.item() if hasattr(self.cross_type_factor, 'item') else self.cross_type_factor
        print(f"initialized PDE_D_DensityDependentThreshold with parameters:")
        print(f"  mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        print(f"  CIL: rho_0={rho0_val}, hill_n={hill_val}, sensing_radius={self.sensing_radius}")
        print(f"  CTC: threshold={ctc_val:.3f} (T={T_val:.2f}, reversal at C1=T*A, Wolpert 1969)")
        if ctf_val > 0 and particle_params is not None:
            n_types = particle_params.shape[0]
            mean_idx = (n_types - 1) / 2.0
            for t in range(n_types):
                t_offset = ctf_val * (t - mean_idx)
                t_val = T_val * (1.0 + t_offset)
                print(f"    Type {t}: CTC threshold = {t_val:.2f} (offset={t_offset:+.2f})")
        print(f"  Pe={self.Pe.item():.3f}, sigma={self.sigma}")
        print(f"  particle->field: consumption={self.consumption_rate.item()}, production={self.production_rate.item()}, influence_radius={self.influence_radius.item():.3f}")
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
                    f"PDE_D_DensityDependentThreshold: particle_params has {n_param_rows} rows but found "
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

            # Apply density-dependent modulation to particle velocities
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
            # Compute local density first (used by fp pass)
            self._compute_local_density(x, edge_index)
            result = self.propagate(edge_index, x=x, mode='pp', parameters=parameters)
            return result

    def _compute_local_density(self, x, edge_index):
        """Count neighbors within sensing_radius for each particle."""
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

            # Concentration-threshold coupling (Wolpert 1969)
            # Density modulation is applied post-aggregation in forward()
            if self.ctc_threshold > 0:
                C1_local = fields_i[:, 0:1]
                A_ref = self.A_ref
                base_T = self.ctc_threshold * A_ref
                steepness = 3.0

                # Per-type thresholds
                if (parameters_i is not None and self.cross_type_factor > 0
                        and x_i.numel() > 0):
                    type_i = x_i[:, 1 + 2*self.dimension].long()
                    n_types = type_i.max().item() + 1 if type_i.numel() > 0 else 1
                    mean_idx = (n_types - 1) / 2.0
                    type_offset = self.cross_type_factor * (type_i.float() - mean_idx)
                    T = base_T * (1.0 + type_offset.unsqueeze(1))
                else:
                    T = base_T

                sign_factor = -torch.tanh(steepness * (C1_local - T) / (A_ref + 1e-6))
                velocity_raw = velocity_raw * sign_factor

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
