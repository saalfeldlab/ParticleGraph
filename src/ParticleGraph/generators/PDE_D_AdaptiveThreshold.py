import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_AdaptiveThreshold(pyg.nn.MessagePassing):
    """
    Particle model with adaptive concentration-threshold coupling (CTC) that
    self-calibrates to any reaction-diffusion mesh model.

    Unlike DurotaxisThreshold which uses a fixed CTC threshold (T = ctc_threshold * A),
    this model computes the threshold adaptively from the local field statistics:
        T = C1_mean + k * C1_std
    where C1_mean and C1_std are running averages updated each step. This makes
    the CTC mechanism mesh-agnostic — it works on Brusselator, Schnakenberg,
    Gray-Scott, or any other reaction-diffusion system.

    Also includes durotaxis gradient amplification (Lo et al. 2000).

    Literature:
    - Wolpert, L. (1969) J Theor Biol 25:1-47
      "Positional information and the spatial pattern of cellular differentiation"
    - Lo, C. M. et al. (2000) Biophysical Journal 79:144-152
      "Cell movement is guided by the rigidity of the substrate"
    - Meinhardt, H. (1982) Models of Biological Pattern Formation, Academic Press
      "Adaptive positional information in morphogen gradient interpretation"

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
    """

    PARAMS_DOC = {
        "model_name": "AdaptiveThreshold",
        "literature": "Wolpert (1969) J Theor Biol 25:1; Lo (2000) Biophys J 79:144; Meinhardt (1982)",
        "description": "Adaptive CTC: threshold = C1_mean + k*C1_std (self-calibrating to any mesh model)",
        "equations": {
            "field_to_particle": "v = M * (1+alpha*clamp(|gradC1|,max=1)) * (-tanh(3*(C1 - T_adaptive))) * (grad_C1+grad_C2) * dir",
            "adaptive_threshold": "T = C1_running_mean + k * C1_running_std  (k from params_mesh[0][7])",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = (p1*exp(-d^(2p2)/(2sigma^2)) - p3*exp(-d^(2p4)/(2sigma^2))) * dir"
        },
        "params_mesh": [
            {
                "row": 0, "description": "C1 field parameters + adaptive CTC",
                "slots": [
                    {"index": 0, "name": "D1", "description": "Diffusion coeff for C1 (mesh model)", "typical_range": [0.01, 0.5]},
                    {"index": 1, "name": "Da_c", "description": "Damkohler number (mesh model)", "typical_range": [1.0, 50.0]},
                    {"index": 2, "name": "A", "description": "Brusselator/mesh param A", "typical_range": [0.5, 5.0]},
                    {"index": 3, "name": "B", "description": "Brusselator/mesh param B", "typical_range": [1.0, 10.0]},
                    {"index": 4, "name": "mu", "description": "Morphological parameter", "typical_range": [0.01, 0.1]},
                    {"index": 5, "name": "M1", "description": "Mobility coefficient for C1 gradients", "typical_range": [-16, 16]},
                    {"index": 6, "name": "grad_amp_alpha", "description": "Durotaxis gradient amplification (0=off)", "typical_range": [0.0, 2.0]},
                    {"index": 7, "name": "adaptive_k", "description": "CTC threshold = C1_mean + k*C1_std (0=at mean, 0.5=above mean)", "typical_range": [0.0, 2.0]}
                ]
            },
            {
                "row": 1, "description": "C2 field parameters",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coeff for C2 (mesh model)", "typical_range": [0.1, 1.0]},
                    {"index": 1, "name": "M2", "description": "Mobility coefficient for C2 gradients", "typical_range": [-16, 16]}
                ]
            },
            {
                "row": 2, "description": "Particle-field coupling + per-type threshold spread",
                "slots": [
                    {"index": 0, "name": "Pe", "description": "Peclet number", "typical_range": [0.5, 2.0]},
                    {"index": 1, "name": "consumption", "description": "Particle consumption rate of C1", "typical_range": [10, 200]},
                    {"index": 2, "name": "production", "description": "Particle production rate of C2", "typical_range": [-200, -10]},
                    {"index": 3, "name": "influence_radius", "description": "Gaussian influence radius for pf coupling", "typical_range": [0.01, 0.1]},
                    {"index": 4, "name": "unused", "description": "Unused (pad)", "typical_range": [0.0, 0.0]},
                    {"index": 5, "name": "cross_type_factor", "description": "Per-type CTC threshold spread (0=same, 0.5=+-50%)", "typical_range": [0.0, 0.5]}
                ]
            }
        ],
        "width_constraint": "ALL rows of params_mesh MUST have same number of columns (8). Pad shorter rows.",
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
        super(PDE_D_AdaptiveThreshold, self).__init__(aggr=aggr_type)

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

        # Durotaxis gradient amplification
        self.grad_amp_alpha = p[0, 6] if p.shape[1] > 6 else 0.0

        # Adaptive CTC: k parameter for threshold = C1_mean + k * C1_std
        self.adaptive_k = p[0, 7] if p.shape[1] > 7 else 0.5

        # Per-type threshold spread
        self.cross_type_factor = p[2, 5] if p.shape[1] > 5 else 0.0

        # Running field statistics (will be updated during simulation via field data)
        self.register_buffer('C1_running_mean', torch.tensor(1.0, device=p.device))
        self.register_buffer('C1_running_std', torch.tensor(0.5, device=p.device))
        self.stats_initialized = False
        self.ema_alpha = 0.05  # Exponential moving average smoothing factor

        # Required for compatibility with base class expectations
        self.A = torch.tensor(1.0, device=p.device)
        self.B = torch.tensor(0.0, device=p.device)

        k_val = self.adaptive_k.item() if hasattr(self.adaptive_k, 'item') else self.adaptive_k
        print(f"initialized PDE_D_AdaptiveThreshold with parameters:")
        print(f"  mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        ga_val = self.grad_amp_alpha.item() if hasattr(self.grad_amp_alpha, 'item') else self.grad_amp_alpha
        print(f"  grad_amp_alpha={ga_val:.3f} (durotaxis, Lo 2000)")
        print(f"  adaptive_k={k_val:.3f} (T = C1_mean + k*C1_std, self-calibrating CTC)")
        ctf_val = self.cross_type_factor.item() if hasattr(self.cross_type_factor, 'item') else self.cross_type_factor
        print(f"  cross_type_factor={ctf_val:.3f}")
        print(f"  Pe={self.Pe.item():.3f}, sigma={self.sigma}")
        print(f"  particle->field: consumption={self.consumption_rate.item()}, production={self.production_rate.item()}, influence_radius={self.influence_radius.item():.3f}")
        if particle_params is not None:
            print(f"  multi-type support: {particle_params.shape[0]} particle types")

    def _update_field_stats(self, C1_values):
        """Update running mean and std of C1 field from particle-interpolated values."""
        if C1_values.numel() == 0:
            return
        batch_mean = C1_values.mean()
        batch_std = C1_values.std() + 1e-6

        if not self.stats_initialized:
            self.C1_running_mean = batch_mean
            self.C1_running_std = batch_std
            self.stats_initialized = True
        else:
            self.C1_running_mean = (1 - self.ema_alpha) * self.C1_running_mean + self.ema_alpha * batch_mean
            self.C1_running_std = (1 - self.ema_alpha) * self.C1_running_std + self.ema_alpha * batch_std

    def forward(self, data, direction='fp'):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        if self.particle_params is not None:
            particle_type = x[:, 1 + 2*self.dimension].long()
            max_type = particle_type.max().item()
            n_param_rows = self.particle_params.shape[0]
            if max_type >= n_param_rows:
                raise ValueError(
                    f"PDE_D_AdaptiveThreshold: particle_params has {n_param_rows} rows but found "
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
            # Update field statistics from the C1 values available to particles
            C1_local = x[:, 6]
            self._update_field_stats(C1_local)

            result = self.propagate(edge_index, x=x, mode='fp', parameters=parameters)
            pos = x[:, 1:self.dimension+1]
            in_box = ((pos >= 0) & (pos <= 1)).all(dim=1, keepdim=True)
            result = result * in_box.float()
            return result
        elif direction == 'pf':
            result = self.propagate(edge_index, x=x, mode='pf', parameters=parameters)
            return result
        else:
            result = self.propagate(edge_index, x=x, mode='pp', parameters=parameters)
            return result

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

            # 1. Durotaxis: amplify velocity at steep gradients (Lo et al. 2000)
            if self.grad_amp_alpha > 0:
                grad_mag = torch.abs(grad_C1)
                grad_mag_clamped = torch.clamp(grad_mag, max=1.0)
                amp_factor = 1.0 + self.grad_amp_alpha * grad_mag_clamped
                velocity_raw = velocity_raw * amp_factor

            # 2. Adaptive CTC: threshold from running field statistics
            C1_local = fields_i[:, 0:1]
            base_T = self.C1_running_mean + self.adaptive_k * self.C1_running_std
            steepness = 3.0

            # Per-type thresholds when multi-type + cross_type_factor > 0
            if (parameters_i is not None and self.cross_type_factor > 0
                    and x_i.numel() > 0):
                type_i = x_i[:, 1 + 2*self.dimension].long()
                n_types = type_i.max().item() + 1 if type_i.numel() > 0 else 1
                mean_idx = (n_types - 1) / 2.0
                type_offset = self.cross_type_factor * (type_i.float() - mean_idx)
                T = base_T * (1.0 + type_offset.unsqueeze(1))
            else:
                T = base_T

            # Normalize by running std instead of A_ref for mesh-agnostic scaling
            scale = self.C1_running_std + 1e-6
            sign_factor = -torch.tanh(steepness * (C1_local - T) / scale)
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
