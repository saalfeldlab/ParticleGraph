import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_SecretionFeedback(pyg.nn.MessagePassing):
    """
    Hybrid particle model combining durotaxis gradient amplification,
    concentration-threshold bistable coupling, AND secretion feedback.

    Extends DurotaxisThreshold with a biologically-motivated secretion
    feedback mechanism: particles near the CTC threshold concentration
    increase their consumption/production rates, creating a positive
    feedback loop that reinforces spatial patterning.

    This model combines three mechanisms:
    1. Durotaxis (Lo et al. 2000): gradient-amplified mobility
    2. ThresholdCoupling (Wolpert 1969): bistable CTC reversal
    3. SecretionFeedback (LEGI, Levchenko & Bhatt 2006): concentration-dependent
       secretion modulation. Particles near C1=T secrete MORE, deepening local
       gradients and reinforcing CTC convergence.

    Secretion modulation:
        consumption_eff = consumption * (1 + beta * exp(-(C1-T)^2 / (2*sigma_s^2)))
    where beta is the feedback strength and sigma_s is the sensitivity width.
    At C1=T, consumption is amplified by (1+beta). Far from T, consumption is baseline.

    Literature:
    - Lo, C. M. et al. (2000) Biophysical Journal 79:144-152
    - Wolpert, L. (1969) J Theor Biol 25:1-47
    - Levchenko, A. & Bhatt, A. (2006) PLoS Computational Biology 2:e108
      "Dynamics of a gradient sensing system with LEGI architecture"
    - Parent, C. A. & Devreotes, P. N. (1999) Science 284:765-770
      "Secretion of cAMP by Dictyostelium: autocrine signaling"

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
    """

    PARAMS_DOC = {
        "model_name": "SecretionFeedback",
        "literature": "Lo (2000) Biophys J; Wolpert (1969) JTB; Levchenko&Bhatt (2006) PLoS CB",
        "description": "Durotaxis + CTC + concentration-dependent secretion feedback (LEGI-inspired)",
        "equations": {
            "field_to_particle": "v = M*(1+alpha*clamp(|gradC1|,max=1)) * (-tanh(3*(C1-T)/A)) * (grad_C1+grad_C2) * dir",
            "particle_to_field": "dC1 = -consumption*(1+beta*exp(-(C1-T)^2/(2*sigma_s^2))) * w(r)",
            "particle_to_particle": "f = (p1*exp(-d^(2p2)/(2sigma^2)) - p3*exp(-d^(2p4)/(2sigma^2))) * dir"
        },
        "params_mesh": [
            {
                "row": 0, "description": "C1 field parameters + CTC threshold",
                "slots": [
                    {"index": 0, "name": "D1", "description": "Diffusion coeff for C1", "typical_range": [0.01, 0.5]},
                    {"index": 1, "name": "Da_c", "description": "Damkohler number", "typical_range": [1.0, 50.0]},
                    {"index": 2, "name": "A", "description": "Brusselator param A (also CTC reference)", "typical_range": [0.5, 5.0]},
                    {"index": 3, "name": "B", "description": "Brusselator param B", "typical_range": [1.0, 10.0]},
                    {"index": 4, "name": "mu", "description": "Morphological parameter", "typical_range": [0.01, 0.1]},
                    {"index": 5, "name": "M1", "description": "Mobility coefficient for C1 gradients", "typical_range": [-16, 16]},
                    {"index": 6, "name": "grad_amp_alpha", "description": "Durotaxis gradient amplification", "typical_range": [0.0, 2.0]},
                    {"index": 7, "name": "ctc_threshold", "description": "CTC threshold (T=ctc*A)", "typical_range": [0.5, 3.0]}
                ]
            },
            {
                "row": 1, "description": "C2 field parameters",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coeff for C2", "typical_range": [0.1, 1.0]},
                    {"index": 1, "name": "M2", "description": "Mobility coefficient for C2 gradients", "typical_range": [-16, 16]},
                    {"index": 2, "name": "secretion_beta", "description": "Secretion feedback strength (0=off, >0=amplify near T)", "typical_range": [0.0, 5.0]},
                    {"index": 3, "name": "secretion_sigma", "description": "Secretion sensitivity width in C1 units", "typical_range": [0.1, 2.0]}
                ]
            },
            {
                "row": 2, "description": "Particle-field coupling + per-type threshold spread",
                "slots": [
                    {"index": 0, "name": "Pe", "description": "Peclet number", "typical_range": [0.5, 2.0]},
                    {"index": 1, "name": "consumption", "description": "Base consumption rate of C1", "typical_range": [10, 200]},
                    {"index": 2, "name": "production", "description": "Base production rate of C2", "typical_range": [-200, -10]},
                    {"index": 3, "name": "influence_radius", "description": "Gaussian influence radius", "typical_range": [0.01, 0.1]},
                    {"index": 4, "name": "unused", "description": "Unused (pad)", "typical_range": [0.0, 0.0]},
                    {"index": 5, "name": "cross_type_factor", "description": "Per-type CTC threshold spread", "typical_range": [0.0, 0.5]}
                ]
            }
        ],
        "width_constraint": "ALL rows of params_mesh MUST have same number of columns (8). Pad shorter rows.",
        "particle_params": {
            "description": "Per-type params from simulation.params (one row per n_particle_types)",
            "slots": [
                {"index": 0, "name": "M1", "description": "Per-type mobility for C1"},
                {"index": 1, "name": "M2", "description": "Per-type mobility for C2"},
                {"index": 2, "name": "consumption", "description": "Per-type base consumption rate"},
                {"index": 3, "name": "production", "description": "Per-type base production rate"},
                {"index": 4, "name": "ar_p1", "description": "Attraction strength"},
                {"index": 5, "name": "ar_p2", "description": "Attraction exponent"},
                {"index": 6, "name": "ar_p3", "description": "Repulsion strength"},
                {"index": 7, "name": "ar_p4", "description": "Repulsion exponent"}
            ]
        }
    }

    def __init__(self, aggr_type='mean', p=None, particle_params=None, bc_dpos=None, dimension=2, sigma=0.005):
        super(PDE_D_SecretionFeedback, self).__init__(aggr=aggr_type)

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

        # CTC threshold
        self.ctc_threshold = p[0, 7] if p.shape[1] > 7 else 0.0
        self.A_ref = p[0, 2]

        # Secretion feedback parameters (from row 1, indices 2-3)
        self.secretion_beta = p[1, 2] if p.shape[1] > 2 else 0.0
        self.secretion_sigma = p[1, 3] if p.shape[1] > 3 else 1.0

        # Per-type threshold spread
        self.cross_type_factor = p[2, 5] if p.shape[1] > 5 else 0.0

        print(f"initialized PDE_D_SecretionFeedback with parameters:")
        print(f"  mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        ga_val = self.grad_amp_alpha.item() if hasattr(self.grad_amp_alpha, 'item') else self.grad_amp_alpha
        print(f"  grad_amp_alpha={ga_val:.3f} (durotaxis, Lo 2000)")
        ctc_val = self.ctc_threshold.item() if hasattr(self.ctc_threshold, 'item') else self.ctc_threshold
        T_val = ctc_val * self.A_ref.item()
        print(f"  ctc_threshold={ctc_val:.3f} (T={T_val:.2f}, Wolpert 1969)")
        beta_val = self.secretion_beta.item() if hasattr(self.secretion_beta, 'item') else self.secretion_beta
        sigma_s_val = self.secretion_sigma.item() if hasattr(self.secretion_sigma, 'item') else self.secretion_sigma
        print(f"  secretion_beta={beta_val:.3f}, secretion_sigma={sigma_s_val:.3f} (LEGI feedback, Levchenko&Bhatt 2006)")
        print(f"  At C1=T: consumption amplified by {1+beta_val:.2f}x")
        ctf_val = self.cross_type_factor.item() if hasattr(self.cross_type_factor, 'item') else self.cross_type_factor
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
                    f"PDE_D_SecretionFeedback: particle_params has {n_param_rows} rows but found "
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

            # 2. Concentration-threshold coupling (Wolpert 1969)
            if self.ctc_threshold > 0:
                C1_local = fields_i[:, 0:1]
                A_ref = self.A_ref
                base_T = self.ctc_threshold * A_ref
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

            # Secretion feedback: modulate consumption/production near CTC threshold
            # (Levchenko & Bhatt 2006, LEGI; Parent & Devreotes 1999)
            if self.secretion_beta > 0 and self.ctc_threshold > 0:
                C1_local = x_i[:, 6]
                base_T = self.ctc_threshold * self.A_ref
                sigma_s = self.secretion_sigma
                # Gaussian amplification centered at T
                feedback = 1.0 + self.secretion_beta * torch.exp(
                    -(C1_local - base_T)**2 / (2.0 * sigma_s**2)
                )
                consumption = consumption * feedback
                production = production * feedback

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
