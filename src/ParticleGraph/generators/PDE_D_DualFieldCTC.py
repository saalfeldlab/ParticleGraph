import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_DualFieldCTC(pyg.nn.MessagePassing):
    """
    Dual-field CTC: C2 modulates the CTC response strength based on C1.

    Extends DampedCTC by adding a second morphogen channel to the CTC decision.
    Instead of sign_factor = -tanh(s*(C1-T)/A), uses:
      sign_factor = -tanh(s*(C1-T)/A) * (1 + beta * tanh(s2*(C2-T2)/A))

    When C2 is near its own threshold T2, the C1-based CTC response is amplified
    (or attenuated if beta < 0). This provides a second channel of positional
    information for particle sorting, inspired by morphogen gradient intersection
    in developmental biology.

    The key difference from all prior CTC variants: the core tanh(C1-T) mechanism
    is PRESERVED (not replaced). C2 acts as a MODULATOR, not an alternative signal.
    This avoids the anti-convergence problem of deadzone, ratio, and gradient-based
    CTC modifications.

    Physics:
    1. fp: Durotaxis gradient-amplified mobility + dual-field CTC coupling
       v = M * (1 + alpha * |gradC1|) * (-tanh(s*(C1-T)/A)) * (1+beta*tanh(s2*(C2-T2)/A)) * grad * dir
    2. pf: Standard consumption/production coupling
    3. pp: Field-damped attraction-repulsion (same as DampedCTC)

    Literature:
    - Wolpert, L. (1969) J Theor Biol 25:1-47
      "Positional information and the spatial pattern of cellular differentiation"
      (Intersecting morphogen gradients for 2D positional specification)
    - Green, J.B.A. & Sharpe, J. (2015) Development 142:1203-1211
      "Positional information and reaction-diffusion: two big ideas in developmental
      biology combine" (Multi-morphogen positional encoding)
    - Painter, K.J. & Hillen, T. (2002) CAMQ 10(4):501-543

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
    """

    PARAMS_DOC = {
        "model_name": "DualFieldCTC",
        "literature": "Wolpert (1969) J Theor Biol 25:1; Green & Sharpe (2015) Development 142:1203",
        "description": "Dual-field CTC: C2 modulates C1-based CTC response strength for 2D positional encoding",
        "equations": {
            "field_to_particle": "v = M * (1+alpha*|gradC1|) * (-tanh(s*(C1-T1)/A)) * (1+beta*tanh(s2*(C2-T2)/A)) * grad * dir",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = f_AR * (1 - damping * exp(-(C1_i - T)^2 / (2*width^2)))"
        },
        "params_mesh": [
            {
                "row": 0, "description": "C1 field parameters + CTC threshold",
                "slots": [
                    {"index": 0, "name": "D1", "description": "Diffusion coeff for C1 (mesh model)", "typical_range": [0.01, 0.5]},
                    {"index": 1, "name": "Da_c", "description": "Damkohler number (mesh model)", "typical_range": [1.0, 50.0]},
                    {"index": 2, "name": "A", "description": "Brusselator param A (mesh model, also CTC reference)", "typical_range": [0.5, 5.0]},
                    {"index": 3, "name": "B", "description": "Brusselator param B (mesh model)", "typical_range": [1.0, 10.0]},
                    {"index": 4, "name": "mu", "description": "Morphological parameter (mesh model)", "typical_range": [0.01, 0.1]},
                    {"index": 5, "name": "M1", "description": "Mobility coefficient for C1 gradients", "typical_range": [-16, 16]},
                    {"index": 6, "name": "grad_amp_alpha", "description": "Durotaxis gradient amplification (0=off)", "typical_range": [0.0, 2.0]},
                    {"index": 7, "name": "ctc_threshold", "description": "CTC threshold for C1 (T1=ctc*A)", "typical_range": [0.5, 3.0]}
                ]
            },
            {
                "row": 1, "description": "C2 field parameters + pp damping + C2 modulation",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coeff for C2 (mesh model)", "typical_range": [0.1, 1.0]},
                    {"index": 1, "name": "M2", "description": "Mobility coefficient for C2 gradients", "typical_range": [-16, 16]},
                    {"index": 2, "name": "pp_damping", "description": "pp damping strength at CTC threshold", "typical_range": [0.0, 0.95]},
                    {"index": 3, "name": "pp_damping_width", "description": "Width of Gaussian damping zone", "typical_range": [0.1, 1.0]},
                    {"index": 4, "name": "c2_beta", "description": "C2 modulation strength (0=off, >0 amplifies, <0 attenuates)", "typical_range": [-0.5, 0.5]},
                    {"index": 5, "name": "c2_threshold", "description": "C2 threshold factor (T2=c2_thresh*A for Brusselator equilibrium B/A)", "typical_range": [0.5, 3.0]},
                    {"index": 6, "name": "c2_steepness", "description": "Steepness of C2 modulation tanh", "typical_range": [1.0, 5.0]},
                    {"index": 7, "name": "unused", "description": "Pad", "typical_range": [0.0, 0.0]}
                ]
            },
            {
                "row": 2, "description": "Particle-field coupling + per-type threshold spread",
                "slots": [
                    {"index": 0, "name": "Pe", "description": "Peclet number", "typical_range": [0.5, 2.0]},
                    {"index": 1, "name": "consumption", "description": "Particle consumption rate of C1", "typical_range": [10, 200]},
                    {"index": 2, "name": "production", "description": "Particle production rate of C2", "typical_range": [-200, -10]},
                    {"index": 3, "name": "influence_radius", "description": "Gaussian influence radius for pf coupling", "typical_range": [0.01, 0.1]},
                    {"index": 4, "name": "fp_drag", "description": "Velocity-dependent drag (0=off)", "typical_range": [0.0, 0.3]},
                    {"index": 5, "name": "cross_type_factor", "description": "Per-type CTC threshold spread", "typical_range": [0.0, 0.5]},
                    {"index": 6, "name": "unused2", "description": "Pad", "typical_range": [0.0, 0.0]},
                    {"index": 7, "name": "unused3", "description": "Pad", "typical_range": [0.0, 0.0]}
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
        super(PDE_D_DualFieldCTC, self).__init__(aggr=aggr_type)

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

        # CTC threshold for C1
        self.ctc_threshold = p[0, 7] if p.shape[1] > 7 else 0.0
        self.A_ref = p[0, 2]

        # Per-type threshold spread
        self.cross_type_factor = p[2, 5] if p.shape[1] > 5 else 0.0

        # pp damping parameters (Painter & Hillen 2002)
        self.pp_damping = p[1, 2] if p.shape[1] > 2 else 0.0
        self.pp_damping_width = p[1, 3] if p.shape[1] > 3 else 0.5

        # C2 modulation parameters (Wolpert 1969; Green & Sharpe 2015)
        self.c2_beta = p[1, 4] if p.shape[1] > 4 else 0.0
        self.c2_threshold = p[1, 5] if p.shape[1] > 5 else 1.0
        self.c2_steepness = p[1, 6] if p.shape[1] > 6 else 3.0

        # fp drag
        self.fp_drag = p[2, 4] if p.shape[1] > 4 else 0.0

        print(f"initialized PDE_D_DualFieldCTC with parameters:")
        print(f"  mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        ga_val = self.grad_amp_alpha.item() if hasattr(self.grad_amp_alpha, 'item') else self.grad_amp_alpha
        print(f"  grad_amp_alpha={ga_val:.3f} (durotaxis, Lo 2000)")
        ctc_val = self.ctc_threshold.item() if hasattr(self.ctc_threshold, 'item') else self.ctc_threshold
        T_val = ctc_val * self.A_ref.item()
        print(f"  ctc_threshold={ctc_val:.3f} (T1={T_val:.2f}, Wolpert 1969)")
        c2b = self.c2_beta.item() if hasattr(self.c2_beta, 'item') else self.c2_beta
        c2t = self.c2_threshold.item() if hasattr(self.c2_threshold, 'item') else self.c2_threshold
        c2s = self.c2_steepness.item() if hasattr(self.c2_steepness, 'item') else self.c2_steepness
        T2_val = c2t * self.A_ref.item()
        print(f"  C2 modulation: beta={c2b:.3f}, T2={T2_val:.2f} (c2_thresh={c2t:.2f}), steepness={c2s:.1f} (Green & Sharpe 2015)")
        damp_val = self.pp_damping.item() if hasattr(self.pp_damping, 'item') else self.pp_damping
        damp_w = self.pp_damping_width.item() if hasattr(self.pp_damping_width, 'item') else self.pp_damping_width
        print(f"  pp_damping={damp_val:.3f}, pp_damping_width={damp_w:.3f}")
        fp_d = self.fp_drag.item() if hasattr(self.fp_drag, 'item') else self.fp_drag
        print(f"  fp_drag={fp_d:.3f}")
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
                    f"PDE_D_DualFieldCTC: particle_params has {n_param_rows} rows but found "
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

            # 2. Concentration-threshold coupling on C1 (Wolpert 1969)
            if self.ctc_threshold > 0:
                C1_local = fields_i[:, 0:1]
                C2_local = fields_i[:, 1:2]
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

                # Core CTC on C1 — PRESERVED exactly as DampedCTC
                sign_factor = -torch.tanh(steepness * (C1_local - T) / (A_ref + 1e-6))

                # 3. C2 modulation (Green & Sharpe 2015)
                # C2 provides a second channel of positional information
                # When beta > 0: C2 near T2 amplifies CTC response
                # When beta < 0: C2 near T2 attenuates CTC response
                c2_beta_val = self.c2_beta
                if hasattr(c2_beta_val, 'item'):
                    c2_beta_check = c2_beta_val.item()
                else:
                    c2_beta_check = float(c2_beta_val)

                if abs(c2_beta_check) > 1e-6:
                    T2 = self.c2_threshold * A_ref
                    c2_steep = self.c2_steepness
                    if hasattr(c2_steep, 'item'):
                        c2_steep = c2_steep
                    c2_mod = 1.0 + c2_beta_val * torch.tanh(c2_steep * (C2_local - T2) / (A_ref + 1e-6))
                    sign_factor = sign_factor * c2_mod

                velocity_raw = velocity_raw * sign_factor

            # 4. Velocity-dependent drag (Tranquillo 1987)
            fp_drag_val = self.fp_drag
            if hasattr(fp_drag_val, 'item'):
                fp_drag_check = fp_drag_val.item()
            else:
                fp_drag_check = float(fp_drag_val)

            if fp_drag_check > 0:
                vel_i = x_i[:, self.dimension+1:2*self.dimension+1]
                speed = torch.sqrt(torch.sum(vel_i**2, dim=1, keepdim=True) + 1e-10)
                drag = 1.0 / (1.0 + fp_drag_check * speed)
                velocity_raw = velocity_raw * drag

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

            # Field-dependent pp damping (Painter & Hillen 2002)
            if self.pp_damping > 0 and self.ctc_threshold > 0:
                C1_local = x_i[:, 6:7].squeeze(1)
                A_ref = self.A_ref
                base_T = self.ctc_threshold * A_ref

                if (parameters_i is not None and self.cross_type_factor > 0
                        and x_i.numel() > 0):
                    type_i = x_i[:, 1 + 2*self.dimension].long()
                    n_types = type_i.max().item() + 1 if type_i.numel() > 0 else 1
                    mean_idx = (n_types - 1) / 2.0
                    type_offset = self.cross_type_factor * (type_i.float() - mean_idx)
                    T_local = base_T * (1.0 + type_offset)
                else:
                    T_local = base_T

                width = self.pp_damping_width * A_ref
                deviation = (C1_local - T_local)
                damping_factor = 1.0 - self.pp_damping * torch.exp(-deviation**2 / (2 * width**2 + 1e-8))
                forces = forces * damping_factor.unsqueeze(1)

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
