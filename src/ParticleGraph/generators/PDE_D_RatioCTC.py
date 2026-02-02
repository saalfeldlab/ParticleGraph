import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_RatioCTC(pyg.nn.MessagePassing):
    """
    CTC using C1/C2 concentration ratio instead of C1 alone, with pp damping.

    Hypothesis: Standard CTC uses only C1 for threshold coupling, wasting the
    C2 information. In Brusselator dynamics, C1 and C2 are anti-correlated at
    Turing peaks (high C1 = low C2). The ratio C1/C2 amplifies the signal-to-noise
    of positional information, potentially providing sharper threshold sensing.

    At Brusselator steady state: C1=A, C2=B/A, so ratio = A^2/B.
    At Turing peaks: C1 high, C2 low -> ratio >> A^2/B.
    At Turing troughs: C1 low, C2 high -> ratio << A^2/B.

    The CTC threshold is set on the ratio: T_ratio = ctc_threshold * A^2/B.
    Particles move toward/away from T_ratio isoline.

    pp damping uses the SAME ratio-based distance from threshold, providing
    consistent position sensing across both fp and pp channels.

    Also includes velocity-dependent fp drag for 2-type compatibility.

    Physics:
    1. fp: Durotaxis + ratio-CTC + velocity drag
       R = C1 / (C2 + eps)
       R_ref = A^2 / B  (steady-state ratio)
       T_ratio = ctc_threshold * R_ref
       v = M * (1+alpha*|gradC1|) * (-tanh(steep*(R - T_ratio)/R_ref)) * grad * dir
       v_damped = v / (1 + fp_drag * |vel_i| / v_ref)
    2. pf: Standard consumption/production
    3. pp: Ratio-damped attraction-repulsion
       f_pp = f_standard * (1 - damping * exp(-(R - T_ratio)^2 / (2*(width*R_ref)^2)))

    Literature:
    - Wolpert, L. (1969) J Theor Biol 25:1-47
      "Positional information and the spatial pattern of cellular differentiation"
    - Lo, C. M. et al. (2000) Biophysical Journal 79:144-152
    - Painter, K. J. & Hillen, T. (2002) Can Appl Math Q 10(4):501-543
    - Tranquillo, R. T. & Lauffenburger, D. A. (1987) J Math Biol 25:229-262
    - Meinhardt, H. (1982) Models of Biological Pattern Formation, Academic Press
      (ratio-based positional information)

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
    """

    PARAMS_DOC = {
        "model_name": "RatioCTC",
        "literature": "Wolpert (1969); Meinhardt (1982); Lo (2000); Painter & Hillen (2002); Tranquillo (1987)",
        "description": "CTC using C1/C2 ratio for positional sensing + ratio-based pp damping + optional fp drag",
        "equations": {
            "field_to_particle": "v = M*(1+alpha*|gradC1|)*(-tanh(3*(R-T_ratio)/R_ref))*grad*dir / (1+fp_drag*|vel|/v_ref)",
            "ratio": "R = C1/(C2+eps), R_ref = A^2/B, T_ratio = ctc_threshold * R_ref",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = f_AR * (1 - damping * exp(-(R-T_ratio)^2 / (2*(width*R_ref)^2)))"
        },
        "params_mesh": [
            {
                "row": 0, "description": "C1 field parameters + CTC threshold",
                "slots": [
                    {"index": 0, "name": "D1", "description": "Diffusion coeff for C1"},
                    {"index": 1, "name": "Da_c", "description": "Damkohler number"},
                    {"index": 2, "name": "A", "description": "Brusselator A"},
                    {"index": 3, "name": "B", "description": "Brusselator B"},
                    {"index": 4, "name": "mu", "description": "Morphological param"},
                    {"index": 5, "name": "M1", "description": "Mobility for C1 gradients"},
                    {"index": 6, "name": "grad_amp_alpha", "description": "Durotaxis amplification"},
                    {"index": 7, "name": "ctc_threshold", "description": "Ratio CTC threshold (T_ratio=ctc*A^2/B)"}
                ]
            },
            {
                "row": 1, "description": "C2 field + pp damping params",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coeff for C2"},
                    {"index": 1, "name": "M2", "description": "Mobility for C2 gradients"},
                    {"index": 2, "name": "pp_damping", "description": "pp damping strength near ratio threshold"},
                    {"index": 3, "name": "pp_damping_width", "description": "Width of ratio damping zone (units of R_ref)"}
                ]
            },
            {
                "row": 2, "description": "Particle-field coupling + fp drag",
                "slots": [
                    {"index": 0, "name": "Pe", "description": "Peclet number"},
                    {"index": 1, "name": "consumption", "description": "Consumption rate of C1"},
                    {"index": 2, "name": "production", "description": "Production rate of C2"},
                    {"index": 3, "name": "influence_radius", "description": "Gaussian pf influence radius"},
                    {"index": 4, "name": "fp_drag", "description": "Velocity-dependent fp drag (0=off)"},
                    {"index": 5, "name": "cross_type_factor", "description": "Per-type ratio threshold spread"}
                ]
            }
        ],
        "width_constraint": "ALL rows of params_mesh MUST have same number of columns (8). Pad shorter rows."
    }

    def __init__(self, aggr_type='mean', p=None, particle_params=None, bc_dpos=None, dimension=2, sigma=0.005):
        super(PDE_D_RatioCTC, self).__init__(aggr=aggr_type)

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

        # Durotaxis
        self.grad_amp_alpha = p[0, 6] if p.shape[1] > 6 else 0.0

        # Brusselator parameters for ratio reference
        self.A_ref = p[0, 2]  # A
        self.B_ref = p[0, 3]  # B
        # Steady-state ratio: R_ref = A^2/B
        self.R_ref = self.A_ref ** 2 / (self.B_ref + 1e-6)

        # CTC threshold on ratio
        self.ctc_threshold = p[0, 7] if p.shape[1] > 7 else 0.0
        # T_ratio = ctc_threshold * R_ref
        self.T_ratio = self.ctc_threshold * self.R_ref

        # Per-type threshold spread
        self.cross_type_factor = p[2, 5] if p.shape[1] > 5 else 0.0

        # pp damping parameters
        self.pp_damping = p[1, 2] if p.shape[1] > 2 else 0.0
        self.pp_damping_width = p[1, 3] if p.shape[1] > 3 else 0.5

        # Velocity-dependent fp drag
        self.fp_drag = p[2, 4] if p.shape[1] > 4 else 0.0
        self.v_ref = 0.01

        print(f"initialized PDE_D_RatioCTC with parameters:")
        print(f"  mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        ga_val = self.grad_amp_alpha.item() if hasattr(self.grad_amp_alpha, 'item') else self.grad_amp_alpha
        print(f"  grad_amp_alpha={ga_val:.3f} (durotaxis, Lo 2000)")
        print(f"  Brusselator: A={self.A_ref.item():.2f}, B={self.B_ref.item():.2f}")
        print(f"  R_ref (steady-state ratio A^2/B) = {self.R_ref.item():.4f}")
        ctc_val = self.ctc_threshold.item() if hasattr(self.ctc_threshold, 'item') else self.ctc_threshold
        print(f"  ctc_threshold={ctc_val:.3f}, T_ratio={self.T_ratio.item():.4f} (Meinhardt 1982)")
        damp_val = self.pp_damping.item() if hasattr(self.pp_damping, 'item') else self.pp_damping
        damp_w = self.pp_damping_width.item() if hasattr(self.pp_damping_width, 'item') else self.pp_damping_width
        print(f"  pp_damping={damp_val:.3f}, pp_damping_width={damp_w:.3f} (ratio-based, Painter & Hillen 2002)")
        fp_drag_val = self.fp_drag.item() if hasattr(self.fp_drag, 'item') else self.fp_drag
        print(f"  fp_drag={fp_drag_val:.3f}, v_ref={self.v_ref:.4f} (Tranquillo 1987)")
        ctf_val = self.cross_type_factor.item() if hasattr(self.cross_type_factor, 'item') else self.cross_type_factor
        if ctf_val > 0 and particle_params is not None:
            n_types = particle_params.shape[0]
            mean_idx = (n_types - 1) / 2.0
            for t in range(n_types):
                t_offset = ctf_val * (t - mean_idx)
                t_ratio = self.T_ratio.item() * (1.0 + t_offset)
                print(f"    Type {t}: ratio threshold = {t_ratio:.4f} (offset={t_offset:+.2f})")
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
                    f"PDE_D_RatioCTC: particle_params has {n_param_rows} rows but found "
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

            # 1. Durotaxis (Lo et al. 2000)
            if self.grad_amp_alpha > 0:
                grad_mag = torch.abs(grad_C1)
                grad_mag_clamped = torch.clamp(grad_mag, max=1.0)
                amp_factor = 1.0 + self.grad_amp_alpha * grad_mag_clamped
                velocity_raw = velocity_raw * amp_factor

            # 2. Ratio-CTC (Wolpert 1969 + Meinhardt 1982)
            if self.ctc_threshold > 0:
                C1_local = fields_i[:, 0:1]
                C2_local = fields_i[:, 1:2]

                # Compute local ratio R = C1 / (C2 + eps)
                R_local = C1_local / (C2_local + 1e-4)

                R_ref = self.R_ref
                base_T_ratio = self.T_ratio
                steepness = 3.0

                # Per-type thresholds
                if (parameters_i is not None and self.cross_type_factor > 0
                        and x_i.numel() > 0):
                    type_i = x_i[:, 1 + 2*self.dimension].long()
                    n_types = type_i.max().item() + 1 if type_i.numel() > 0 else 1
                    mean_idx = (n_types - 1) / 2.0
                    type_offset = self.cross_type_factor * (type_i.float() - mean_idx)
                    T = base_T_ratio * (1.0 + type_offset.unsqueeze(1))
                else:
                    T = base_T_ratio

                sign_factor = -torch.tanh(steepness * (R_local - T) / (R_ref + 1e-6))
                velocity_raw = velocity_raw * sign_factor

            # 3. Velocity-dependent fp drag (Tranquillo & Lauffenburger 1987)
            if self.fp_drag > 0:
                vel_i = x_i[:, 1+self.dimension:1+2*self.dimension]
                speed = torch.sqrt(torch.sum(vel_i**2, dim=1, keepdim=True))
                drag_factor = 1.0 / (1.0 + self.fp_drag * speed / self.v_ref)
                velocity_raw = velocity_raw * drag_factor

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

            # Ratio-based pp damping (Painter & Hillen 2002)
            if self.pp_damping > 0 and self.ctc_threshold > 0:
                C1_local = x_i[:, 6:7].squeeze(1)
                C2_local = x_i[:, 7:8].squeeze(1)

                # Local ratio
                R_local = C1_local / (C2_local + 1e-4)

                R_ref = self.R_ref
                base_T_ratio = self.T_ratio

                # Per-type threshold for damping zone
                if (parameters_i is not None and self.cross_type_factor > 0
                        and x_i.numel() > 0):
                    type_i = x_i[:, 1 + 2*self.dimension].long()
                    n_types = type_i.max().item() + 1 if type_i.numel() > 0 else 1
                    mean_idx = (n_types - 1) / 2.0
                    type_offset = self.cross_type_factor * (type_i.float() - mean_idx)
                    T_local = base_T_ratio * (1.0 + type_offset)
                else:
                    T_local = base_T_ratio

                width = self.pp_damping_width * R_ref
                deviation = (R_local - T_local)
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
