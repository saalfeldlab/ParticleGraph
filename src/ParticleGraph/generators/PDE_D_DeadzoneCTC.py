import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_DeadzoneCTC(pyg.nn.MessagePassing):
    """
    CTC with receptor-adaptation deadzone + pp damping + fp drag.

    Modifies the standard CTC tanh transfer function by introducing a flat
    "deadzone" of width epsilon around the threshold concentration T. When
    |C1 - T| < epsilon*A, the CTC sign factor is exactly zero, meaning
    particles near the threshold experience NO chemotactic driving force.
    They settle purely through pp damping. Outside the deadzone, the standard
    tanh response applies with shifted argument.

    This addresses the 2-type convergence bottleneck: particles near T
    oscillate because the tanh function transitions smoothly through zero,
    producing small but sign-alternating forces as C1 fluctuates around T.
    The deadzone eliminates these oscillations entirely within the
    adaptation zone.

    Physical motivation: receptor adaptation / perfect adaptation in
    chemotaxis. Cells near their target concentration adapt their receptor
    sensitivity, becoming insensitive to small fluctuations. Only large
    deviations from the adapted state trigger a migration response.

    Transfer function:
        sign_factor = 0                                        if |C1-T| < eps*A
        sign_factor = -tanh(steep * (C1-T-eps*A) / A)         if C1 > T + eps*A
        sign_factor = -tanh(steep * (C1-T+eps*A) / A)         if C1 < T - eps*A

    Also includes all DragDampedCTC features: durotaxis, pp damping, fp drag.

    Literature:
    - Barkai, N. & Leibler, S. (1997) Nature 387:913-917
      "Robustness in simple biochemical networks"
    - Wolpert, L. (1969) J Theor Biol 25:1-47
      "Positional information and the spatial pattern of cellular differentiation"
    - Lo, C. M. et al. (2000) Biophysical Journal 79:144-152
    - Painter, K. J. & Hillen, T. (2002) Can Appl Math Q 10(4):501-543
    - Tranquillo, R. T. & Lauffenburger, D. A. (1987) J Math Biol 25:229-262

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
    """

    PARAMS_DOC = {
        "model_name": "DeadzoneCTC",
        "literature": "Barkai & Leibler (1997) Nature 387:913; Wolpert (1969); Lo (2000); Painter & Hillen (2002); Tranquillo (1987)",
        "description": "CTC with receptor-adaptation deadzone near threshold + pp damping + fp drag",
        "equations": {
            "field_to_particle": "v = M*(1+alpha*|gradC1|)*sign_factor*grad*dir / (1+fp_drag*|vel|/v_ref)",
            "deadzone_ctc": "sign_factor = 0 if |C1-T|<eps*A, else -tanh(3*(C1-T∓eps*A)/A)",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = f_AR * (1 - damping * exp(-(C1_i - T)^2 / (2*width^2)))"
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
                    {"index": 7, "name": "ctc_threshold", "description": "CTC threshold (T=ctc*A)"}
                ]
            },
            {
                "row": 1, "description": "C2 field + pp damping + deadzone params",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coeff for C2"},
                    {"index": 1, "name": "M2", "description": "Mobility for C2 gradients"},
                    {"index": 2, "name": "pp_damping", "description": "pp damping strength near T"},
                    {"index": 3, "name": "pp_damping_width", "description": "Width of pp damping zone"},
                    {"index": 4, "name": "deadzone_eps", "description": "Deadzone half-width (units of A). 0=standard CTC."},
                    {"index": 5, "name": "unused1", "description": "Unused (pad)"}
                ]
            },
            {
                "row": 2, "description": "Particle-field coupling + fp drag",
                "slots": [
                    {"index": 0, "name": "Pe", "description": "Peclet number"},
                    {"index": 1, "name": "consumption", "description": "Consumption rate of C1"},
                    {"index": 2, "name": "production", "description": "Production rate of C2"},
                    {"index": 3, "name": "influence_radius", "description": "Gaussian pf influence radius"},
                    {"index": 4, "name": "fp_drag", "description": "Velocity-dependent fp drag"},
                    {"index": 5, "name": "cross_type_factor", "description": "Per-type CTC threshold spread"}
                ]
            }
        ],
        "width_constraint": "ALL rows of params_mesh MUST have same number of columns (8). Pad shorter rows with 0.0."
    }

    def __init__(self, aggr_type='mean', p=None, particle_params=None, bc_dpos=None, dimension=2, sigma=0.005):
        super(PDE_D_DeadzoneCTC, self).__init__(aggr=aggr_type)

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

        # Per-type threshold spread
        self.cross_type_factor = p[2, 5] if p.shape[1] > 5 else 0.0

        # pp damping parameters (Painter & Hillen 2002)
        self.pp_damping = p[1, 2] if p.shape[1] > 2 else 0.0
        self.pp_damping_width = p[1, 3] if p.shape[1] > 3 else 0.5

        # Deadzone half-width (Barkai & Leibler 1997)
        self.deadzone_eps = p[1, 4] if p.shape[1] > 4 else 0.0

        # Velocity-dependent fp drag (Tranquillo & Lauffenburger 1987)
        self.fp_drag = p[2, 4] if p.shape[1] > 4 else 0.0
        self.v_ref = 0.01

        print(f"initialized PDE_D_DeadzoneCTC with parameters:")
        print(f"  mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        ga_val = self.grad_amp_alpha.item() if hasattr(self.grad_amp_alpha, 'item') else self.grad_amp_alpha
        print(f"  grad_amp_alpha={ga_val:.3f} (durotaxis, Lo 2000)")
        ctc_val = self.ctc_threshold.item() if hasattr(self.ctc_threshold, 'item') else self.ctc_threshold
        T_val = ctc_val * self.A_ref.item()
        print(f"  ctc_threshold={ctc_val:.3f} (T={T_val:.2f}, Wolpert 1969)")
        eps_val = self.deadzone_eps.item() if hasattr(self.deadzone_eps, 'item') else self.deadzone_eps
        print(f"  DEADZONE: eps={eps_val:.3f} (half-width={eps_val*self.A_ref.item():.3f}, Barkai & Leibler 1997)")
        print(f"    Particles within |C1-T| < {eps_val*self.A_ref.item():.3f} experience ZERO fp force")
        damp_val = self.pp_damping.item() if hasattr(self.pp_damping, 'item') else self.pp_damping
        damp_w = self.pp_damping_width.item() if hasattr(self.pp_damping_width, 'item') else self.pp_damping_width
        print(f"  pp_damping={damp_val:.3f}, pp_damping_width={damp_w:.3f} (Painter & Hillen 2002)")
        fp_drag_val = self.fp_drag.item() if hasattr(self.fp_drag, 'item') else self.fp_drag
        print(f"  fp_drag={fp_drag_val:.3f}, v_ref={self.v_ref:.4f} (Tranquillo 1987)")
        ctf_val = self.cross_type_factor.item() if hasattr(self.cross_type_factor, 'item') else self.cross_type_factor
        if ctf_val > 0 and particle_params is not None:
            n_types = particle_params.shape[0]
            mean_idx = (n_types - 1) / 2.0
            for t in range(n_types):
                t_offset = ctf_val * (t - mean_idx)
                t_val_t = T_val * (1.0 + t_offset)
                print(f"    Type {t}: CTC threshold = {t_val_t:.2f} (offset={t_offset:+.2f})")
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
                    f"PDE_D_DeadzoneCTC: particle_params has {n_param_rows} rows but found "
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

            # 2. CTC with deadzone (Wolpert 1969 + Barkai & Leibler 1997)
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

                deviation = C1_local - T
                eps_width = self.deadzone_eps * A_ref  # deadzone half-width in C1 units

                if eps_width > 0:
                    # Deadzone CTC: zero force within |deviation| < eps_width
                    # Shifted tanh outside the deadzone
                    # Use smooth approximation to avoid discontinuity:
                    # sign_factor = -tanh(steep * (|dev| - eps) * sign(dev) / A)
                    #             * smoothstep(|dev|, eps)
                    abs_dev = torch.abs(deviation)
                    shifted_dev = abs_dev - eps_width
                    shifted_dev_signed = shifted_dev * torch.sign(deviation)

                    # Smooth transition at deadzone boundary using sigmoid
                    # transition_width controls sharpness of deadzone edge
                    transition_width = 0.1 * A_ref  # smooth over 10% of A
                    gate = torch.sigmoid((abs_dev - eps_width) / (transition_width + 1e-8))

                    sign_factor = -torch.tanh(steepness * shifted_dev_signed / (A_ref + 1e-6)) * gate
                else:
                    # Standard CTC (no deadzone)
                    sign_factor = -torch.tanh(steepness * deviation / (A_ref + 1e-6))

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
