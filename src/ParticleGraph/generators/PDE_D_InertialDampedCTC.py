import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_InertialDampedCTC(pyg.nn.MessagePassing):
    """
    CTC with inertial (momentum-based) response smoothing for 1-type convergence.

    The 1-type CTC barrier arises because particles oscillate around the CTC
    threshold: when a particle crosses T, the CTC sign_factor instantly flips,
    pushing it back. The particle overshoots, crosses T again, flips again —
    creating a sustained oscillation that prevents convergence (plateau=0).

    MemoryDampedCTC (Block 28) tried to fix this by smoothing the INPUT signal
    (C1_mem = running average of C1). This failed because smoothing the input
    delays the response but doesn't prevent the flip-overshoot cycle.

    InertialDampedCTC takes a fundamentally different approach: smooth the OUTPUT.
    Instead of computing sign_factor from instantaneous C1 and directly using it,
    we maintain a "CTC momentum" buffer that evolves with inertia:

        ctc_momentum(t) = (1-beta) * ctc_momentum(t-1) + beta * sign_factor_instant(t)
        effective_sign_factor = ctc_momentum(t)

    The key difference: when a particle crosses T and sign_factor_instant flips
    from +1 to -1, the ctc_momentum doesn't flip instantly — it GRADUALLY
    transitions. This prevents the sharp reversal that causes overshoot.

    The inertia parameter beta controls the transition speed:
    - beta=1.0: no inertia (instant flip, same as DampedCTC)
    - beta=0.1: high inertia (slow transition, ~10 steps to fully reverse)
    - beta=0.3: moderate inertia (recommended starting point)

    This is analogous to underdamped Langevin dynamics where particles have
    momentum that resists instantaneous velocity changes.

    Physics:
    1. fp: Durotaxis + INERTIA-SMOOTHED CTC
       sign_instant = -tanh(steep*(C1-T)/A)
       ctc_momentum = (1-beta)*ctc_momentum_prev + beta*sign_instant
       v = M * (1+alpha*|gradC1|) * ctc_momentum * grad * dir
    2. pf: Standard consumption/production coupling
    3. pp: Field-damped attraction-repulsion (Painter & Hillen 2002)

    Literature:
    - Kramers, H. A. (1940) Physica 7(4):284-304
      "Brownian motion in a field of force and the diffusion model of chemical reactions"
    - Wolpert, L. (1969) J Theor Biol 25:1-47
      "Positional information and the spatial pattern of cellular differentiation"
    - Painter, K. J. & Hillen, T. (2002) Can Appl Math Q 10(4):501-543
      "Volume-filling and quorum-sensing in models for chemosensitive movement"
    - Lo, C. M. et al. (2000) Biophysical Journal 79:144-152
      "Cell movement is guided by the rigidity of the substrate"

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
    """

    PARAMS_DOC = {
        "model_name": "InertialDampedCTC",
        "literature": "Kramers (1940); Wolpert (1969); Painter & Hillen (2002); Lo (2000)",
        "description": "CTC with inertial momentum: smooth CTC sign_factor transition prevents overshoot oscillation",
        "equations": {
            "field_to_particle": "v = M*(1+alpha*|gradC1|)*ctc_momentum*grad*dir",
            "ctc_momentum": "ctc_mom(t) = (1-beta)*ctc_mom(t-1) + beta*(-tanh(3*(C1-T)/A))",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = f_AR * (1 - damping * exp(-(C1_i - T)^2 / (2*width^2)))"
        },
        "params_mesh": [
            {
                "row": 0, "description": "C1 field parameters + CTC threshold",
                "slots": [
                    {"index": 0, "name": "D1", "description": "Diffusion coeff for C1", "typical_range": [0.01, 0.5]},
                    {"index": 1, "name": "Da_c", "description": "Damkohler number", "typical_range": [1.0, 50.0]},
                    {"index": 2, "name": "A", "description": "Brusselator A (CTC reference)", "typical_range": [0.5, 5.0]},
                    {"index": 3, "name": "B", "description": "Brusselator B", "typical_range": [1.0, 10.0]},
                    {"index": 4, "name": "mu", "description": "Morphological param", "typical_range": [0.01, 0.1]},
                    {"index": 5, "name": "M1", "description": "Mobility for C1 gradients", "typical_range": [-16, 16]},
                    {"index": 6, "name": "grad_amp_alpha", "description": "Durotaxis amplification", "typical_range": [0.0, 2.0]},
                    {"index": 7, "name": "ctc_threshold", "description": "CTC threshold (T=ctc*A)", "typical_range": [0.5, 3.0]}
                ]
            },
            {
                "row": 1, "description": "C2 field + pp damping + inertia params",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coeff for C2", "typical_range": [0.1, 1.0]},
                    {"index": 1, "name": "M2", "description": "Mobility for C2 gradients", "typical_range": [-16, 16]},
                    {"index": 2, "name": "pp_damping", "description": "pp damping strength near T", "typical_range": [0.0, 1.0]},
                    {"index": 3, "name": "pp_damping_width", "description": "Width of pp damping zone (units of A)", "typical_range": [0.1, 1.0]},
                    {"index": 4, "name": "inertia_beta", "description": "CTC momentum update rate (0=infinite inertia, 1=no inertia)", "typical_range": [0.05, 0.5]},
                    {"index": 5, "name": "unused5", "description": "Pad", "typical_range": [0.0, 0.0]},
                    {"index": 6, "name": "unused6", "description": "Pad", "typical_range": [0.0, 0.0]},
                    {"index": 7, "name": "unused7", "description": "Pad", "typical_range": [0.0, 0.0]}
                ]
            },
            {
                "row": 2, "description": "Particle-field coupling params",
                "slots": [
                    {"index": 0, "name": "Pe", "description": "Peclet number", "typical_range": [0.5, 2.0]},
                    {"index": 1, "name": "consumption", "description": "Consumption rate of C1", "typical_range": [10, 200]},
                    {"index": 2, "name": "production", "description": "Production rate of C2", "typical_range": [-200, -10]},
                    {"index": 3, "name": "influence_radius", "description": "Gaussian pf influence radius", "typical_range": [0.01, 0.1]},
                    {"index": 4, "name": "unused4", "description": "Pad", "typical_range": [0.0, 0.0]},
                    {"index": 5, "name": "cross_type_factor", "description": "Per-type CTC threshold spread", "typical_range": [0.0, 0.5]},
                    {"index": 6, "name": "unused6", "description": "Pad", "typical_range": [0.0, 0.0]},
                    {"index": 7, "name": "unused7", "description": "Pad", "typical_range": [0.0, 0.0]}
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
        super(PDE_D_InertialDampedCTC, self).__init__(aggr=aggr_type)

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

        # Inertial CTC momentum parameter (Kramers 1940)
        self.inertia_beta = p[1, 4] if p.shape[1] > 4 else 1.0
        if not isinstance(self.inertia_beta, torch.Tensor):
            self.inertia_beta = torch.tensor(float(self.inertia_beta), device=p.device)

        # CTC momentum buffer — initialized on first fp pass
        self.ctc_momentum = None

        ib_val = self.inertia_beta.item() if hasattr(self.inertia_beta, 'item') else self.inertia_beta
        print(f"initialized PDE_D_InertialDampedCTC with parameters:")
        print(f"  mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        ga_val = self.grad_amp_alpha.item() if hasattr(self.grad_amp_alpha, 'item') else self.grad_amp_alpha
        print(f"  grad_amp_alpha={ga_val:.3f} (durotaxis, Lo 2000)")
        ctc_val = self.ctc_threshold.item() if hasattr(self.ctc_threshold, 'item') else self.ctc_threshold
        T_val = ctc_val * self.A_ref.item()
        print(f"  ctc_threshold={ctc_val:.3f} (T={T_val:.2f}, Wolpert 1969)")
        damp_val = self.pp_damping.item() if hasattr(self.pp_damping, 'item') else self.pp_damping
        damp_w = self.pp_damping_width.item() if hasattr(self.pp_damping_width, 'item') else self.pp_damping_width
        print(f"  pp_damping={damp_val:.3f}, pp_damping_width={damp_w:.3f} (Painter & Hillen 2002)")
        print(f"  inertia_beta={ib_val:.3f} (Kramers 1940)")
        print(f"    Effective inertia timescale: ~{1.0/max(ib_val, 0.001):.0f} steps")
        ctf_val = self.cross_type_factor.item() if hasattr(self.cross_type_factor, 'item') else self.cross_type_factor
        if ctf_val > 0 and particle_params is not None:
            n_types = particle_params.shape[0]
            mean_idx = (n_types - 1) / 2.0
            for t in range(n_types):
                t_offset = ctf_val * (t - mean_idx)
                t_val = T_val * (1.0 + t_offset)
                print(f"    Type {t}: CTC threshold = {t_val:.2f} (offset={t_offset:+.2f})")
        print(f"  Pe={self.Pe.item():.3f}, sigma={self.sigma}")
        if particle_params is not None:
            print(f"  multi-type support: {particle_params.shape[0]} particle types")

    def _update_ctc_momentum(self, sign_factor_instant, n_entities):
        """Update the CTC momentum buffer with inertial smoothing.

        sign_factor_instant: the instantaneous CTC sign factor per edge
        n_entities: total number of nodes+particles (for per-entity tracking)

        Note: We track momentum per-entity by aggregating the incoming sign factors.
        Since message() computes per-edge, we accumulate edge-level momentum.
        """
        if self.ctc_momentum is None or self.ctc_momentum.shape[0] != sign_factor_instant.shape[0]:
            self.ctc_momentum = sign_factor_instant.clone().detach()
        else:
            # Check if sizes match (they should for edge-level tracking)
            if self.ctc_momentum.shape[0] == sign_factor_instant.shape[0]:
                self.ctc_momentum = ((1.0 - self.inertia_beta) * self.ctc_momentum
                                     + self.inertia_beta * sign_factor_instant).detach()
            else:
                # Size changed (different edges) — reinitialize
                self.ctc_momentum = sign_factor_instant.clone().detach()

    def forward(self, data, direction='fp'):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        if self.particle_params is not None:
            particle_type = x[:, 1 + 2*self.dimension].long()
            max_type = particle_type.max().item()
            n_param_rows = self.particle_params.shape[0]
            if max_type >= n_param_rows:
                raise ValueError(
                    f"PDE_D_InertialDampedCTC: particle_params has {n_param_rows} rows but found "
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

            # 2. INERTIAL Concentration-threshold coupling (Kramers 1940 + Wolpert 1969)
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

                # Compute instantaneous sign factor
                sign_factor_instant = -torch.tanh(steepness * (C1_local - T) / (A_ref + 1e-6))

                # Apply inertial smoothing
                if self.inertia_beta < 1.0:
                    self._update_ctc_momentum(sign_factor_instant, x_i.shape[0])
                    velocity_raw = velocity_raw * self.ctc_momentum
                else:
                    velocity_raw = velocity_raw * sign_factor_instant

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
