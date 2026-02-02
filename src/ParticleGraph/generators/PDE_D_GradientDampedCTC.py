import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_GradientDampedCTC(pyg.nn.MessagePassing):
    """
    CTC with gradient-magnitude-based pp damping (instead of concentration-based).

    Hypothesis: The DampedCTC plateau at 0.9550 (3-type) and 0.6891 (2-type) is
    limited by the concentration-based pp damping mechanism, which dampens pp forces
    when C1 ≈ T. This variant instead dampens pp forces where the GRADIENT |∇C1|
    is large — i.e., at Turing pattern boundaries. The idea is that at boundaries
    (high gradient), particles are already being strongly guided by fp forces and
    don't need pp forces to organize; while in flat-concentration regions, pp forces
    remain strong to maintain spatial order.

    Physics:
    1. fp: Durotaxis + CTC (same as DampedCTC)
       v = M * (1+alpha*|gradC1|) * (-tanh(3*(C1-T)/A)) * grad * dir
    2. pf: Standard consumption/production
    3. pp: Gradient-damped attraction-repulsion
       f_pp = f_standard * (1 - damping * sigmoid(grad_steepness * (|gradC1| - grad_threshold)))
       When |gradC1| > grad_threshold (at boundaries), pp forces are reduced.

    This is fundamentally different from DampedCTC where damping depends on
    concentration proximity to threshold (C1 ≈ T).

    For 2-type: also includes velocity-dependent fp drag (same as DragDampedCTC).

    Literature:
    - Painter, K. J. & Hillen, T. (2002) Can Appl Math Q 10(4):501-543
      "Volume-filling and quorum-sensing in models for chemosensitive movement"
    - Wolpert, L. (1969) J Theor Biol 25:1-47
      "Positional information and the spatial pattern of cellular differentiation"
    - Lo, C. M. et al. (2000) Biophysical Journal 79:144-152
      "Cell movement is guided by the rigidity of the substrate"
    - Tranquillo, R. T. & Lauffenburger, D. A. (1987) J Math Biol 25:229-262
      "Stochastic model of leukocyte chemosensory movement"

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
    """

    PARAMS_DOC = {
        "model_name": "GradientDampedCTC",
        "literature": "Painter & Hillen (2002); Wolpert (1969); Lo (2000); Tranquillo (1987)",
        "description": "CTC + gradient-magnitude pp damping + optional fp drag",
        "equations": {
            "field_to_particle": "v = M*(1+alpha*|gradC1|)*(-tanh(3*(C1-T)/A))*grad*dir / (1 + fp_drag*|vel|/v_ref)",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = f_AR * (1 - damping * sigmoid(steep*(|gradC1_i| - grad_thresh)))"
        },
        "params_mesh": [
            {
                "row": 0, "description": "C1 field parameters + CTC threshold",
                "slots": [
                    {"index": 0, "name": "D1", "description": "Diffusion coeff for C1"},
                    {"index": 1, "name": "Da_c", "description": "Damkohler number"},
                    {"index": 2, "name": "A", "description": "Brusselator A (CTC reference)"},
                    {"index": 3, "name": "B", "description": "Brusselator B"},
                    {"index": 4, "name": "mu", "description": "Morphological param"},
                    {"index": 5, "name": "M1", "description": "Mobility for C1 gradients"},
                    {"index": 6, "name": "grad_amp_alpha", "description": "Durotaxis amplification"},
                    {"index": 7, "name": "ctc_threshold", "description": "CTC threshold (T=ctc*A)"}
                ]
            },
            {
                "row": 1, "description": "C2 field + gradient pp damping params",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coeff for C2"},
                    {"index": 1, "name": "M2", "description": "Mobility for C2 gradients"},
                    {"index": 2, "name": "pp_damping", "description": "pp damping strength at high gradients (0=off)"},
                    {"index": 3, "name": "grad_threshold", "description": "Gradient threshold for pp damping onset (units of field std)"},
                    {"index": 4, "name": "grad_steepness", "description": "Sigmoid steepness for gradient damping transition"},
                    {"index": 5, "name": "unused1"},
                    {"index": 6, "name": "unused2"},
                    {"index": 7, "name": "unused3"}
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
                    {"index": 5, "name": "cross_type_factor", "description": "Per-type CTC threshold spread"}
                ]
            }
        ],
        "width_constraint": "ALL rows of params_mesh MUST have same number of columns (8). Pad shorter rows."
    }

    def __init__(self, aggr_type='mean', p=None, particle_params=None, bc_dpos=None, dimension=2, sigma=0.005):
        super(PDE_D_GradientDampedCTC, self).__init__(aggr=aggr_type)

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

        # Gradient-based pp damping parameters
        self.pp_damping = p[1, 2] if p.shape[1] > 2 else 0.0
        self.grad_threshold = p[1, 3] if p.shape[1] > 3 else 0.3
        self.grad_steepness = p[1, 4] if p.shape[1] > 4 else 10.0

        # Velocity-dependent fp drag
        self.fp_drag = p[2, 4] if p.shape[1] > 4 else 0.0
        self.v_ref = 0.01

        # Storage for per-particle gradient magnitudes (computed in fp, used in pp)
        self.local_grad_mag = None

        print(f"initialized PDE_D_GradientDampedCTC with parameters:")
        print(f"  mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        ga_val = self.grad_amp_alpha.item() if hasattr(self.grad_amp_alpha, 'item') else self.grad_amp_alpha
        print(f"  grad_amp_alpha={ga_val:.3f} (durotaxis, Lo 2000)")
        ctc_val = self.ctc_threshold.item() if hasattr(self.ctc_threshold, 'item') else self.ctc_threshold
        T_val = ctc_val * self.A_ref.item()
        print(f"  ctc_threshold={ctc_val:.3f} (T={T_val:.2f}, Wolpert 1969)")
        damp_val = self.pp_damping.item() if hasattr(self.pp_damping, 'item') else self.pp_damping
        gt_val = self.grad_threshold.item() if hasattr(self.grad_threshold, 'item') else self.grad_threshold
        gs_val = self.grad_steepness.item() if hasattr(self.grad_steepness, 'item') else self.grad_steepness
        print(f"  pp_damping={damp_val:.3f}, grad_threshold={gt_val:.3f}, grad_steepness={gs_val:.3f} (gradient-based)")
        fp_drag_val = self.fp_drag.item() if hasattr(self.fp_drag, 'item') else self.fp_drag
        print(f"  fp_drag={fp_drag_val:.3f}, v_ref={self.v_ref:.4f}")
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
                    f"PDE_D_GradientDampedCTC: particle_params has {n_param_rows} rows but found "
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

            # 2. CTC (Wolpert 1969)
            if self.ctc_threshold > 0:
                C1_local = fields_i[:, 0:1]
                A_ref = self.A_ref
                base_T = self.ctc_threshold * A_ref
                steepness = 3.0

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

            # 3. Velocity-dependent fp drag (Tranquillo & Lauffenburger 1987)
            if self.fp_drag > 0:
                vel_i = x_i[:, 1+self.dimension:1+2*self.dimension]
                speed = torch.sqrt(torch.sum(vel_i**2, dim=1, keepdim=True))
                drag_factor = 1.0 / (1.0 + self.fp_drag * speed / self.v_ref)
                velocity_raw = velocity_raw * drag_factor

            # Store gradient magnitude for use in pp damping
            # Use the magnitude of C1 gradient contribution as proxy
            grad_mag_for_damping = torch.sqrt(torch.sum(grad_C1**2, dim=1))
            # This will be aggregated by PyG; we use it per-message in pp mode instead
            # Store in a way accessible to pp mode via x_i fields
            # Actually, we can't easily pass between modes in message passing.
            # Instead, compute gradient magnitude directly from field differences in pp mode.

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

            # Gradient-magnitude pp damping
            # Compute local gradient magnitude from field differences between pp neighbors
            if self.pp_damping > 0:
                # Use field values stored in x_i columns 6:8
                C1_i = x_i[:, 6:7].squeeze(1)
                C1_j = x_j[:, 6:7].squeeze(1)

                # Approximate gradient magnitude from pairwise field difference
                dC1_pp = torch.abs(C1_j - C1_i)
                # Normalize by distance to get gradient estimate
                grad_est = dC1_pp / (dist_safe + 1e-6)

                # Sigmoid damping: damp pp forces where gradient is large
                # damping_factor = 1 - damping * sigmoid(steepness * (grad - threshold))
                damping_factor = 1.0 - self.pp_damping * torch.sigmoid(
                    self.grad_steepness * (grad_est - self.grad_threshold)
                )
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
