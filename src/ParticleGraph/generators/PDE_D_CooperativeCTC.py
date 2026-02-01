import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_CooperativeCTC(pyg.nn.MessagePassing):
    """
    Cooperative concentration-threshold coupling with tunable steepness and
    quorum-sensing density enhancement.

    This model extends the DurotaxisThreshold framework with two key additions:

    1. **Tunable CTC steepness**: The CTC tanh switch steepness (hardcoded at 3.0
       in DurotaxisThreshold) is exposed as a configurable parameter. This controls
       how sharp the bistable transition is around the threshold concentration.
       Higher steepness -> sharper switching -> stronger convergence drive.

    2. **Quorum-sensing cooperative enhancement**: The effective CTC steepness
       increases with local particle density, implementing a positive feedback
       loop analogous to bacterial quorum sensing:
           steepness_eff = steepness_base * (1 + qs_strength * n_neighbors / n_ref)
       When particles aggregate, they cooperatively strengthen each other's
       coupling to the field, potentially breaking convergence plateaus.

    This creates a mechanism where isolated particles have weak CTC coupling
    (exploring) while clustered particles have strong CTC coupling (converging),
    enabling self-organized phase transitions in convergence behavior.

    Literature:
    - Wolpert, L. (1969) J Theor Biol 25:1-47
      "Positional information and the spatial pattern of cellular differentiation"
    - Lo, C. M. et al. (2000) Biophysical Journal 79:144-152
      "Cell movement is guided by the rigidity of the substrate"
    - Miller, M.B. & Bassler, B.L. (2001) Annual Review of Microbiology 55:165-199
      "Quorum sensing in bacteria"
    - Waters, C.M. & Bassler, B.L. (2005) Annual Review of Cell and Dev Biol 21:319-346
      "Quorum sensing: cell-to-cell communication in bacteria"

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
    """

    PARAMS_DOC = {
        "model_name": "CooperativeCTC",
        "literature": "Wolpert (1969) J Theor Biol 25:1; Lo (2000) Biophys J 79:144; Miller & Bassler (2001) Annu Rev Microbiol 55:165",
        "description": "Durotaxis + tunable CTC steepness + quorum-sensing cooperative density enhancement",
        "equations": {
            "field_to_particle": "v = M * (1+alpha*clamp(|gradC1|,max=1)) * (-tanh(steepness_eff*(C1-T)/A)) * (grad_C1+grad_C2) * dir",
            "steepness_effective": "steepness_eff = steepness_base * (1 + qs_strength * n_neighbors / n_ref)",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = (p1*exp(-d^(2p2)/(2sigma^2)) - p3*exp(-d^(2p4)/(2sigma^2))) * dir"
        },
        "params_mesh": [
            {
                "row": 0, "description": "C1 field parameters + CTC + steepness",
                "slots": [
                    {"index": 0, "name": "D1", "description": "Diffusion coeff for C1", "typical_range": [0.01, 0.5]},
                    {"index": 1, "name": "Da_c", "description": "Damkohler number", "typical_range": [1.0, 50.0]},
                    {"index": 2, "name": "A", "description": "Brusselator param A (also CTC reference)", "typical_range": [0.5, 5.0]},
                    {"index": 3, "name": "B", "description": "Brusselator param B", "typical_range": [1.0, 10.0]},
                    {"index": 4, "name": "mu", "description": "Morphological parameter", "typical_range": [0.01, 0.1]},
                    {"index": 5, "name": "M1", "description": "Mobility for C1 gradients", "typical_range": [-16, 16]},
                    {"index": 6, "name": "ctc_steepness", "description": "CTC tanh switch steepness (DurotaxisThreshold uses 3.0)", "typical_range": [1.0, 10.0]},
                    {"index": 7, "name": "ctc_threshold", "description": "CTC threshold (T=ctc*A)", "typical_range": [0.5, 3.0]}
                ]
            },
            {
                "row": 1, "description": "C2 field parameters + quorum sensing",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coeff for C2", "typical_range": [0.1, 1.0]},
                    {"index": 1, "name": "M2", "description": "Mobility for C2 gradients", "typical_range": [-16, 16]},
                    {"index": 2, "name": "qs_strength", "description": "Quorum sensing strength (0=off, >0=density enhances CTC)", "typical_range": [0.0, 5.0]},
                    {"index": 3, "name": "qs_n_ref", "description": "Quorum sensing reference neighbor count", "typical_range": [5, 50]}
                ]
            },
            {
                "row": 2, "description": "Particle-field coupling + cross-type factor",
                "slots": [
                    {"index": 0, "name": "Pe", "description": "Peclet number", "typical_range": [0.5, 2.0]},
                    {"index": 1, "name": "consumption", "description": "Consumption rate of C1", "typical_range": [10, 200]},
                    {"index": 2, "name": "production", "description": "Production rate of C2", "typical_range": [-200, -10]},
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
        super(PDE_D_CooperativeCTC, self).__init__(aggr=aggr_type)

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

        # Tunable CTC steepness (slot 6 of row 0) — DurotaxisThreshold hardcodes 3.0
        self.ctc_steepness = p[0, 6] if p.shape[1] > 6 else 3.0

        # CTC threshold
        self.ctc_threshold = p[0, 7] if p.shape[1] > 7 else 0.0
        self.A_ref = p[0, 2]

        # Per-type threshold spread
        self.cross_type_factor = p[2, 5] if p.shape[1] > 5 else 0.0

        # Quorum sensing parameters (row 1, slots 2-3)
        self.qs_strength = p[1, 2] if p.shape[1] > 2 else 0.0
        self.qs_n_ref = p[1, 3] if p.shape[1] > 3 else 20.0

        # Required for compatibility
        self.A = torch.tensor(1.0, device=p.device)
        self.B = torch.tensor(0.0, device=p.device)

        print(f"initialized PDE_D_CooperativeCTC with parameters:")
        print(f"  mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        steep_val = self.ctc_steepness.item() if hasattr(self.ctc_steepness, 'item') else self.ctc_steepness
        ctc_val = self.ctc_threshold.item() if hasattr(self.ctc_threshold, 'item') else self.ctc_threshold
        T_val = ctc_val * self.A_ref.item()
        print(f"  ctc_steepness={steep_val:.2f} (tunable, DurotaxisThreshold uses 3.0)")
        print(f"  ctc_threshold={ctc_val:.3f} (T={T_val:.2f}, reversal at C1=T*A)")
        qs_val = self.qs_strength.item() if hasattr(self.qs_strength, 'item') else self.qs_strength
        qs_ref = self.qs_n_ref.item() if hasattr(self.qs_n_ref, 'item') else self.qs_n_ref
        print(f"  quorum_sensing: strength={qs_val:.2f}, n_ref={qs_ref:.0f} (Miller & Bassler 2001)")
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
                    f"PDE_D_CooperativeCTC: particle_params has {n_param_rows} rows but found "
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
            # For fp mode with quorum sensing, we need neighbor counts
            # Pre-compute neighbor counts per particle for quorum sensing
            if self.qs_strength > 0:
                # Count pp-range neighbors for each target node
                target_nodes = edge_index[0]
                n_nodes = x.size(0)
                neighbor_counts = torch.zeros(n_nodes, device=x.device)
                neighbor_counts.scatter_add_(0, target_nodes, torch.ones_like(target_nodes, dtype=torch.float))
                self._neighbor_counts = neighbor_counts
            else:
                self._neighbor_counts = None

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

            # Concentration-threshold coupling with tunable steepness
            if self.ctc_threshold > 0:
                C1_local = fields_i[:, 0:1]
                A_ref = self.A_ref
                base_T = self.ctc_threshold * A_ref

                # Base steepness (tunable via config, not hardcoded)
                steepness = self.ctc_steepness

                # Quorum sensing: enhance steepness based on local neighbor density
                if self.qs_strength > 0 and self._neighbor_counts is not None:
                    n_neighbors = self._neighbor_counts[edge_index_i]
                    qs_factor = 1.0 + self.qs_strength * n_neighbors / self.qs_n_ref
                    steepness = steepness * qs_factor.unsqueeze(1)

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
