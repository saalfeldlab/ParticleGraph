import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_FieldModulated(pyg.nn.MessagePassing):
    """
    Field-dependent mobility and field-modulated particle-particle adhesion.

    Combines two related field-concentration-dependent features:
    1. FDM: Mobility depends on local C1 deviation from Brusselator steady state A
    2. Field-modulated pp: Adhesion strength scales with local C1 concentration

    Literature:
    - Hillen, T. & Painter, K. J. (2009) J Math Biol 58:183-217
      "A user's guide to PDE models for chemotaxis"
    - Hynes, R. O. (2002) Cell 110:673-687
      "Integrins: bidirectional, allosteric signaling machines"
    - Schwartz, M. A. & Ginsberg, M. H. (2002) Nat Cell Biol 4:E65-E68
      "Networks and crosstalk: integrin signaling spreads"

    Physics:
    FDM (positive alpha): M_eff = M * (1 + alpha * clamp((C1-A)^2/A^2, max=4))
    FDM (negative alpha): M_eff = M / (1 + |alpha| * clamp((C1-A)^2/A^2, max=4))
    Field-modulated pp: f_eff = f * (1 + alpha * clamp(C1/C1_ref, 0, 2))

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
    """

    PARAMS_DOC = {
        "model_name": "FieldModulated",
        "literature": "Hillen & Painter (2009) J Math Biol 58:183-217; Hynes (2002) Cell 110:673-687",
        "description": "Field-dependent mobility + field-modulated particle-particle adhesion",
        "equations": {
            "field_to_particle": "v = M * fdm_factor * (grad_C1 + grad_C2) * dir; fdm_factor depends on (C1-A)^2/A^2",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = AR_force * (1 + pp_field_mod * clamp(C1/C1_ref, 0, 2))"
        },
        "params_mesh": [
            {
                "row": 0, "description": "C1 field parameters (shared with mesh model) + FDM control",
                "slots": [
                    {"index": 0, "name": "D1", "description": "Diffusion coeff for C1 (mesh model)", "typical_range": [0.01, 0.5]},
                    {"index": 1, "name": "Da_c", "description": "Damkohler number (mesh model)", "typical_range": [1.0, 50.0]},
                    {"index": 2, "name": "A", "description": "Brusselator param A (mesh model, also FDM reference)", "typical_range": [0.5, 5.0]},
                    {"index": 3, "name": "B", "description": "Brusselator param B (mesh model)", "typical_range": [1.0, 10.0]},
                    {"index": 4, "name": "mu", "description": "Morphological parameter (mesh model)", "typical_range": [0.01, 0.1]},
                    {"index": 5, "name": "M1", "description": "Mobility coefficient for C1 gradients", "typical_range": [-16, 16]},
                    {"index": 6, "name": "fdm_alpha", "description": "Field-dependent mobility (0=off, >0=faster at peaks, <0=slower at peaks)", "typical_range": [-2.0, 2.0]}
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
                "row": 2, "description": "Particle-field coupling + field-modulated pp control",
                "slots": [
                    {"index": 0, "name": "Pe", "description": "Peclet number", "typical_range": [0.5, 2.0]},
                    {"index": 1, "name": "consumption", "description": "Particle consumption rate of C1", "typical_range": [10, 200]},
                    {"index": 2, "name": "production", "description": "Particle production rate of C2", "typical_range": [-200, -10]},
                    {"index": 3, "name": "influence_radius", "description": "Gaussian influence radius for pf coupling", "typical_range": [0.01, 0.1]},
                    {"index": 4, "name": "unused", "description": "Unused (pad)", "typical_range": [0.0, 0.0]},
                    {"index": 5, "name": "unused", "description": "Unused (pad)", "typical_range": [0.0, 0.0]},
                    {"index": 6, "name": "pp_field_mod", "description": "Field-modulated pp adhesion (0=off, >0=stronger at peaks)", "typical_range": [0.0, 1.0]}
                ]
            }
        ],
        "width_constraint": "ALL rows of params_mesh MUST have same number of columns (7). Pad shorter rows.",
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
        super(PDE_D_FieldModulated, self).__init__(aggr=aggr_type)

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

        # FDM: field-dependent mobility
        self.fdm_alpha = p[0, 6] if p.shape[1] > 6 else 0.0
        self.A_ref = p[0, 2]

        # Field-modulated pp adhesion
        if p.shape[0] > 2 and p.shape[1] > 6:
            self.pp_field_mod = p[2, 6]
        else:
            self.pp_field_mod = 0.0

        print(f"initialized PDE_D_FieldModulated with parameters:")
        print(f"  mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        fdm_val = self.fdm_alpha.item() if hasattr(self.fdm_alpha, 'item') else self.fdm_alpha
        print(f"  fdm_alpha={fdm_val:.3f} (M_eff depends on (C1-A)^2/A^2, Hillen & Painter 2009)")
        ppfm_val = self.pp_field_mod.item() if hasattr(self.pp_field_mod, 'item') else self.pp_field_mod
        print(f"  pp_field_mod={ppfm_val:.3f} (f_eff = f*(1+alpha*C1_norm), Hynes 2002)")
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
                    f"PDE_D_FieldModulated: particle_params has {n_param_rows} rows but found "
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

            # Field-dependent mobility (FDM)
            if self.fdm_alpha != 0:
                C1_local = fields_i[:, 0:1]
                A_ref = self.A_ref
                deviation_sq = (C1_local - A_ref) ** 2 / (A_ref ** 2 + 1e-6)
                deviation_sq = torch.clamp(deviation_sq, max=4.0)

                if self.fdm_alpha > 0:
                    fdm_factor = 1.0 + self.fdm_alpha * deviation_sq
                else:
                    fdm_factor = 1.0 / (1.0 + torch.abs(self.fdm_alpha) * deviation_sq)

                velocity_raw = velocity_raw * fdm_factor

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

                # Field-modulated pp adhesion
                if self.pp_field_mod > 0:
                    C1_local = x_i[:, 6]
                    C1_ref = torch.clamp(torch.abs(C1_local).mean(), min=1.0)
                    C1_norm = torch.clamp(C1_local / C1_ref, min=0.0, max=2.0)
                    field_factor = 1.0 + self.pp_field_mod * C1_norm
                    f = f * field_factor

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
