import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D(pyg.nn.MessagePassing):
    """
    Base particle model for diffusiophoresis with attraction-repulsion.

    This is the minimal base model. For extended physics, use a PDE_D variant:
    - PDE_D_LogSensing: Weber-Fechner logarithmic gradient sensing
    - PDE_D_Anisotropic: Directional mobility
    - PDE_D_ActiveMatter: Self-propelled particles + Vicsek alignment
    - PDE_D_DensityDependent: Contact inhibition of locomotion
    - PDE_D_Chiral: Gradient-perpendicular drift (spiral trajectories)
    - PDE_D_Durotaxis: Gradient-amplified mobility
    - PDE_D_MichaelisMenten: Saturating consumption/production kinetics
    - PDE_D_CrossTypeAdhesion: Steinberg differential adhesion sorting
    - PDE_D_FieldModulated: Field-dependent mobility + field-modulated pp
    - PDE_D_ThresholdCoupling: Bistable concentration-threshold coupling

    Receives p=params_mesh (shared, 3 rows x N columns) and optionally
    particle_params from simulation.params (one row per particle type).

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
    """

    PARAMS_DOC = {
        "model_name": "PDE_D",
        "description": "Base particle dynamics: linear diffusiophoresis + consumption/production + attraction-repulsion",
        "equations": {
            "field_to_particle": "v = (M1*grad_C1 + M2*grad_C2) * dir",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = (p1*exp(-d^(2p2)/(2sigma^2)) - p3*exp(-d^(2p4)/(2sigma^2))) * dir"
        },
        "params_mesh": [
            {
                "row": 0, "description": "C1 field parameters (shared with mesh model)",
                "slots": [
                    {"index": 0, "name": "D1", "description": "Diffusion coeff for C1 (used by mesh model)", "typical_range": [0.01, 0.5]},
                    {"index": 1, "name": "Da_c", "description": "Damkohler number (used by mesh model)", "typical_range": [1.0, 50.0]},
                    {"index": 2, "name": "A", "description": "Brusselator param A (used by mesh model)", "typical_range": [0.5, 5.0]},
                    {"index": 3, "name": "B", "description": "Brusselator param B (used by mesh model)", "typical_range": [1.0, 10.0]},
                    {"index": 4, "name": "mu", "description": "Morphological parameter (used by mesh model)", "typical_range": [0.01, 0.1]},
                    {"index": 5, "name": "M1", "description": "Mobility coefficient for C1 gradients", "typical_range": [-16, 16]}
                ]
            },
            {
                "row": 1, "description": "C2 field parameters",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coeff for C2 (used by mesh model)", "typical_range": [0.1, 1.0]},
                    {"index": 1, "name": "M2", "description": "Mobility coefficient for C2 gradients", "typical_range": [-16, 16]}
                ]
            },
            {
                "row": 2, "description": "Particle-field coupling",
                "slots": [
                    {"index": 0, "name": "Pe", "description": "Peclet number", "typical_range": [0.5, 2.0]},
                    {"index": 1, "name": "consumption", "description": "Particle consumption rate of C1", "typical_range": [10, 200]},
                    {"index": 2, "name": "production", "description": "Particle production rate of C2", "typical_range": [-200, -10]},
                    {"index": 3, "name": "influence_radius", "description": "Gaussian influence radius for pf coupling", "typical_range": [0.01, 0.1]}
                ]
            }
        ],
        "particle_params": {
            "description": "Per-type params from simulation.params (one row per n_particle_types)",
            "slots": [
                {"index": 0, "name": "M1", "description": "Per-type mobility for C1 (overrides params_mesh[0][5])"},
                {"index": 1, "name": "M2", "description": "Per-type mobility for C2 (overrides params_mesh[1][1])"},
                {"index": 2, "name": "consumption", "description": "Per-type consumption rate"},
                {"index": 3, "name": "production", "description": "Per-type production rate"},
                {"index": 4, "name": "ar_p1", "description": "Attraction strength: p1*exp(-d^(2*p2)/(2*sigma^2))"},
                {"index": 5, "name": "ar_p2", "description": "Attraction exponent"},
                {"index": 6, "name": "ar_p3", "description": "Repulsion strength: p3*exp(-d^(2*p4)/(2*sigma^2))"},
                {"index": 7, "name": "ar_p4", "description": "Repulsion exponent"}
            ]
        }
    }

    def __init__(self, aggr_type='mean', p=None, particle_params=None, bc_dpos=None, dimension=2, sigma=0.005):
        super(PDE_D, self).__init__(aggr=aggr_type)

        self.p = p
        self.particle_params = particle_params
        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.sigma = sigma

        # Diffusiophoretic mobilities
        self.M1 = p[0, 5]
        self.M2 = p[1, 1]

        # Particle effects on fields
        self.consumption_rate = p[2, 1]
        self.production_rate = p[2, 2]
        self.influence_radius = p[2, 3]

        # Peclet number
        self.Pe = p[2, 0]

        # Particle-particle repulsion
        self.repulsion_strength = 50
        self.repulsion_range = 0.04

        print(f"initialized PDE_D (base) with parameters:")
        print(f"  mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        print(f"  Pe={self.Pe.item():.3f}, sigma={self.sigma}")
        print(f"  particle->field: consumption={self.consumption_rate.item()}, production={self.production_rate.item()}, influence_radius={self.influence_radius.item():.3f}")
        if particle_params is not None:
            print(f"  multi-type support: {particle_params.shape[0]} particle types")
            print(f"  per-type params: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]")

    def forward(self, data, direction='fp'):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # Extract per-type parameters if available
        if self.particle_params is not None:
            particle_type = x[:, 1 + 2*self.dimension].long()
            max_type = particle_type.max().item()
            n_param_rows = self.particle_params.shape[0]
            if max_type >= n_param_rows:
                raise ValueError(
                    f"PDE_D: particle_params has {n_param_rows} rows but found "
                    f"particle type {max_type}. Add {max_type + 1} rows to "
                    f"simulation.params (one per particle type)."
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
        else:  # direction == 'pp'
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
            # Linear diffusiophoresis: v = M1 * grad_C1 + M2 * grad_C2
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

            velocities = (M1 * grad_C1 + M2 * grad_C2) * dir_norm
            return velocities

        elif mode == 'pf':
            # Particle -> Field: consumption/production
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
            # Particle -> Particle: PDE_A-style attraction-repulsion
            if parameters_i is not None:
                p1 = parameters_i[:, 4]
                p2 = parameters_i[:, 5]
                p3 = parameters_i[:, 6]
                p4 = parameters_i[:, 7]

                f = (p1 * torch.exp(-dist ** (2 * p2) / (2 * self.sigma ** 2))
                     - p3 * torch.exp(-dist ** (2 * p4) / (2 * self.sigma ** 2)))

                forces = f[:, None] * d_pos / dist_safe.unsqueeze(1)
            else:
                # Fallback: simple exponential repulsion
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
