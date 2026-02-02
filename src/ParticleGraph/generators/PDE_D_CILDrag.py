import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_CILDrag(pyg.nn.MessagePassing):
    """
    Contact Inhibition of Locomotion (CIL) with velocity-dependent fp drag.

    Combines two orthogonal damping mechanisms that target different failure modes:
    1. CIL (Mayor & Carmona-Fontaine 2010): Reduces mobility at high local density,
       preventing runaway aggregation. Targets spatial overshoot.
    2. fp_drag (Tranquillo & Lauffenburger 1987): Reduces chemotactic response at
       high velocity, preventing temporal overshoot. Targets velocity oscillations.

    Critically, this variant does NOT include CTC (concentration-threshold coupling).
    CTC + CIL was shown to create a positive feedback loop causing field blowup
    for 1-type particles (Iter 187). By removing CTC, CIL and fp_drag provide
    independent stabilization without mutual amplification.

    Physics:
    1. fp: Density-modulated diffusiophoresis with velocity drag
       v_raw = M * nabla_C
       v_cil = v_raw * f(rho)        where f(rho) = 1/(1+(rho/rho_0)^n)
       v_final = v_cil / (1 + fp_drag * |vel| / v_ref)
    2. pf: Standard consumption/production coupling
    3. pp: Standard attraction-repulsion (no field-dependent damping since no CTC)

    Literature:
    - Mayor, R. & Carmona-Fontaine, C. (2010) Trends in Cell Biology 20:319-328
      "Keeping in touch with contact inhibition of locomotion"
    - Tranquillo, R. T. & Lauffenburger, D. A. (1987) J Math Biol 25:229-262
      "Stochastic model of leukocyte chemosensory movement"
    - Cates, M. E. & Tailleur, J. (2015) ARCMP 6:219-244
      "Motility-induced phase separation"

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
    """

    PARAMS_DOC = {
        "model_name": "CILDrag",
        "literature": "Mayor & Carmona-Fontaine (2010); Tranquillo & Lauffenburger (1987); Cates & Tailleur (2015)",
        "description": "CIL density-dependent mobility + velocity-dependent fp drag (no CTC)",
        "equations": {
            "field_to_particle": "v = M * nabla_C * f(rho) / (1 + fp_drag * |vel| / v_ref)",
            "density_function": "f(rho) = 1 / (1 + (rho/rho_0)^n)",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = (p1*exp(-d^(2p2)/(2sigma^2)) - p3*exp(-d^(2p4)/(2sigma^2))) * dir"
        },
        "params_mesh": [
            {
                "row": 0, "description": "C1 field parameters",
                "slots": [
                    {"index": 0, "name": "D1", "description": "Diffusion coeff for C1", "typical_range": [0.01, 0.5]},
                    {"index": 1, "name": "Da_c", "description": "Damkohler number", "typical_range": [1.0, 50.0]},
                    {"index": 2, "name": "A", "description": "Brusselator A", "typical_range": [0.5, 5.0]},
                    {"index": 3, "name": "B", "description": "Brusselator B", "typical_range": [1.0, 10.0]},
                    {"index": 4, "name": "mu", "description": "Morphological param", "typical_range": [0.01, 0.1]},
                    {"index": 5, "name": "M1", "description": "Mobility for C1 gradients", "typical_range": [-16, 16]},
                    {"index": 6, "name": "unused_0", "description": "Unused (pad)", "typical_range": [0, 0]},
                    {"index": 7, "name": "unused_1", "description": "Unused (pad)", "typical_range": [0, 0]}
                ]
            },
            {
                "row": 1, "description": "C2 field parameters",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coeff for C2", "typical_range": [0.1, 1.0]},
                    {"index": 1, "name": "M2", "description": "Mobility for C2 gradients", "typical_range": [-16, 16]},
                    {"index": 2, "name": "unused_2", "description": "Unused (pad)", "typical_range": [0, 0]},
                    {"index": 3, "name": "unused_3", "description": "Unused (pad)", "typical_range": [0, 0]}
                ]
            },
            {
                "row": 2, "description": "Particle-field coupling + CIL + fp drag",
                "slots": [
                    {"index": 0, "name": "Pe", "description": "Peclet number", "typical_range": [0.5, 2.0]},
                    {"index": 1, "name": "consumption", "description": "Consumption rate of C1", "typical_range": [10, 200]},
                    {"index": 2, "name": "production", "description": "Production rate of C2", "typical_range": [-200, -10]},
                    {"index": 3, "name": "influence_radius", "description": "Gaussian pf influence radius", "typical_range": [0.01, 0.1]},
                    {"index": 4, "name": "rho_0", "description": "CIL critical density threshold", "typical_range": [10.0, 50.0]},
                    {"index": 5, "name": "hill_n", "description": "Hill coefficient for CIL", "typical_range": [1.0, 4.0]},
                    {"index": 6, "name": "fp_drag", "description": "Velocity-dependent fp drag coefficient", "typical_range": [0.0, 2.0]},
                    {"index": 7, "name": "unused_4", "description": "Unused (pad)", "typical_range": [0, 0]}
                ]
            }
        ],
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
        super(PDE_D_CILDrag, self).__init__(aggr=aggr_type)

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

        # CIL density-dependent parameters (Mayor & Carmona-Fontaine 2010)
        self.rho_0 = p[2, 4] if p.shape[1] > 4 and p[2, 4] != 0 else 28.0
        self.hill_n = p[2, 5] if p.shape[1] > 5 and p[2, 5] != 0 else 2.0
        self.sensing_radius = 0.05

        # Convert to proper tensors if needed
        if not isinstance(self.rho_0, torch.Tensor):
            self.rho_0 = torch.tensor(float(self.rho_0), device=p.device)
        if not isinstance(self.hill_n, torch.Tensor):
            self.hill_n = torch.tensor(float(self.hill_n), device=p.device)

        # Velocity-dependent fp drag (Tranquillo & Lauffenburger 1987)
        self.fp_drag = p[2, 6] if p.shape[1] > 6 else 0.0
        self.v_ref = 0.01

        if not isinstance(self.fp_drag, torch.Tensor):
            self.fp_drag = torch.tensor(float(self.fp_drag), device=p.device)

        # Storage for local density (computed in pp pass, used in fp pass)
        self.local_density = None

        # Report configuration
        rho0_val = self.rho_0.item() if hasattr(self.rho_0, 'item') else self.rho_0
        hill_val = self.hill_n.item() if hasattr(self.hill_n, 'item') else self.hill_n
        fp_drag_val = self.fp_drag.item() if hasattr(self.fp_drag, 'item') else self.fp_drag
        print(f"initialized PDE_D_CILDrag with parameters:")
        print(f"  mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        print(f"  CIL: rho_0={rho0_val}, hill_n={hill_val}, sensing_radius={self.sensing_radius} (Mayor 2010)")
        print(f"  fp_drag={fp_drag_val:.3f}, v_ref={self.v_ref:.4f} (Tranquillo 1987)")
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
                    f"PDE_D_CILDrag: particle_params has {n_param_rows} rows but found "
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

            # Apply CIL density-dependent modulation (computed in pp pass)
            if self.local_density is not None:
                n_total = x.size(0)
                n_particles = self.local_density.size(0)
                n_nodes = n_total - n_particles

                ratio = self.local_density / self.rho_0
                modulation = 1.0 / (1.0 + ratio ** self.hill_n)

                mod_full = torch.ones(n_total, 1, device=x.device)
                mod_full[n_nodes:, 0] = modulation
                result = result * mod_full

            pos = x[:, 1:self.dimension+1]
            in_box = ((pos >= 0) & (pos <= 1)).all(dim=1, keepdim=True)
            result = result * in_box.float()
            return result
        elif direction == 'pf':
            result = self.propagate(edge_index, x=x, mode='pf', parameters=parameters)
            return result
        else:  # direction == 'pp'
            self._compute_local_density(x, edge_index)
            result = self.propagate(edge_index, x=x, mode='pp', parameters=parameters)
            return result

    def _compute_local_density(self, x, edge_index):
        """Count particle neighbors within sensing_radius for CIL."""
        n_particles = x.size(0)
        target_nodes = edge_index[1]

        pos_i = x[edge_index[1], 1:self.dimension+1]
        pos_j = x[edge_index[0], 1:self.dimension+1]
        d_pos = self.bc_dpos(pos_j - pos_i)
        dist = torch.sqrt(torch.sum(d_pos**2, dim=1))

        within_radius = dist < self.sensing_radius
        counts = torch.zeros(n_particles, device=x.device)
        counts.scatter_add_(0, target_nodes[within_radius],
                           torch.ones(within_radius.sum(), device=x.device))

        self.local_density = counts

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

            # Velocity-dependent fp drag (Tranquillo & Lauffenburger 1987)
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
