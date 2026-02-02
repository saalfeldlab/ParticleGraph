import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D_GradientCIL(pyg.nn.MessagePassing):
    """
    Gradient-adaptive contact inhibition of locomotion (CIL) for 1-type particles.

    Extends DensityDependent by making the critical density threshold (rho_0)
    depend on the local field gradient magnitude. In regions with steep concentration
    gradients (near Turing pattern boundaries), rho_0 is HIGHER, allowing denser
    packing. In flat-field regions (far from patterns), rho_0 is LOWER, causing
    faster dispersal. This creates preferential accumulation at pattern edges.

    Physical motivation: cells at tissue boundaries experience stronger ECM
    (extracellular matrix) gradients, which upregulate adhesion molecules and
    reduce CIL sensitivity, allowing tighter packing (Lo et al. 2000; Theveneau
    et al. 2010). This is the haptotaxis-CIL coupling.

    The gradient-adaptive rho_0:
        rho_0_local = rho_0 * (1 + grad_sensitivity * |grad_C1| / grad_ref)
    where:
        grad_sensitivity controls how much gradient amplifies rho_0
        grad_ref is a normalization scale for gradient magnitude

    Physics:
    1. fp: Linear diffusiophoresis with density-dependent modulation
       v = M * f(rho, |grad_C1|) * nabla_C
       f(rho, g) = 1 / (1 + (rho / rho_0(g))^n)
       rho_0(g) = rho_0 * (1 + grad_sensitivity * |g| / grad_ref)
    2. pf: Standard consumption/production coupling
    3. pp: Standard attraction-repulsion

    Literature:
    - Cates, M. E. & Tailleur, J. (2015) ARCMP 6:219-244
      "Motility-induced phase separation"
    - Mayor, R. & Carmona-Fontaine, C. (2010) Trends Cell Biol 20:319-328
      "Keeping in touch with contact inhibition of locomotion"
    - Lo, C. M. et al. (2000) Biophysical Journal 79:144-152
      "Cell movement is guided by the rigidity of the substrate"
    - Theveneau, E. et al. (2010) Dev Cell 19:39-53
      "Collective chemotaxis requires contact-dependent cell polarity"

    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
    """

    PARAMS_DOC = {
        "model_name": "GradientCIL",
        "literature": "Cates & Tailleur (2015); Mayor & Carmona-Fontaine (2010); Lo (2000); Theveneau et al. (2010)",
        "description": "Gradient-adaptive CIL: density threshold increases at steep field gradients, enabling tighter packing at pattern boundaries",
        "equations": {
            "field_to_particle": "v = M * f(rho, |grad_C1|) * nabla_C",
            "density_function": "f(rho, g) = 1 / (1 + (rho / rho_0(g))^n)",
            "adaptive_threshold": "rho_0(g) = rho_0 * (1 + grad_sensitivity * |g| / grad_ref)",
            "particle_to_field": "dC1 = -consumption * w(r), dC2 = production * w(r)",
            "particle_to_particle": "f = (p1*exp(-d^(2p2)/(2sigma^2)) - p3*exp(-d^(2p4)/(2sigma^2))) * dir"
        },
        "params_mesh": [
            {
                "row": 0, "description": "C1 field parameters",
                "slots": [
                    {"index": 0, "name": "D1", "description": "Diffusion coeff for C1", "typical_range": [0.01, 0.5]},
                    {"index": 1, "name": "Da_c", "description": "Damkohler number", "typical_range": [1.0, 50.0]},
                    {"index": 2, "name": "A", "description": "Brusselator param A", "typical_range": [0.5, 5.0]},
                    {"index": 3, "name": "B", "description": "Brusselator param B", "typical_range": [1.0, 10.0]},
                    {"index": 4, "name": "mu", "description": "Morphological param", "typical_range": [0.01, 0.1]},
                    {"index": 5, "name": "M1", "description": "Mobility for C1 gradients", "typical_range": [-16, 16]},
                    {"index": 6, "name": "grad_sensitivity", "description": "How much gradient amplifies rho_0 (0=off)", "typical_range": [0.0, 5.0]},
                    {"index": 7, "name": "grad_ref", "description": "Reference gradient scale for normalization", "typical_range": [0.01, 1.0]}
                ]
            },
            {
                "row": 1, "description": "C2 field parameters + CIL params",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coeff for C2", "typical_range": [0.1, 1.0]},
                    {"index": 1, "name": "M2", "description": "Mobility for C2 gradients", "typical_range": [-16, 16]},
                    {"index": 2, "name": "rho_0", "description": "Base critical density threshold", "typical_range": [10, 60]},
                    {"index": 3, "name": "hill_n", "description": "Hill coefficient (cooperativity)", "typical_range": [1, 4]}
                ]
            },
            {
                "row": 2, "description": "Particle-field coupling",
                "slots": [
                    {"index": 0, "name": "Pe", "description": "Peclet number", "typical_range": [0.5, 2.0]},
                    {"index": 1, "name": "consumption", "description": "C1 consumption rate", "typical_range": [10, 200]},
                    {"index": 2, "name": "production", "description": "C2 production rate", "typical_range": [-200, -10]},
                    {"index": 3, "name": "influence_radius", "description": "Gaussian pf influence radius", "typical_range": [0.01, 0.1]}
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
        super(PDE_D_GradientCIL, self).__init__(aggr=aggr_type)

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

        # Gradient-adaptive CIL parameters
        self.grad_sensitivity = p[0, 6] if p.shape[1] > 6 else 0.0
        self.grad_ref = p[0, 7] if p.shape[1] > 7 else 0.1

        # Density-dependent mobility parameters
        self.rho_0 = p[1, 2] if p.shape[1] > 2 and p[1, 2] != 0 else 35.0
        self.hill_n = p[1, 3] if p.shape[1] > 3 and p[1, 3] != 0 else 2.0
        self.sensing_radius = 0.05

        # Convert to proper tensors if needed
        if not isinstance(self.rho_0, torch.Tensor):
            self.rho_0 = torch.tensor(float(self.rho_0), device=p.device)
        if not isinstance(self.hill_n, torch.Tensor):
            self.hill_n = torch.tensor(float(self.hill_n), device=p.device)
        if not isinstance(self.grad_sensitivity, torch.Tensor):
            self.grad_sensitivity = torch.tensor(float(self.grad_sensitivity), device=p.device)
        if not isinstance(self.grad_ref, torch.Tensor):
            self.grad_ref = torch.tensor(float(self.grad_ref), device=p.device)

        # Storage for local density and gradient magnitude
        self.local_density = None
        self.local_grad_mag = None

        rho0_val = self.rho_0.item() if hasattr(self.rho_0, 'item') else self.rho_0
        hill_val = self.hill_n.item() if hasattr(self.hill_n, 'item') else self.hill_n
        gs_val = self.grad_sensitivity.item() if hasattr(self.grad_sensitivity, 'item') else self.grad_sensitivity
        gr_val = self.grad_ref.item() if hasattr(self.grad_ref, 'item') else self.grad_ref
        print(f"initialized PDE_D_GradientCIL with parameters:")
        print(f"  mobility: M1={self.M1.item()}, M2={self.M2.item()}")
        print(f"  density-dependent: rho_0={rho0_val}, hill_n={hill_val}, sensing_radius={self.sensing_radius}")
        print(f"  gradient-adaptive: grad_sensitivity={gs_val:.3f}, grad_ref={gr_val:.3f}")
        print(f"    At |grad_C1|=0: rho_0_eff = {rho0_val:.1f}")
        print(f"    At |grad_C1|=grad_ref: rho_0_eff = {rho0_val * (1 + gs_val):.1f}")
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
                    f"PDE_D_GradientCIL: particle_params has {n_param_rows} rows but found "
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

            # Apply gradient-adaptive density-dependent modulation
            if self.local_density is not None:
                n_total = x.size(0)
                n_particles = self.local_density.size(0)
                n_nodes = n_total - n_particles

                # Compute gradient-adaptive rho_0 for each particle
                if self.local_grad_mag is not None and self.grad_sensitivity > 0:
                    rho_0_local = self.rho_0 * (1.0 + self.grad_sensitivity * self.local_grad_mag / (self.grad_ref + 1e-8))
                else:
                    rho_0_local = self.rho_0

                # Hill function with adaptive threshold
                ratio = self.local_density / (rho_0_local + 1e-8)
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
            self._compute_local_gradient(x, edge_index)
            result = self.propagate(edge_index, x=x, mode='pp', parameters=parameters)
            return result

    def _compute_local_density(self, x, edge_index):
        """Count particle neighbors within sensing radius."""
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

    def _compute_local_gradient(self, x, edge_index):
        """Estimate local field gradient magnitude for each particle from pp graph.

        Uses interpolated C1 values already stored in x[:, 6] (the particle's
        local C1 from the interpolation pass). Computes gradient magnitude from
        differences with neighbors.
        """
        n_particles = x.size(0)
        target_nodes = edge_index[1]
        source_nodes = edge_index[0]

        pos_i = x[target_nodes, 1:self.dimension+1]
        pos_j = x[source_nodes, 1:self.dimension+1]
        d_pos = self.bc_dpos(pos_j - pos_i)
        dist = torch.sqrt(torch.sum(d_pos**2, dim=1))

        C1_i = x[target_nodes, 6]
        C1_j = x[source_nodes, 6]
        dC1 = C1_j - C1_i

        # Estimate gradient magnitude as |dC1/dr| weighted by proximity
        dist_safe = torch.clamp(dist, min=1e-6)
        within_radius = dist < self.sensing_radius
        grad_est = torch.abs(dC1) / dist_safe

        # Average gradient magnitude per particle
        grad_sum = torch.zeros(n_particles, device=x.device)
        grad_count = torch.zeros(n_particles, device=x.device)
        grad_sum.scatter_add_(0, target_nodes[within_radius], grad_est[within_radius])
        grad_count.scatter_add_(0, target_nodes[within_radius],
                               torch.ones(within_radius.sum(), device=x.device))
        grad_count = torch.clamp(grad_count, min=1.0)
        self.local_grad_mag = grad_sum / grad_count

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

            velocities = (M1 * grad_C1 + M2 * grad_C2) * dir_norm
            return velocities

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
