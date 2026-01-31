import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D(pyg.nn.MessagePassing):
    """Particle model for diffusiophoresis with PDE_A-style attraction-repulsion.

    Receives p=params_mesh (shared, 3 rows x N columns) and optionally
    particle_params from simulation.params (one row per particle type).

    See PARAMS_DOC for the complete parameter layout.
    """

    # PARAMS_DOC: Self-documenting parameter structure for LLM-guided exploration.
    # CRITICAL: All rows of params_mesh must have the SAME number of columns.
    # If you add columns to one row (e.g. row 2 index 7), you MUST pad the other
    # rows to the same width. torch.tensor() requires a rectangular array.
    PARAMS_DOC = {
        "model_name": "PDE_D",
        "description": "Particle dynamics: diffusiophoresis + attraction-repulsion + optional features",
        "params_mesh": [
            {
                "row": 0,
                "description": "C1 field parameters (shared with mesh model PDE_Diffusiophoresis)",
                "slots": [
                    {"index": 0, "name": "D1", "description": "Diffusion coeff for C1 (used by mesh model)", "typical_range": [0.01, 0.5]},
                    {"index": 1, "name": "Da_c", "description": "Damkohler number (used by mesh model)", "typical_range": [1.0, 50.0]},
                    {"index": 2, "name": "A", "description": "Brusselator param A (used by mesh model)", "typical_range": [0.5, 5.0]},
                    {"index": 3, "name": "B", "description": "Brusselator param B (used by mesh model)", "typical_range": [1.0, 10.0]},
                    {"index": 4, "name": "mu", "description": "Morphological parameter (used by mesh model)", "typical_range": [0.01, 0.1]},
                    {"index": 5, "name": "M1", "description": "Mobility coefficient for C1 gradients", "typical_range": [-16, 16]},
                    {"index": 6, "name": "fdm_alpha", "description": "Field-dependent mobility strength (0=off, >0=faster at peaks, <0=slower at peaks)", "typical_range": [-2.0, 2.0]}
                ]
            },
            {
                "row": 1,
                "description": "C2 field parameters + particle feature controls",
                "slots": [
                    {"index": 0, "name": "D2", "description": "Diffusion coeff for C2 (used by mesh model)", "typical_range": [0.1, 1.0]},
                    {"index": 1, "name": "M2", "description": "Mobility coefficient for C2 gradients", "typical_range": [-16, 16]},
                    {"index": 2, "name": "mm_Km", "description": "Michaelis-Menten half-saturation (0=constant rate)", "typical_range": [0.0, 0.5]},
                    {"index": 3, "name": "grad_amp_alpha", "description": "Durotaxis gradient amplification (0=off)", "typical_range": [0.0, 2.0]},
                    {"index": 4, "name": "chirality", "description": "Chiral drift fraction (0=straight, >0=CCW, <0=CW)", "typical_range": [-0.3, 0.3]},
                    {"index": 5, "name": "ddm_beta", "description": "Density-dependent mobility / contact inhibition (0=off)", "typical_range": [0.0, 1.0]}
                ]
            },
            {
                "row": 2,
                "description": "Particle-field coupling + particle-particle feature controls",
                "slots": [
                    {"index": 0, "name": "Pe", "description": "Peclet number", "typical_range": [0.5, 2.0]},
                    {"index": 1, "name": "consumption", "description": "Particle consumption rate of C1", "typical_range": [10, 200]},
                    {"index": 2, "name": "production", "description": "Particle production rate of C2", "typical_range": [-200, -10]},
                    {"index": 3, "name": "influence_radius", "description": "Gaussian influence radius for pf coupling", "typical_range": [0.01, 0.1]},
                    {"index": 4, "name": "log_sensing_K", "description": "Weber-Fechner half-saturation (0=linear sensing)", "typical_range": [0.0, 2.0]},
                    {"index": 5, "name": "cross_type_factor", "description": "Cross-type adhesion inversion (0=off, >0=Steinberg sorting)", "typical_range": [0.0, 0.5]},
                    {"index": 6, "name": "pp_field_mod", "description": "Field-modulated pp adhesion (0=off)", "typical_range": [0.0, 1.0]},
                    {"index": 7, "name": "alignment_strength", "description": "Vicsek velocity alignment (0=off)", "typical_range": [0.0, 1.0]}
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
        },
        "width_constraint": "ALL rows of params_mesh MUST have the same number of columns. Default is 6. If enabling features at indices 6-7, pad ALL rows to 8."
    }

    def __init__(self, aggr_type='mean', p=None, particle_params=None, bc_dpos=None, dimension=2, sigma=0.005):
        super(PDE_D, self).__init__(aggr=aggr_type)

        self.p = p  # Mesh params (shared across all types)
        self.particle_params = particle_params  # Per-type particle params (optional)
        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.sigma = sigma  # For attraction-repulsion kernel

        # Global parameters from mesh (used as fallback when particle_params=None)
        # Diffusiophoretic parameters
        self.M1 = p[0, 5]       # Mobility coefficient for C₁
        self.M2 = p[1, 1]       # Mobility coefficient for C₂

        # Particle effects on fields
        self.consumption_rate = p[2, 1]
        self.production_rate = p[2, 2]
        self.influence_radius = p[2, 3]

        # Péclet number
        self.Pe = p[2, 0]

        # Particle-particle repulsion parameters
        # Block 2 code change: increased range 0.025->0.04 to encourage spreading
        self.repulsion_strength = 50
        self.repulsion_range = 0.04

        # Block 8 code change: Logarithmic gradient sensing (Weber-Fechner law)
        # p[2, 4] controls sensing mode:
        #   0.0 = linear sensing (v = M * grad_C, default, backward compatible)
        #   >0  = logarithmic sensing with half-saturation K = p[2,4]
        #         v = M * grad_C / (K + |C|)
        # Literature: Keller & Segel (1971) J Theor Biol 30:225-234
        #   Logarithmic chemotactic sensitivity: chi(C) = chi_0 / C
        #   Biological cells sense relative, not absolute, concentration differences
        # Effect: Compresses dynamic range — particles respond equally to weak and strong gradients
        #   This should create more uniform filaments instead of dense point clusters
        self.log_sensing_K = p[2, 4] if p.shape[1] > 4 else 0.0

        # Block 9 code change: Cross-type differential adhesion factor
        # p[2, 5] controls cross-type pp interaction modification:
        #   0.0 = no modification (backward compatible, default)
        #   >0  = cross-type force = -cross_type_factor * same-type force
        #         This inverts attraction→repulsion for cross-type pairs
        # Literature: Steinberg (1963) Science 141:401-408
        # Effect: Same-type particles attract, cross-type particles repel → cell sorting
        self.cross_type_factor = p[2, 5] if p.shape[1] > 5 else 0.0

        # Block 10 code change: Michaelis-Menten concentration-dependent feedback
        # p[1, 2] controls Km (half-saturation constant):
        #   0.0 = constant consumption/production rates (backward compatible, default)
        #   >0  = Michaelis-Menten: effective_rate = base_rate * |C1| / (Km + |C1|)
        # Literature: Michaelis & Menten (1913) Biochem Z 49:333-369
        # Effect: Consumption/production depends on local field concentration
        #   Creates nonlinear feedback — particles affect fields more at concentration peaks
        #   Could produce sharper pattern boundaries or oscillatory coupling dynamics
        self.mm_Km = p[1, 2] if p.shape[1] > 2 else 0.0

        # Block 11 code change: Gradient-amplified mobility (durotaxis)
        # p[1, 3] controls gradient amplification strength (alpha):
        #   0.0 = standard constant mobility (backward compatible, default)
        #   >0  = mobility scales with local gradient magnitude:
        #         M_effective = M * (1 + alpha * clamp(|grad_C1|, max=1.0))
        # Literature: Lo et al. (2000) Biophys J 79:144-152 "Cell movement is guided
        #   by the rigidity of the substrate"; Isenberg et al. (2009) Biophys J 97:1313-1322
        # Block 12 fix: gradient clamped at max_grad=1.0 to prevent boundary mask
        #   artifacts (|grad|~5-20) from catastrophic amplification (iter 43: 72.60% retention).
        #   Interior FHN gradients ~0.1-1.0 are preserved.
        self.grad_amp_alpha = p[1, 3] if p.shape[1] > 3 else 0.0

        # Block 13 code change: Chiral diffusiophoresis (gradient-perpendicular drift)
        # p[1, 4] controls chirality (perpendicular drift fraction):
        #   0.0 = pure gradient-parallel motion (backward compatible, default)
        #   >0  = CCW spiral: adds perpendicular component rotated 90° CCW
        #   <0  = CW spiral: perpendicular component rotated 90° CW
        # Literature: Löwen (2016) Eur Phys J Special Topics 225:2319-2331;
        #   Friedrich & Jülicher (2007) PNAS 104:13256
        # Effect: v_total = v_parallel + chirality * v_perpendicular
        #   where v_perpendicular = (-vy, vx) rotation of the gradient-parallel velocity
        #   Creates spiral trajectories around concentration features rather than
        #   direct ascent/descent. Particles orbit peaks/valleys instead of collecting there.
        self.chirality = p[1, 4] if p.shape[1] > 4 else 0.0

        # Block 14 code change: Field-modulated particle-particle adhesion
        # p[2, 6] controls concentration-dependent pp force scaling:
        #   0.0 = constant pp forces (backward compatible, default)
        #   >0  = pp force scales with local field: f_eff = f * (1 + alpha * C1_norm)
        #         where C1_norm = clamp(C1_local / C1_ref, 0, 2), C1_ref = A (Brusselator steady state)
        # Literature: Hynes (2002) Cell 110:673-687 "Integrins: bidirectional,
        #   allosteric signaling machines"; Schwartz & Ginsberg (2002) Nat Cell Biol
        #   4:E65-E68 "Networks and crosstalk: integrin signaling spreads"
        # Biological motivation: cell-cell adhesion (integrin/cadherin) is regulated by
        #   local growth factor concentration. In high-signal regions, cells form stronger
        #   adhesions; in low-signal regions, cells are more loosely associated.
        # Effect: Particles cluster TIGHTER at Turing spot peaks (high C1) and remain
        #   more dispersed between spots. Creates differential compaction that should
        #   enhance morphological complexity at pattern boundaries.
        if p.shape[0] > 2 and p.shape[1] > 6:
            self.pp_field_mod = p[2, 6]
        else:
            self.pp_field_mod = 0.0

        # Block 15 code change: Density-dependent mobility (contact inhibition)
        # p[1, 5] controls density-dependent slowdown strength (beta):
        #   0.0 = constant mobility (backward compatible, default)
        #   >0  = v_eff = v / (1 + beta * n_neighbors)
        #         Particles with many pp neighbors move slower
        # Literature: Mayor & Carmona-Fontaine (2010) Trends Cell Biol 20:319-328;
        #   Stramer & Mayor (2017) Nat Rev Mol Cell Biol 18:43-55
        # Effect: Sharp cluster boundaries — interior immobilized, edge cells free
        self.ddm_beta = p[1, 5] if p.shape[1] > 5 else 0.0
        # Storage for neighbor count computed during 'pp' pass, used in 'fp' pass
        self._neighbor_count = None

        # Block 16 code change: Velocity alignment (Vicsek-style collective motion)
        # p[2, 7] controls alignment strength:
        #   0.0 = no alignment (backward compatible, default)
        #   >0  = f_align = alignment * (v_neighbor - v_self) * weight(distance)
        #         Added to pp forces. Creates coherent flows/streams/flocking.
        # Literature: Vicsek et al. (1995) Phys Rev Lett 75:1226-1229;
        #   Chaté et al. (2008) Phys Rev E 77:046113
        # Effect: Inter-particle velocity coordination → streaming/flocking
        if p.shape[0] > 2 and p.shape[1] > 7:
            self.alignment_strength = p[2, 7]
        else:
            self.alignment_strength = 0.0

        # Block 13 code change: Field-dependent mobility (FDM)
        # p[0, 6] controls fdm_alpha (field-dependent mobility strength):
        #   0.0 = constant mobility (backward compatible, default)
        #   >0  = M_eff = M * (1 + fdm_alpha * clamp((C1-A)^2/A^2, max=4.0))
        #         Particles move FASTER at field peaks/troughs (far from steady state)
        #         and SLOWER near the Brusselator steady state (C1 ≈ A)
        #   <0  = M_eff = M / (1 + |fdm_alpha| * clamp((C1-A)^2/A^2, max=4.0))
        #         Particles move SLOWER at field peaks/troughs and FASTER near steady state
        # Literature: Hillen & Painter (2009) J Math Biol 58:183-217
        #   "A user's guide to PDE models for chemotaxis" — concentration-dependent
        #   chemotactic sensitivity chi(C) is standard in Keller-Segel family models.
        #   Also: Painter & Hillen (2002) Can Appl Math Q 10:501-543 — volume-filling
        #   chemotaxis where cells move slower in crowded (high-concentration) regions.
        # Rationale: The 7/10 ceiling (96 iterations, Blocks 1-12) is a COUPLING
        #   bottleneck: particles respond uniformly to gradients. FDM creates nonlinear
        #   coupling where accumulation dynamics depend on local field state, enabling
        #   multi-scale organization (tight clusters at peaks, dispersed at boundaries).
        # Effect: With positive alpha, particles accumulate more strongly at Turing peaks
        #   (where C1 >> A) because their mobility is amplified there. This creates
        #   positive feedback: peaks attract particles faster → particles consume C1 →
        #   pattern reorganizes. With negative alpha, particles are more mobile near
        #   steady state and immobilized at peaks → smeared distributions.
        if p.shape[1] > 6:
            self.fdm_alpha = p[0, 6]
        else:
            self.fdm_alpha = 0.0

        # Brusselator parameter A for FDM normalization
        # Stored from mesh params row 0, index 2
        self.A_ref = p[0, 2]

        # Report configuration
        print(f"initialized PDE_D with parameters:")
        print(f"mobility: M₁={self.M1.item()}, M₂={self.M2.item()}")
        if hasattr(self, 'log_sensing_K'):
            K_val = self.log_sensing_K.item() if hasattr(self.log_sensing_K, 'item') else self.log_sensing_K
            if K_val > 0:
                print(f"logarithmic sensing: K={K_val:.3f} (Weber-Fechner: v = M*grad_C/(K+|C|))")
            else:
                print(f"sensing mode: linear (K=0)")
        print(f"Pe={self.Pe.item():.3f}, sigma={self.sigma}")
        print(f"particle→Field: consumption={self.consumption_rate.item()}, production={self.production_rate.item()}, influence_radius={self.influence_radius.item():.3f}")
        if hasattr(self, 'cross_type_factor'):
            ctf_val = self.cross_type_factor.item() if hasattr(self.cross_type_factor, 'item') else self.cross_type_factor
            if ctf_val > 0:
                print(f"cross-type differential adhesion: factor={ctf_val:.2f} (Steinberg sorting)")
            else:
                print(f"cross-type adhesion: off (factor=0)")
        if hasattr(self, 'mm_Km'):
            mm_val = self.mm_Km.item() if hasattr(self.mm_Km, 'item') else self.mm_Km
            if mm_val > 0:
                print(f"Michaelis-Menten feedback: Km={mm_val:.3f} (rate = base * |C1|/(Km+|C1|))")
            else:
                print(f"consumption/production: constant rate (Km=0)")
        if hasattr(self, 'grad_amp_alpha'):
            ga_val = self.grad_amp_alpha.item() if hasattr(self.grad_amp_alpha, 'item') else self.grad_amp_alpha
            if ga_val > 0:
                print(f"gradient-amplified mobility (durotaxis): alpha={ga_val:.3f} (M_eff = M*(1+alpha*clamp(|gradC|,max=1.0)))")
            else:
                print(f"gradient amplification: off (alpha=0)")
        if hasattr(self, 'chirality'):
            ch_val = self.chirality.item() if hasattr(self.chirality, 'item') else self.chirality
            if ch_val != 0:
                print(f"chiral diffusiophoresis: chirality={ch_val:.3f} ({'CCW' if ch_val > 0 else 'CW'} spiral, Löwen 2016)")
            else:
                print(f"chiral drift: off (chirality=0)")
        if hasattr(self, 'pp_field_mod'):
            ppfm_val = self.pp_field_mod.item() if hasattr(self.pp_field_mod, 'item') else self.pp_field_mod
            if ppfm_val > 0:
                print(f"field-modulated pp adhesion: alpha={ppfm_val:.3f} (f_eff = f*(1+alpha*C1_norm), Hynes 2002)")
            else:
                print(f"field-modulated pp: off (alpha=0)")
        if hasattr(self, 'ddm_beta'):
            ddm_val = self.ddm_beta.item() if hasattr(self.ddm_beta, 'item') else self.ddm_beta
            if ddm_val > 0:
                print(f"density-dependent mobility (CIL): beta={ddm_val:.3f} (v_eff = v/(1+beta*n_neighbors), Mayor 2010)")
            else:
                print(f"density-dependent mobility: off (beta=0)")
        if hasattr(self, 'alignment_strength'):
            align_val = self.alignment_strength.item() if hasattr(self.alignment_strength, 'item') else self.alignment_strength
            if align_val > 0:
                print(f"velocity alignment (Vicsek): strength={align_val:.3f} (f_align = alpha*(v_j-v_i)*w(d), Vicsek 1995)")
            else:
                print(f"velocity alignment: off (strength=0)")
        if hasattr(self, 'fdm_alpha'):
            fdm_val = self.fdm_alpha.item() if hasattr(self.fdm_alpha, 'item') else self.fdm_alpha
            if fdm_val != 0:
                print(f"field-dependent mobility (FDM): alpha={fdm_val:.3f} (M_eff = M*(1+alpha*clamp((C1-A)^2/A^2,max=4)), Hillen & Painter 2009)")
            else:
                print(f"field-dependent mobility: off (alpha=0)")
        if particle_params is not None:
            print(f"multi-type support: {particle_params.shape[0]} particle types")
            print(f"per-type params: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]")
    
    def forward(self, data, direction='fp'):
        """
        Compute interactions based on direction
        """
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # Extract per-type parameters if available (like PDE_A pattern)
        if self.particle_params is not None:
            # Particle type is at index 1 + 2*dimension (after pos_x, pos_y)
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
            # Step 1: Interpolate fields from mesh to particles
            result = self.propagate(edge_index, x=x, mode='interpolate', parameters=parameters)

            # For out-of-box particles, return zero fields (no valid mesh data)
            pos = x[:, 1:self.dimension+1]
            in_box = ((pos >= 0) & (pos <= 1)).all(dim=1, keepdim=True)
            result = result * in_box.float()

            return result
        elif direction == 'fp':
            # Step 2: Calculate diffusiophoretic velocities
            result = self.propagate(edge_index, x=x, mode='fp', parameters=parameters)

            # Zero out velocities for particles outside [0,1] box
            # These particles have no valid field data to interpolate from
            pos = x[:, 1:self.dimension+1]
            in_box = ((pos >= 0) & (pos <= 1)).all(dim=1, keepdim=True)
            result = result * in_box.float()

            # Block 15: Density-dependent mobility (contact inhibition)
            # Scale velocity by 1/(1 + beta * n_neighbors)
            # Uses neighbor count computed during prior 'pp' pass
            if hasattr(self, 'ddm_beta') and self.ddm_beta > 0 and self._neighbor_count is not None:
                # _neighbor_count has size [n_particles_in_pp_graph]
                # result has size [n_nodes + n_particles] (fp graph includes mesh nodes)
                # Only particles (indices >= n_nodes) should be scaled
                n_result = result.shape[0]
                n_neighbors = self._neighbor_count.shape[0]
                if n_neighbors <= n_result:
                    # Create scaling factor: 1/(1 + beta * count)
                    scale = 1.0 / (1.0 + self.ddm_beta * self._neighbor_count)
                    # Apply to the LAST n_neighbors entries (particles are after mesh nodes)
                    result[-n_neighbors:] = result[-n_neighbors:] * scale.unsqueeze(1)

            return result
        elif direction == 'pf':
            # Particle → Field effects
            result = self.propagate(edge_index, x=x, mode='pf', parameters=parameters)
            return result
        else:  # direction == 'pp'
            # Particle → Particle repulsion
            result = self.propagate(edge_index, x=x, mode='pp', parameters=parameters)

            # Block 15: Count pp neighbors for density-dependent mobility
            # Store neighbor count per particle for use in subsequent 'fp' call
            if hasattr(self, 'ddm_beta') and self.ddm_beta > 0:
                n_particles = x.shape[0]
                # Count incoming edges per node (= number of pp neighbors)
                neighbor_count = torch.zeros(n_particles, device=x.device)
                neighbor_count.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], device=x.device))
                self._neighbor_count = neighbor_count

            return result
    
    def message(self, edge_index_i, edge_index_j, x_i, x_j, mode=None, parameters_i=None):
        """
        Compute messages based on mode

        Per-type params layout (when parameters_i provided):
        [M1, M2, consumption, production, repulsion_strength, repulsion_range]
        """
        # Get positions
        pos_i = x_i[:, 1:self.dimension+1]
        pos_j = x_j[:, 1:self.dimension+1]

        # Calculate displacement vectors with boundary conditions
        d_pos = self.bc_dpos(pos_j - pos_i)
        dist = torch.sqrt(torch.sum(d_pos**2, dim=1))
        dist_safe = torch.clamp(dist, min=1e-6)

        if mode == 'interpolate':
            # Interpolate fields from mesh nodes to particles
            # x_j are mesh nodes (senders), x_i are particles (receivers)

            # Get field values from mesh
            C1_mesh = x_j[:, 6:7]
            C2_mesh = x_j[:, 7:8]

            # Distance-based weights (inverse distance or Gaussian)
            weight = torch.exp(-dist / 0.01).unsqueeze(1)  # Gaussian kernel

            # Return weighted fields for aggregation
            return torch.cat([C1_mesh * weight, C2_mesh * weight, weight], dim=1)

        elif mode == 'fp':
            # Field differences
            fields_i = x_i[:, 6:8]  # Particle fields [C₁, C₂]
            fields_j = x_j[:, 6:8]  # Mesh fields [C₁, C₂]

            dC1 = fields_j[:, 0:1] - fields_i[:, 0:1]
            dC2 = fields_j[:, 1:2] - fields_i[:, 1:2]

            # Use smoothing kernel to avoid sharp gradients
            kernel = torch.exp(-dist / 0.05)  # Smoothing length scale

            # Direction vector
            dir_norm = d_pos / dist_safe.unsqueeze(1)

            # Smooth gradient estimation (not raw difference/distance)
            # Scale by domain size since positions are [0,1] but physics expects [0,32]
            domain_scale = 32.0
            grad_C1 = (dC1 * kernel.unsqueeze(1)) / (dist_safe.unsqueeze(1) * domain_scale)
            grad_C2 = (dC2 * kernel.unsqueeze(1)) / (dist_safe.unsqueeze(1) * domain_scale)

            # Get mobility coefficients (per-type or global)
            if parameters_i is not None:
                M1 = parameters_i[:, 0:1]  # Per-particle type
                M2 = parameters_i[:, 1:2]
            else:
                M1 = self.M1  # Global fallback
                M2 = self.M2

            # Block 8: Logarithmic gradient sensing (Weber-Fechner law)
            # Keller & Segel (1971): chi(C) = chi_0 / C gives logarithmic response
            # Implementation: v = M * grad_C / (K + |C|) where K = half-saturation
            # When K=0: linear sensing (v = M * grad_C), backward compatible
            # When K>0: logarithmic sensing, compresses dynamic range
            #   - At low |C| << K: v ≈ M * grad_C / K (attenuated, avoids singularity)
            #   - At high |C| >> K: v ≈ M * grad_C / |C| (logarithmic, relative sensing)
            # Effect: Particles respond to relative gradients, not absolute — should create
            #   more uniform filaments instead of dense clusters at concentration peaks
            if hasattr(self, 'log_sensing_K') and self.log_sensing_K > 0:
                # Get local field concentrations for Weber-Fechner denominator
                C1_local = torch.abs(fields_i[:, 0:1]) + 1e-6  # Avoid division by zero
                C2_local = torch.abs(fields_i[:, 1:2]) + 1e-6
                K = self.log_sensing_K
                # Logarithmic sensitivity: scale gradient by 1/(K + |C|)
                log_grad_C1 = grad_C1 / (K + C1_local)
                log_grad_C2 = grad_C2 / (K + C2_local)
                velocity_raw = (M1 * log_grad_C1 + M2 * log_grad_C2) * dir_norm
            else:
                # Linear sensing (default, backward compatible)
                velocity_raw = (M1 * grad_C1 + M2 * grad_C2) * dir_norm

            # Block 11: Gradient-amplified mobility (durotaxis)
            # Lo et al. (2000): cells migrate faster on stiffer substrates
            # Here, gradient magnitude serves as "stiffness" — particles respond
            # more strongly at pattern boundaries where gradients are steep.
            # M_effective = M * (1 + alpha * clamp(|grad_C1|, max=1.0))
            # When alpha=0: no change (backward compatible)
            # When alpha>0: amplifies velocity at high-gradient regions
            # Block 12 fix: clamp grad_mag at 1.0 to prevent boundary mask artifacts
            # (|grad|~5-20) from catastrophic amplification. Interior FHN pattern
            # gradients are typically 0.1-1.0, so clamping preserves intended physics.
            if hasattr(self, 'grad_amp_alpha') and self.grad_amp_alpha > 0:
                # Compute local gradient magnitude from C1 gradient
                grad_mag = torch.abs(grad_C1)  # |dC1/dr| per edge
                # Clamp to prevent boundary mask artifacts from dominating
                grad_mag_clamped = torch.clamp(grad_mag, max=1.0)
                # Amplification factor: 1 + alpha * clamped_grad
                amp_factor = 1.0 + self.grad_amp_alpha * grad_mag_clamped
                velocity_raw = velocity_raw * amp_factor

            # Block 13: Chiral diffusiophoresis (gradient-perpendicular drift)
            # Löwen (2016): chirality creates spiral trajectories along gradients
            # Friedrich & Jülicher (2007): sperm chemotaxis has perpendicular component
            # v_total = v_parallel + chirality * rotate_90(v_parallel)
            # rotate_90_CCW(vx, vy) = (-vy, vx) in 2D
            # When chirality=0: no change (backward compatible)
            # When chirality>0: particles spiral CCW around concentration features
            # When chirality<0: particles spiral CW
            if hasattr(self, 'chirality') and self.chirality != 0:
                # velocity_raw has shape [n_edges, 2] for 2D
                v_perp = torch.zeros_like(velocity_raw)
                v_perp[:, 0] = -velocity_raw[:, 1]  # -vy
                v_perp[:, 1] = velocity_raw[:, 0]   # +vx
                velocity_raw = velocity_raw + self.chirality * v_perp

            # Block 13: Field-dependent mobility (FDM)
            # Hillen & Painter (2009): concentration-dependent chemotactic sensitivity
            # M_eff depends on local C1 value relative to Brusselator steady state A.
            # Positive alpha: particles move faster at peaks/troughs (far from A)
            # Negative alpha: particles move slower at peaks/troughs (immobilized at features)
            # Uses (C1 - A)^2 / A^2 as dimensionless deviation, clamped for stability.
            if hasattr(self, 'fdm_alpha') and self.fdm_alpha != 0:
                C1_local = fields_i[:, 0:1]  # Local C1 at particle position
                A_ref = self.A_ref
                # Dimensionless squared deviation from steady state
                deviation_sq = (C1_local - A_ref) ** 2 / (A_ref ** 2 + 1e-6)
                deviation_sq = torch.clamp(deviation_sq, max=4.0)

                if self.fdm_alpha > 0:
                    # Positive: amplify mobility at peaks/troughs
                    fdm_factor = 1.0 + self.fdm_alpha * deviation_sq
                else:
                    # Negative: suppress mobility at peaks/troughs
                    fdm_factor = 1.0 / (1.0 + torch.abs(self.fdm_alpha) * deviation_sq)

                velocity_raw = velocity_raw * fdm_factor

            velocities = velocity_raw

            return velocities

        elif mode == 'pf':
            # Particle → Field: Calculate field updates
            # Gaussian influence based on distance
            weights = torch.exp(-dist**2 / (2 * (self.influence_radius/3)**2))

            # Get consumption/production rates (per-type or global)
            if parameters_i is not None:
                consumption = parameters_i[:, 2]
                production = parameters_i[:, 3]
            else:
                consumption = self.consumption_rate
                production = self.production_rate

            # Block 10: Michaelis-Menten concentration-dependent modulation
            # When mm_Km > 0, scale consumption/production by local field concentration
            # rate_effective = base_rate * |C1| / (Km + |C1|)
            # This makes particle-field coupling stronger at high concentrations
            # and weaker at low concentrations — nonlinear feedback
            if hasattr(self, 'mm_Km') and self.mm_Km > 0:
                # x_i contains particle features; field values at indices 6:8
                C1_local = torch.abs(x_i[:, 6]) + 1e-6  # Local C1 at particle position
                mm_factor = C1_local / (self.mm_Km + C1_local)  # Michaelis-Menten factor [0, 1)
                consumption = consumption * mm_factor
                production = production * mm_factor

            # Create field updates [C₁, C₂]
            field_updates = torch.zeros((pos_i.size(0), 2), device=pos_i.device)
            field_updates[:, 0] = -consumption * weights
            field_updates[:, 1] = production * weights

            return field_updates

        else:  # mode == 'pp'
            # Particle → Particle: PDE_A-style attraction-repulsion
            # Formula: f = (p1 * exp(-d^(2*p2) / (2σ²)) - p3 * exp(-d^(2*p4) / (2σ²))) * direction
            #
            # Block 9: Cross-type differential adhesion (Steinberg 1963)
            # When cross_type_factor > 0 and particles are different types:
            #   force = -cross_type_factor * same_type_force
            # This inverts the force for cross-type pairs: what attracts same-type
            # repels cross-type, creating spontaneous cell sorting.

            if parameters_i is not None:
                # Per-type attraction-repulsion parameters
                p1 = parameters_i[:, 4]  # ar_p1: attraction strength
                p2 = parameters_i[:, 5]  # ar_p2: attraction exponent
                p3 = parameters_i[:, 6]  # ar_p3: repulsion strength
                p4 = parameters_i[:, 7]  # ar_p4: repulsion exponent

                # PDE_A formula: attraction - repulsion
                f = (p1 * torch.exp(-dist ** (2 * p2) / (2 * self.sigma ** 2))
                     - p3 * torch.exp(-dist ** (2 * p4) / (2 * self.sigma ** 2)))

                # Cross-type differential adhesion
                # Check if sender and receiver are different types
                if hasattr(self, 'cross_type_factor') and self.cross_type_factor > 0:
                    type_i = x_i[:, 1 + 2*self.dimension].long()
                    type_j = x_j[:, 1 + 2*self.dimension].long()
                    cross_type = (type_i != type_j).float()
                    # Invert force for cross-type pairs:
                    # same_type: f unchanged; cross_type: f * (-cross_type_factor)
                    modifier = 1.0 - cross_type * (1.0 + self.cross_type_factor)
                    f = f * modifier

                # Block 14: Field-modulated pp adhesion (Hynes 2002, Schwartz & Ginsberg 2002)
                # When pp_field_mod > 0, scale pp force by local field concentration:
                # f_eff = f * (1 + alpha * C1_norm)
                # where C1_norm = clamp(C1_local / C1_ref, 0, 2)
                # C1_ref = A (Brusselator steady state) normalizes so modulation ~1x at steady state
                # At Turing peaks (C1 > A): stronger adhesion → tighter clusters
                # At Turing troughs (C1 < A): weaker adhesion → more dispersed
                # This creates differential compaction: pattern peaks become dense cores,
                # inter-spot regions remain loose → enhanced morphological contrast
                if hasattr(self, 'pp_field_mod') and self.pp_field_mod > 0:
                    # Get local C1 concentration at particle i position
                    C1_local = x_i[:, 6]  # Field C1 interpolated to particle position
                    # Normalize by Brusselator steady state (A ~ 4-6 typically)
                    C1_ref = torch.clamp(torch.abs(C1_local).mean(), min=1.0)
                    C1_norm = torch.clamp(C1_local / C1_ref, min=0.0, max=2.0)
                    # Modulate force: stronger at peaks, weaker at troughs
                    field_factor = 1.0 + self.pp_field_mod * C1_norm
                    f = f * field_factor

                # Apply force in direction of neighbor (attraction positive, repulsion negative)
                forces = f[:, None] * d_pos / dist_safe.unsqueeze(1)

                # Block 16: Velocity alignment (Vicsek 1995, Chaté 2008)
                # Add alignment force: each particle is pushed toward matching its
                # neighbors' velocities. Uses NORMALIZED velocity difference to avoid
                # force scale mismatch (pp forces are O(sigma) but velocities can be
                # O(100+) from diffusiophoresis). Block 17 fix: normalize + clamp.
                # Velocity is stored at x[:, 3:5] (vx, vy) for 2D.
                if hasattr(self, 'alignment_strength') and self.alignment_strength > 0:
                    vel_i = x_i[:, self.dimension+1:2*self.dimension+1]  # v_self [vx, vy]
                    vel_j = x_j[:, self.dimension+1:2*self.dimension+1]  # v_neighbor [vx, vy]
                    # Distance-weighted alignment: closer neighbors have more influence
                    align_weight = torch.exp(-dist / 0.04).unsqueeze(1)  # Same scale as pp range
                    # Normalize velocity difference to unit direction, then scale
                    # to pp-compatible magnitude (sigma-based)
                    vel_diff = vel_j - vel_i
                    vel_diff_mag = torch.norm(vel_diff, dim=1, keepdim=True).clamp(min=1e-6)
                    vel_diff_dir = vel_diff / vel_diff_mag  # Unit direction of alignment
                    # Scale alignment force to match pp force scale (~sigma)
                    f_align = self.alignment_strength * vel_diff_dir * align_weight * self.sigma
                    # Hard clamp for numerical safety
                    f_align = torch.clamp(f_align, min=-0.1, max=0.1)
                    forces = forces + f_align

            else:
                # Fallback: simple exponential repulsion (backward compatible)
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
        """
        Process aggregated messages
        """
        if mode == 'interpolate':
            # Normalize weighted average
            C1_weighted = aggr_out[:, 0:1]
            C2_weighted = aggr_out[:, 1:2]
            weight_sum = aggr_out[:, 2:3]
            
            # Avoid division by zero
            weight_sum = torch.clamp(weight_sum, min=1e-10)
            
            # Return normalized fields
            return torch.cat([C1_weighted / weight_sum, C2_weighted / weight_sum], dim=1)
        else:
            return aggr_out