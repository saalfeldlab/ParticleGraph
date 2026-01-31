import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy

class PDE_D(pyg.nn.MessagePassing):
    """
    Computes interactions between particles and fields, and between particles.
    Implements diffusiophoresis with PDE_A-style attraction-repulsion.

    Supports multiple particle types when particle_params is provided.
    Per-type params layout: [M1, M2, consumption, production, ar_p1, ar_p2, ar_p3, ar_p4]
      - M1, M2: Mobility coefficients for diffusiophoresis
      - consumption, production: Particle effects on fields
      - ar_p1, ar_p2: Attraction term parameters (p1*exp(-d^(2*p2)/(2σ²)))
      - ar_p3, ar_p4: Repulsion term parameters (p3*exp(-d^(2*p4)/(2σ²)))

    Block 9 code change: Cross-type differential adhesion
    Literature: Steinberg (1963) Science 141:401-408 "Reconstruction of tissues by
    dissociated cells"; Foty & Steinberg (2005) Dev Biol 278:255-263
    When n_particle_types > 1, pp interactions are type-dependent:
      - Same-type pairs use params as-is (attraction + repulsion from ar_p1-p4)
      - Cross-type pairs SWAP attraction/repulsion: what attracts same-type REPELS
        cross-type and vice versa. This creates differential adhesion — cells of the
        same type stick together, cells of different types push apart.
    Controlled by p[2, 5] (cross_type_factor):
      0.0 = no cross-type modification (backward compatible, default)
      >0  = cross-type interactions are scaled by -cross_type_factor
            (negative = inverted attraction/repulsion for cross-type pairs)

    Block 10 code change: Michaelis-Menten concentration-dependent consumption/production
    Literature: Michaelis & Menten (1913) Biochem Z 49:333-369 "Die Kinetik der
    Invertinwirkung"; Johnson & Goody (2011) Biochemistry 50:8264-8269 (modern review)
    Standard enzyme kinetics: rate = Vmax * [S] / (Km + [S])
    Applied to particle-field coupling: consumption/production rates become dependent
    on the local field concentration rather than being constant.
    Controlled by p[1, 2] (mm_Km):
      0.0 = constant rate (backward compatible, default)
      >0  = Michaelis-Menten kinetics: effective_rate = base_rate * |C1| / (Km + |C1|)
            At low |C1| << Km: rate ≈ base_rate * |C1| / Km (linear, weak)
            At high |C1| >> Km: rate ≈ base_rate (saturated, full strength)
            This creates nonlinear feedback: particles consume/produce more where
            field concentrations are strong, less where they are weak.

    Block 11 code change: Gradient-amplified mobility (durotaxis)
    Literature: Lo et al. (2000) Biophys J 79:144-152 "Cell movement is guided by
    the rigidity of the substrate"; Isenberg et al. (2009) Biophys J 97:1313-1322
    Cells migrate faster in regions of steep chemical/mechanical gradients. Here,
    the local field gradient magnitude acts as a "stiffness" signal that modulates
    particle mobility — particles at pattern boundaries (steep gradients) respond
    more strongly than particles in flat-field regions.
    Controlled by p[1, 3] (grad_amp_alpha):
      0.0 = constant mobility (backward compatible, default)
      >0  = gradient-amplified: M_effective = M * (1 + alpha * clamp(|grad_C1|, max=1.0))
            This creates selective concentration at pattern edges rather than
            broad peak/valley occupation.
    Block 12 fix: Added gradient clamping (max_grad=1.0) to prevent boundary mesh
    mask artifacts (|grad| ~ 5-20) from catastrophically amplifying mobility. Interior
    FHN pattern gradients are typically |grad| ~ 0.1-1.0, so clamping at 1.0 preserves
    the intended physics while neutralizing boundary artifacts. Iter 43 showed alpha=2.0
    without clamping caused 72.60% retention (catastrophic boundary escape).
    """

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

            return result
        elif direction == 'pf':
            # Particle → Field effects
            result = self.propagate(edge_index, x=x, mode='pf', parameters=parameters)
            return result
        else:  # direction == 'pp'
            # Particle → Particle repulsion
            result = self.propagate(edge_index, x=x, mode='pp', parameters=parameters)
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

                # Apply force in direction of neighbor (attraction positive, repulsion negative)
                forces = f[:, None] * d_pos / dist_safe.unsqueeze(1)
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