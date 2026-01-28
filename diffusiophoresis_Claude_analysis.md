# Pattern Exploration Log: diffusiophoresis

## Iter 1: 5/10
Node: id=1, parent=root
Mode/Strategy: baseline
Config: params_mesh=[[0.05, 15.0, 4.5, 6.5, 0.04, -16], [0.8, 16, 0, 0, 0, 0], [1.0, 180, -180, 0.05, 0, 0]], n_frames=2000, delta_t=0.0005, n_particles=9600, n_nodes=10000
n_particle_types: 1
Metrics: entropy=0.74, plateau=0.00, in_box=85.1%
Score: 5/10
Visual: Clear Turing spot patterns in C1/C2 fields. Particles start uniform, progressively cluster around field spots. Strong field pattern development (C1_std=3.96, pattern_growth=246). Particle clusters visible but some escape from domain.
Mutation: None (baseline configuration)
Observation: Field dynamics working well - Brusselator produces clear spots. Particle-field coupling creates visible co-localization. Issues: plateau=0 indicates particles still moving at end (not stabilized), 15% particle escape suggests mobility parameters may be too high.
Next: parent=1, explore reducing mobility to improve particle retention

