import torch
from ParticleGraph.utils import *
import torch_geometric.data as data


def MPM_3D_step(
        model_MPM,
        X,
        V,
        C,
        F,
        T,
        Jp,
        M,
        n_particles,
        n_grid,
        dt,
        dx,
        inv_dx,
        mu_0,
        lambda_0,
        p_vol,
        offsets,
        particle_offsets,
        expansion_factor,
        gravity,
        friction,
        frame,
        device,
):
    """
    3D MPM substep implementation
    """

    # Material masks
    liquid_mask = (T.squeeze() == 0)
    jelly_mask = (T.squeeze() == 1)
    snow_mask = (T.squeeze() == 2)

    # Initialize mass
    p_mass = M.squeeze(-1)

    # Create 3D identity matrices for all particles
    identity = torch.eye(3, device=device).unsqueeze(0).expand(n_particles, -1, -1)

    # Calculate F ############################################################################################

    # Update deformation gradient: F = (I + dt * C) * F_old
    F = (identity + dt * C) @ F

    # Hardening coefficient
    h = torch.exp(10 * (1.0 - Jp.squeeze()))
    h = torch.where(jelly_mask, torch.tensor(0.3, device=device), h)

    # Lamé parameters
    mu = mu_0 * h
    la = lambda_0 * h
    mu = torch.where(liquid_mask, torch.tensor(0.0, device=device), mu)

    # SVD decomposition for 3x3 matrices
    U, sig, Vh = torch.linalg.svd(F, driver='gesvdj')

    # SVD sign correction without in-place ops for 3D
    det_U = torch.det(U)
    det_Vh = torch.det(Vh)
    neg_det_U = det_U < 0  # [n_particles] bool tensor
    neg_det_Vh = det_Vh < 0

    # Reshape masks for broadcasting
    neg_det_U_mask = neg_det_U.unsqueeze(-1).unsqueeze(-1)  # [n_particles,1,1]
    neg_det_sig_U_mask = neg_det_U.unsqueeze(-1)  # [n_particles,1]
    neg_det_Vh_mask = neg_det_Vh.unsqueeze(-1).unsqueeze(-1)  # [n_particles,1,1]
    neg_det_sig_Vh_mask = neg_det_Vh.unsqueeze(-1)  # [n_particles,1]

    # Flip signs on last columns/rows accordingly, out-of-place for 3D
    U = torch.where(
        neg_det_U_mask.expand_as(U),
        torch.cat([U[:, :, :-1], -U[:, :, -1:].clone()], dim=2),
        U
    )
    sig = torch.where(
        neg_det_sig_U_mask.expand_as(sig),
        torch.cat([sig[:, :-1], -sig[:, -1:].clone()], dim=1),
        sig
    )
    Vh = torch.where(
        neg_det_Vh_mask.expand_as(Vh),
        torch.cat([Vh[:, :-1, :], -Vh[:, -1:, :].clone()], dim=1),
        Vh
    )

    # Clamp singular values for snow plasticity (3D)
    sig_clamped = torch.clamp(sig, 1.0 - 2.5e-2, 1.0 + 4.5e-3)
    sig = torch.where(snow_mask.unsqueeze(-1).expand_as(sig), sig_clamped, sig)

    # Update Jp for snow
    old_J = torch.det(F)
    new_J = torch.prod(sig, dim=1)
    Jp = torch.where(snow_mask.unsqueeze(-1), Jp * old_J / new_J, Jp)

    # Reconstruct F
    F = U @ torch.diag_embed(sig) @ Vh

    # Calculate stress tensor (3D Cauchy stress)
    J = torch.det(F)

    # Create diagonal sigma matrix for 3D
    sig_matrix = torch.diag_embed(sig)

    # 3D stress calculation
    stress = (2 * mu.unsqueeze(-1).unsqueeze(-1) * (F - U @ Vh) @ F.transpose(-2, -1) +
              la.unsqueeze(-1).unsqueeze(-1) * (J - 1).unsqueeze(-1).unsqueeze(-1) *
              torch.eye(3, device=device).unsqueeze(0).expand(n_particles, -1, -1)) / J.unsqueeze(-1).unsqueeze(-1)

    # P2G Transfer ############################################################################################

    # Initialize 3D grid
    grid_m = torch.zeros((n_grid, n_grid, n_grid), dtype=torch.float32, device=device)
    grid_v = torch.zeros((n_grid, n_grid, n_grid, 3), dtype=torch.float32, device=device)  # 3D velocity

    # Particle to grid mapping
    base = (X * inv_dx - 0.5).int()  # [n_particles, 3]
    fx = X * inv_dx - base.float()  # [n_particles, 3]

    # 3D B-spline weights
    w_0 = 0.5 * (1.5 - fx) ** 2
    w_1 = 0.75 - (fx - 1) ** 2
    w_2 = 0.5 * (fx - 0.5) ** 2
    w = torch.stack([w_0, w_1, w_2], dim=1)  # [n_particles, 3, 3]

    # 27-neighbor loop for 3D
    for i in range(3):
        for j in range(3):
            for k in range(3):
                offset = torch.tensor([i, j, k], device=device)
                dpos = (offset.float() - fx) * dx  # [n_particles, 3]
                weight = w[:, i, 0] * w[:, j, 1] * w[:, k, 2]  # [n_particles]

                # Grid indices with boundary checking
                grid_pos = base + offset  # [n_particles, 3]
                valid_mask = ((grid_pos >= 0).all(dim=1) &
                              (grid_pos < n_grid).all(dim=1))

                if valid_mask.any():
                    valid_particles = torch.where(valid_mask)[0]
                    valid_grid_pos = grid_pos[valid_particles]
                    valid_weight = weight[valid_particles]

                    # Mass contribution
                    mass_contrib = p_mass[valid_particles] * valid_weight

                    # Momentum contribution (including affine and stress)
                    if len(valid_particles) > 0:
                        affine_contrib = torch.bmm(C[valid_particles], dpos[valid_particles].unsqueeze(-1)).squeeze(-1)
                        stress_contrib = -dt * p_vol * torch.bmm(stress[valid_particles],
                                                                 dpos[valid_particles].unsqueeze(-1)).squeeze(-1)
                    else:
                        continue

                    # Weighted momentum contribution
                    weighted_velocity = valid_weight.unsqueeze(-1) * V[valid_particles]
                    weighted_affine = valid_weight.unsqueeze(-1) * affine_contrib
                    weighted_stress = valid_weight.unsqueeze(-1) * stress_contrib

                    momentum = (p_mass[valid_particles].unsqueeze(-1) *
                                (weighted_velocity + weighted_affine + weighted_stress))

                    # Scatter to grid
                    for idx in range(len(valid_particles)):
                        gx, gy, gz = valid_grid_pos[idx]
                        grid_m[gx, gy, gz] += mass_contrib[idx]
                        grid_v[gx, gy, gz] += momentum[idx]

    # Grid Operations ############################################################################################

    # Convert momentum to velocity
    valid_mass_mask = grid_m > 0
    grid_v[valid_mass_mask] = grid_v[valid_mass_mask] / grid_m[valid_mass_mask].unsqueeze(-1)

    # Apply gravity (3D)
    gravity_3d = torch.tensor([0.0, gravity, 0.0], device=device)  # Gravity in y-direction
    grid_v[valid_mass_mask] += dt * gravity_3d

    # Boundary conditions for 3D grid
    boundary_mask = torch.zeros_like(valid_mass_mask)

    # Set boundary conditions for all 6 faces of the cube
    boundary_mask[0, :, :] = True  # x=0 face
    boundary_mask[-1, :, :] = True  # x=n_grid-1 face
    boundary_mask[:, 0, :] = True  # y=0 face
    boundary_mask[:, -1, :] = True  # y=n_grid-1 face
    boundary_mask[:, :, 0] = True  # z=0 face
    boundary_mask[:, :, -1] = True  # z=n_grid-1 face

    # Apply boundary conditions with friction
    grid_v[boundary_mask] *= friction

    # G2P Transfer ############################################################################################

    # Reset particle velocities and affine matrix
    new_v = torch.zeros_like(V)
    new_C = torch.zeros_like(C)

    # Grid to particle transfer
    for i in range(3):
        for j in range(3):
            for k in range(3):
                offset = torch.tensor([i, j, k], device=device)
                dpos = (offset.float() - fx) * dx  # [n_particles, 3]
                weight = w[:, i, 0] * w[:, j, 1] * w[:, k, 2]  # [n_particles]

                # Grid indices with boundary checking
                grid_pos = base + offset  # [n_particles, 3]
                valid_mask = ((grid_pos >= 0).all(dim=1) &
                              (grid_pos < n_grid).all(dim=1))

                if valid_mask.any():
                    valid_particles = torch.where(valid_mask)[0]
                    valid_grid_pos = grid_pos[valid_particles]
                    valid_weight = weight[valid_particles]
                    valid_dpos = dpos[valid_particles]

                    if len(valid_particles) == 0:
                        continue

                    # Gather grid velocities
                    grid_vel = torch.stack([
                        grid_v[valid_grid_pos[:, 0], valid_grid_pos[:, 1], valid_grid_pos[:, 2]]
                    ])

                    if len(grid_vel.shape) == 3:
                        grid_vel = grid_vel.squeeze(0)

                    # Update particle velocity
                    new_v[valid_particles] += valid_weight.unsqueeze(-1) * grid_vel

                    # Update affine matrix (3x3)
                    if len(valid_particles) > 0:
                        new_C[valid_particles] += (4 * inv_dx * inv_dx * valid_weight).unsqueeze(-1).unsqueeze(
                            -1) * torch.bmm(
                            grid_vel.unsqueeze(-1), valid_dpos.unsqueeze(1)
                        )

    # Update particle positions
    X = X + dt * new_v

    # Boundary conditions for particles
    X = torch.clamp(X, 0.01, 0.99)

    # Update particle velocities where they hit boundaries
    boundary_hit = (X <= 0.01) | (X >= 0.99)
    new_v = torch.where(boundary_hit, new_v * 0.0, new_v)

    # Prepare grid outputs (flatten for compatibility)
    GM = grid_m.flatten()
    GV = grid_v.reshape(-1, 3)

    return X, new_v, new_C, F, T, Jp, M, stress, GM, GV