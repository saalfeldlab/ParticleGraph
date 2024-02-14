def choose_training_model(model_config, device):

    if model_config['model'] == 'PDE_G':
        model = Interaction_Particles(aggr_type=aggr_type, model_config=model_config, device=device, bc_dpos=bc_dpos)
    if model_config['model'] == 'PDE_E':
        model = Interaction_Particles(aggr_type=aggr_type, model_config=model_config, device=device, bc_dpos=bc_dpos)
    if (model_config['model'] == 'PDE_A') | (model_config['model'] == 'PDE_B'):
        model = Interaction_Particles(aggr_type=aggr_type, model_config=model_config, device=device, bc_dpos=bc_dpos)
    if (model_config['model'] == 'DiffMesh'):
        model = Mesh_Laplacian(aggr_type=aggr_type, model_config=model_config, device=device, bc_dpos=bc_dpos)
    if (model_config['model'] == 'WaveMesh'):
        model = Mesh_Laplacian(aggr_type=aggr_type, model_config=model_config, device=device, bc_dpos=bc_dpos)
    if (model_config['model'] == 'RD_RPS_Mesh'):
        model = Mesh_RPS(aggr_type=aggr_type, model_config=model_config, device=device, bc_dpos=bc_dpos)

    return model, mesh, bc_pos, bc_dpos
