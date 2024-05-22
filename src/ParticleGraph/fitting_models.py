import numpy as np


def power_model(x, a, b):
    return a / (x ** b)


def linear_model(x, a, b):
    return a * x + b


def boids_model(x, a, b, c):
    xdiff = x[:, 0:2]
    vdiff = x[:, 2:4]
    r = np.concatenate((x[:, 4:5], x[:, 4:5]), axis=1)

    total = a * xdiff + b * vdiff - c * xdiff / r
    total = np.sqrt(total[:, 0] ** 2 + total[:, 1] ** 2)

    return total


def _aux_reaction_diffusion(x, a, b, c, d, e, f, g, h, i, cc, idx):
    u = x[:, 3]
    v = x[:, 4]
    w = x[:, 5]

    laplacian = cc * x[:, idx]

    uu = u * u
    uv = u * v
    uw = u * w
    vv = v * v
    vw = v * w
    ww = w * w

    return laplacian + a * uu + b * uv + c * uw + d * vv + e * vw + f * ww + g * u + h * v + i * w

def _aux_reaction_diffusion_L(x, cc, idx):
    u = x[:, 3]
    v = x[:, 4]
    w = x[:, 5]
    a = x[:, 6]
    b = x[:, 7]
    c = x[:, 8]
    d = x[:, 9]
    e = x[:, 10]
    f = x[:, 11]
    g = x[:, 12]
    h = x[:, 13]
    i = x[:, 14]

    laplacian = cc * x[:, idx]

    uu = u * u
    uv = u * v
    uw = u * w
    vv = v * v
    vw = v * w
    ww = w * w

    return laplacian + a * uu + b * uv + c * uw + d * vv + e * vw + f * ww + g * u + h * v + i * w


def reaction_diffusion_model(variable_name):
    match variable_name:
        case 'u':
            return lambda x, a, b, c, d, e, f, g, h, i, cc: _aux_reaction_diffusion(x, a, b, c, d, e, f, g, h, i, cc, 0)
        case 'v':
            return lambda x, a, b, c, d, e, f, g, h, i, cc: _aux_reaction_diffusion(x, a, b, c, d, e, f, g, h, i, cc, 1)
        case 'w':
            return lambda x, a, b, c, d, e, f, g, h, i, cc: _aux_reaction_diffusion(x, a, b, c, d, e, f, g, h, i, cc, 2)


def reaction_diffusion_model_L(variable_name):
    match variable_name:
        case 'u':
            return lambda x, cc: _aux_reaction_diffusion_L(x, cc, 0)
        case 'v':
            return lambda x, cc: _aux_reaction_diffusion_L(x, cc, 1)
        case 'w':
            return lambda x, cc: _aux_reaction_diffusion_L(x, cc, 2)