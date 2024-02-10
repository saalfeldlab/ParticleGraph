import numpy as np


def linear_model(x, a, b):
    return a * x + b


def power_model(x, a, b):
    return a / (x**b)


def boids_model(x, a, b, c):

    xdiff = x[:, 0:2]
    vdiff = x[:, 2:4]
    r = np.concatenate((x[:,4:5],x[:,4:5]),axis=1)

    sum = a * xdiff + b * vdiff - c * xdiff / r
    sum = np.sqrt(sum[:,0]**2 + sum[:,1]**2)

    return sum


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

    return 0.05 * laplacian + a * uu + b * uv + c * uw + d * vv + e * vw + f * ww + g * u + h * v + i * w


def reaction_diffusion_model(variable_name):
    match variable_name:
        case 'u':
            return lambda x, a, b, c, d, e, f, g, h, i, cc: _aux_reaction_diffusion(x, a, b, c, d, e, f, g, h, i, cc, 0)
        case 'v':
            return lambda x, a, b, c, d, e, f, g, h, i, cc: _aux_reaction_diffusion(x, a, b, c, d, e, f, g, h, i, cc, 1)
        case 'w':
            return lambda x, a, b, c, d, e, f, g, h, i, cc: _aux_reaction_diffusion(x, a, b, c, d, e, f, g, h, i, cc, 2)
