import numpy as np


def func_pow(x, a, b):
    return a / (x ** b)


def func_lin(x, a, b):
    return a * x + b


def func_boids(x, a, b, c):
    xdiff = x[:, 0:2]
    vdiff = x[:, 2:4]
    r = np.concatenate((x[:, 4:5], x[:, 4:5]), axis=1)

    sum = a * xdiff + b * vdiff - c * xdiff / r
    sum = np.sqrt(sum[:, 0] ** 2 + sum[:, 1] ** 2)

    return sum


def func_RD1(x, a, b, c, d, e, f, g, h, i, cc):
    u = x[:, 3]
    v = x[:, 4]
    w = x[:, 5]

    laplacian_u = cc * x[:, 0]
    laplacian_v = cc * x[:, 1]
    laplacian_w = cc * x[:, 2]

    uu = u * u
    uv = u * v
    uw = u * w
    vv = v * v
    vw = v * w
    ww = w * w

    du = 0.05 * laplacian_u + a * uu + b * uv + c * uw + d * vv + e * vw + f * ww + g * u + h * v + i * w

    return du


def func_RD2(x, a, b, c, d, e, f, g, h, i, cc):
    u = x[:, 3]
    v = x[:, 4]
    w = x[:, 5]

    laplacian_u = cc * x[:, 0]
    laplacian_v = cc * x[:, 1]
    laplacian_w = cc * x[:, 2]

    uu = u * u
    uv = u * v
    uw = u * w
    vv = v * v
    vw = v * w
    ww = w * w

    dv = 0.05 * laplacian_v + a * uu + b * uv + c * uw + d * vv + e * vw + f * ww + g * u + h * v + i * w

    return dv


def func_RD3(x, a, b, c, d, e, f, g, h, i, cc):
    u = x[:, 3]
    v = x[:, 4]
    w = x[:, 5]

    laplacian_u = cc * x[:, 0]
    laplacian_v = cc * x[:, 1]
    laplacian_w = cc * x[:, 2]

    uu = u * u
    uv = u * v
    uw = u * w
    vv = v * v
    vw = v * w
    ww = w * w

    dw = 0.05 * laplacian_w + a * uu + b * uv + c * uw + d * vv + e * vw + f * ww + g * u + h * v + i * w

    return dw
