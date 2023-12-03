import numpy as np


def forward(ic, n_t, dt, n_x, dx):
    """
    calcualte u with ic by First Order Upwind Scheme
    :param ic: initial condition
    :param n_t: time steps to take
    :param dt: time step
    :param n_x: number of grid points
    :param dx: grid step
    :return: final u
    """
    result = ic.reshape((n_x, 1))
    for j in range(n_t - 1):
        u = np.vstack((0.5, result[1:n_x, [j]] - (result[1:n_x, [j]] - result[0:n_x - 1, [j]]) * dt / (dx * 2)))
        result = np.hstack((result, u))
    return result[:, n_t - 1]


# Target
def obj(u, u_data, dx):
    return np.sum((u - u_data) ** 2) * dx


def penalty(g, n_x, dx, beta):
    return beta / dx * np.sum((g[1:n_x] - g[0:n_x - 1]) ** 2)


# Jacobian
def obj_g(n_x):
    """
    1 x n_x
    """
    return np.zeros((1, n_x))


def obj_u(u, u_data, n_t, n_x, dx):
    """
    1 x n_x * n_t
    """
    res = (u - u_data) * 2 * dx
    res = res.reshape((1, n_x))
    return np.hstack((np.zeros((1, n_x * (n_t - 1))), res))


def n_g(n_t, n_x):
    """
    n_x * n_t x n_x
    """
    res = np.zeros((n_x * n_t, n_x))
    for i in range(n_x):
        res[i, i] = -1
    return res


def n_u(n_t, dt, n_x, dx, alpha):
    """
    n_x * n_t x n_x * n_t
    """
    res = np.zeros((n_x * n_t, n_x * n_t))
    # initial condition
    for i in range(n_x):
        res[i, i] = 1

    for i in range(n_x, n_x * n_t):
        # boundary condition
        if i % n_x == n_x - 1:
            res[i, i - n_x + 1] = 1
        # interior advection
        else:
            res[i, i - n_x + 1] = -1 / dt + alpha / dx
            res[i, i - n_x] = - alpha / dx
            res[i, i + 1] = 1 / dt
    return res


def penalty_g(g, n_x, dx, beta):
    """
    n_x,
    """
    res = np.zeros(n_x)
    res[1:n_x] = 2 * beta / dx * (g[1:n_x] - g[0:n_x - 1])
    res[0:n_x - 1] = res[0:n_x - 1] - 2 * beta / dx * (g[1:n_x] - g[0:n_x - 1])
    return res


def grad(g, n_t, dt, n_x, dx, u_data, alpha, beta):
    """
    n_x,
    """
    u = forward(g, n_t, dt, n_x, dx)

    nu = n_u(n_t, dt, n_x, dx, alpha)
    obju = obj_u(u, u_data, n_t, n_x, dx)
    psi = np.linalg.solve(nu.T, obju.T)

    ng = n_g(n_t, n_x)
    objg = obj_g(n_x)
    obj_grad = objg - np.dot(psi.T, ng)
    return obj_grad.reshape(n_x) + penalty_g(g, n_x, dx, beta)
