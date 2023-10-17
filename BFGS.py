import numpy as np
from rosenbrock import f_objective, gradients, hessian
def BFGS(x0, x1, alpha_0, gamma, epsilon, iter_max = 50, iter_maxLS = 20):
    A = np.eye(2)
    x_0 = np.array(0, 0)
    g_0 = np.array(0, 0)
    for iter in range(iter_max):
        # evaluate f and g at x = (x0,x1)^T
        f = f_objective(x0, x1)
        g = gradients(x0, x1)
        if iter > 1:
            s = np.array([x0 - x_0[0], x1 - x_0[1]])
            y = np.array([g[0] - g_0[0], g[1] - g_0[1]])
            rho = 1 / (y[0] * s[0] + y[1] * s[1])
