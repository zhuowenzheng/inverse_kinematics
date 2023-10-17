import numpy as np
from rosenbrock import f_objective, gradients, hessian


def BFGS(x0, x1, alpha_0, gamma, epsilon, iter_max=50, iter_maxLS=20):
    A = np.eye(2)
    x_0 = np.array([0, 0])
    g_0 = np.array([0, 0])
    for iter in range(iter_max):
        # evaluate f and g at x = (x0,x1)^T
        f = f_objective(x0, x1)
        g = gradients(x0, x1)
        if iter > 0:
            s = np.array([x0 - x_0[0], x1 - x_0[1]])
            y = np.array([g[0] - g_0[0], g[1] - g_0[1]])
            rho = 1 / (y[0] * s[0] + y[1] * s[1])
            A = (np.eye(2) - rho * np.outer(s, y)) @ A @ (np.eye(2) - rho * np.outer(y, s)) + rho * np.outer(s, s)
        p = -A @ g
        # line search
        a = alpha_0
        delta_x = np.zeros(2)
        for iterLS in range(iter_maxLS):
            delta_x = a * p
            f1 = f_objective(x0 + delta_x[0], x1 + delta_x[1])
            if f1 < f:
                break
            else:
                a *= gamma
        # step
        x_0 = np.array([x0, x1])
        g_0 = np.array([g[0], g[1]])
        x0 += delta_x[0]
        x1 += delta_x[1]
        print("iter = ", iter, "f = ", f, "x = ", x0, x1)
        if np.linalg.norm(g) < epsilon:
            print("converged at ", iter)
            break
    # precison 6 digits behind the decimal point
    print("x = ", round(x0, 6), round(x1, 6))


if __name__ == "__main__":
    epsilon = 1e-6
    alpha = 1
    gamma = 0.8
    x0 = -1
    x1 = 0
    BFGS(x0, x1, alpha, gamma, epsilon)
