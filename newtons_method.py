import numpy as np
from rosenbrock import f_objective, gradients, hessian

def newtons(x0, x1, alpha_0, gamma, epsilon, iter_max = 50, iter_maxLS = 20):
    for iter in range(iter_max):
        f = f_objective(x0, x1)
        g = gradients(x0, x1)
        H = hessian(x0, x1)
        p = -np.linalg.solve(H, g)
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
        x0 += delta_x[0]
        x1 += delta_x[1]
        print("iter = ", iter, "f = ", f, "x = ", x0, x1)
# check convergence
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
    newtons(x0, x1, alpha, gamma, epsilon)
