import numpy as np
from rosenbrock import f_objective, gradients, hessian

def grad(x0, x1 ,alpha,epsilon,iter_max = 50):
    for iter in range(iter_max):
        # evaluate f and g at x = (x0,x1)^T
        f = f_objective(x0, x1)
        g = gradients(x0, x1)
        p = -g
        # step
        delta_x = alpha * p
        # update x
        x0 = x0 + delta_x[0]
        x1 = x1 + delta_x[1]
        print("iter = ", iter, "f = ", f, "x = ", x0, x1)
        # check convergence
        if np.linalg.norm(g) < epsilon:
            print("converged at ", iter)
            break
    # precison 6 digits behind the decimal point
    print("x = ", round(x0, 6), round(x1, 6))

if __name__ == "__main__":
    epsilon = 1e-6
    alpha = 0.1
    x0 = -1
    x1 = 0
    grad(x0, x1, alpha, epsilon)
