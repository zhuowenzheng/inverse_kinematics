import numpy as np
from rosenbrock import f_objective, gradients

def grad(x, alpha, epsilon, f_obj, grad, iter_max=50):
    n = len(x)  # Get the dimension of the vector x
    for iter in range(iter_max):
        # Evaluate the objective function and its gradient at the current point
        f = f_obj(x)
        g = grad(x)
        p = -g
        # Step
        delta_x = alpha * p
        # Update x
        x += delta_x

        print("iter = ", iter, "f = ", f, "g =", g, "x = ", x)
        # Check convergence
        if np.linalg.norm(g) < epsilon:
            print("converged at ", iter)
            break
    # Precision 6 digits behind the decimal point
    print("x = ", np.round(x, 6))
    return x
if __name__ == "__main__":
    epsilon = 1e-6
    alpha = 0.1
    init_x = np.array([-1.0, 0.0])
    grad(init_x, alpha, epsilon, f_objective, gradients)
