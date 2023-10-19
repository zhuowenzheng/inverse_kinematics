import numpy as np
from rosenbrock import f_objective, gradients

def line_search(x, alpha_0, gamma, epsilon, f_obj, grad, iter_max=50, iter_max_LS=20):
    n = len(x)  # Get the dimension of the vector x

    for iter in range(iter_max):
        # Evaluate the objective function and its gradient at the current point
        f = f_obj(x)
        g = grad(x)
        p = -g

        # Line search
        a = alpha_0
        delta_x = np.zeros(n)
        for i in range(iter_max_LS):
            delta_x = a * p
            f1 = f_obj(x + delta_x)
            if f1 < f:
                break
            else:
                a *= gamma  # Reduce the step size

        # Step
        x += delta_x
        if np.linalg.norm(g) < epsilon:
            print("converged at ", iter)
            break

    # Precision 6 digits behind the decimal point
    print("x = ", np.round(x, 6))

if __name__ == "__main__":
    epsilon = 1e-6
    alpha = 1
    gamma = 0.8
    init_x = np.array([-1.0, 0.0])
    line_search(init_x, alpha, gamma, epsilon, f_objective, gradients)
