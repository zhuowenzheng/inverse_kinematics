import numpy as np
from rosenbrock import f_objective, gradients

import sys

def BFGS(x, alpha_0, gamma, epsilon, f_obj, grad, iter_max=50, iter_maxLS=20):
    global iter
    n = len(x)  # Get the dimension of the vector x

    A = np.eye(n)  # Initialize the Hessian approximation matrix to the identity matrix
    x_0 = np.zeros(x.shape)
    g_0 = np.zeros(x.shape)

    for iter in range(iter_max):
        # Evaluate the objective function and its gradient at the current point
        f = f_obj(x)
        g = grad(x)

        if iter > 0:
            s = x - x_0  # Compute the change in x
            y = g - g_0  # Compute the change in the gradient
            rho = 1 / np.dot(y, s)  # Compute the scaling factor rho
            A = (np.eye(n) - rho * np.outer(s, y)) @ A @ (np.eye(n) - rho * np.outer(y, s)) + rho * np.outer(s, s)  # Update the Hessian approximation matrix

        # print("A = ", A)
        p = np.dot(-A, g)  # Compute the search direction
        # print("p = ", p)
        # Line search
        a = alpha_0
        delta_x = np.zeros(x.shape)
        for iterLS in range(iter_maxLS):
            delta_x = a * p
            # print("delta_x = ", delta_x)
            # print("x + delta_x = ", x + delta_x)
            f1 = f_obj(x + delta_x)
            # print("f1 = ", f1)
            if f1 < f:
                break
            else:
                a *= gamma  # Reduce the step size
        # print("alpha = ", a)
        # Update the current point and the previous values of x and g
        x_0 = x.copy()
        g_0 = g.copy()
        x += delta_x  # Take a step along the search direction

        # print("iter = ", iter, "f = ", f, "g =", g, "x = ", x)
        if np.linalg.norm(g) < epsilon:
            print("BFGS converged at ", iter + 1)
            break

    # Print the final solution to a precision of 6 digits behind the decimal point
    return x, iter + 1

if __name__ == "__main__":

    epsilon = 1e-6
    alpha = 1
    gamma = 0.8
    init_x = np.array([-1.0, 0.0])
    x, iter_ = BFGS(init_x, alpha, gamma, epsilon, f_objective, gradients)
    x = np.round(x, 6)
    print("x = ", x)

    # Save the output to resources/outputA5.txt
    resource_dir = sys.argv[1]
    file_path = resource_dir + '/outputA5.txt'

    with open(file_path, 'w') as file:
        # Iterate through the array and write each element to the file
        file.write(f'{iter_}\n')
        for value in x:
            file.write(f'{value}\n')