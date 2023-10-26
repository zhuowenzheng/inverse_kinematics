import numpy as np
from rosenbrock import f_objective, gradients

import sys

def grad(x, alpha, epsilon, f_obj, grad, iter_max=50):
    global iter
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
        # print("iter = ", iter, "f = ", f, "g =", g, "x = ", x)
        # Check convergence
        if np.linalg.norm(g) < epsilon:
            print("Gradient descent converged at ", iter + 1)
            break
    # Precision 6 digits behind the decimal point

    return x, iter + 1
if __name__ == "__main__":
    epsilon = 1e-6
    alpha = 0.1
    init_x = np.array([-1.0, 0.0])
    x, iter_ = grad(init_x, alpha, epsilon, f_objective, gradients)

    x = np.round(x, 6)
    print("x = ", x)

    # Save the output to resources/outputA2.txt
    resource_dir = sys.argv[1]
    file_path = resource_dir + '/outputA2.txt'

    with open(file_path, 'w') as file:
        # Iterate through the array and write each element to the file
        file.write(f'{iter_}\n')
        for value in x:
            file.write(f'{value}\n')