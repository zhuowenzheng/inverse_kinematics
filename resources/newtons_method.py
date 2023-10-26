import sys

import numpy as np
from rosenbrock import f_objective, gradients, hessian

def newtons(x, alpha_0, gamma, epsilon, f_obj, grad, hess, iter_max=50, iter_maxLS=20):
    global iter
    n = len(x)  # Get the dimension of the vector x

    for iter in range(iter_max):
        # evaluate f, g, and H at x
        f = f_obj(x)
        g = grad(x)
        H = hess(x)
        p = -np.linalg.solve(H, g)
        # line search
        a = alpha_0
        delta_x = np.zeros(n)
        for iterLS in range(iter_maxLS):
            delta_x = a * p
            f1 = f_obj(x + delta_x)
            if f1 < f:
                break
            else:
                a *= gamma
        # step
        x += delta_x
        # print("iter = ", iter, "f = ", f, "x = ", x)
        # check convergence
        if np.linalg.norm(g) < epsilon:
            print("NM converged at ", iter + 1)
            break
    return x, iter + 1

if __name__ == "__main__":
    epsilon = 1e-6
    alpha = 1
    gamma = 0.8
    init_x = np.array([-1.0, 0.0])
    result, iter_ = newtons(init_x, alpha, gamma, epsilon, f_objective, gradients, hessian)
    result = np.round(result, 6)
    print("x = ", result)

    # Save the output to resources/outputA4.txt
    resource_dir = sys.argv[1]
    file_path = resource_dir + '/outputA4.txt'

    with open(file_path, 'w') as file:
        # Iterate through the array and write each element to the file
        file.write(f'{iter_}\n')
        for value in result:
            file.write(f'{value}\n')