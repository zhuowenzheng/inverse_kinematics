import sys

import numpy as np

a = 1
b = 1


def f_objective(x):
    return pow((a - x[0]), 2) + b * pow((x[1] - pow(x[0], 2)), 2)


def gradients(x):
    return 2 * (np.array([x[0] - a + 2 * b * x[0] * (pow(x[0], 2) - x[1]), b * (x[1] - pow(x[0], 2))]))


def hessian(x):
    return np.array([[2 + 12 * b * pow(x[0], 2) - 4 * b * x[1], -4 * b * x[0]], [-4 * b * x[0], 2 * b]])


if __name__ == "__main__":
    x = np.array([-1, 0])

    f_obj = f_objective(x)
    grads = gradients(x)
    hess = hessian(x)

    # Save the output to resources/outputA1.txt
    resource_dir = sys.argv[1]
    file_path = resource_dir + '/outputA1.txt'

    with open(file_path, 'w') as file:
        # Write the objective function value
        file.write(f"{f_obj}\n")
        # Write the gradients
        file.write(f"{grads[0]}\n{grads[1]}\n")
        # Write the Hessian matrix
        file.write(f"{hess[0, 0]}  {hess[0, 1]}\n{hess[1, 0]}  {hess[1, 1]}\n")
