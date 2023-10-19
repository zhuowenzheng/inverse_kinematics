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

    print(f_objective(x))
    print(gradients(x))
    print(hessian(x))
