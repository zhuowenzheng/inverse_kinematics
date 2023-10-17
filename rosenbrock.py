import numpy as np

a = 1
b = 1


def f_objective(x0, x1):
    return pow((a - x0), 2) + b * pow((x1 - pow(x0, 2)), 2)


def gradients(x0, x1):
    return 2 * (np.array([x0 - a + 2 * b * x0 * (pow(x0, 2) - x1), b * (x1 - pow(x0, 2))]))


def hessian(x0, x1):
    return np.array([[2 + 12 * b * pow(x0, 2) - 4 * b * x1, -4 * b * x0], [-4 * b * x0, 2 * b]])


if __name__ == "__main__":
    x0 = -1
    x1 = 0

    print(f_objective(x0, x1))
    print(gradients(x0, x1))
    print(hessian(x0, x1))
