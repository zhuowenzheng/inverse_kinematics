import numpy as np
from Link import Link
from gradient_descent import grad
from BFGS import BFGS

wtar = 1e3
wreg = 1e0
ptar = np.array([0, 1, 1])

link = Link()
theta = link.get_angle()
link.set_position(0, 0)

def f(x, link = link):
    theta = x[0]
    T = np.eye(3)
    T[0, 2] = link.get_position()[0]
    T[1, 2] = link.get_position()[1]

    R = np.eye(3)
    R[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
    r = np.array([1, 0, 1])
    p = T @ R @ r
    delta_p = p - ptar
    return np.array([0.5 * wtar * delta_p @ delta_p + 0.5 * wreg * pow(theta, 2)])

def g(x, link = link):
    theta = x[0]

    T = np.eye(3)
    T[0, 2] = link.get_position()[0]
    T[1, 2] = link.get_position()[1]

    R = np.eye(3)
    R[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

    R_prime = np.eye(3)
    R_prime[0:2, 0:2] = np.array([[-np.sin(theta), -np.cos(theta)],
                                  [np.cos(theta), -np.sin(theta)]])
    R_prime[2, 2] = 0
    r = np.array([1, 0, 1])
    p = T @ R @ r  # Assuming you meant to use R_prime here based on previous code
    p_prime = T @ R_prime @ r
    return np.array([wtar * (p - ptar) @ p_prime + wreg * theta])


if __name__ == "__main__":
    epsilon = 1e-6
    alpha = 1.0
    gamma = 0.5
    x = np.array([theta])

    result = BFGS(x, alpha, gamma, epsilon, f, g, iter_max=5)

    # wrapping
    while result > np.pi:
        result -= 2 * np.pi
    while result < -np.pi:
        result += 2 * np.pi

    print("x = ", np.round(result, 6))
