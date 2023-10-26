# n-link with Newton
import numpy as np
from Link import Link
from gradient_descent import grad
from BFGS import BFGS
from newtons_method import newtons

import matplotlib.pyplot as plt
from matplotlib import collections as mc


def translation_matrix(link):
    T = np.eye(3)
    T[0, 2] = link.get_position()[0]
    T[1, 2] = link.get_position()[1]
    return T


def create_rotation_matrices(link):
    theta = link.get_angle()
    R = np.eye(3)
    R_prime = np.eye(3)
    R_double_prime = np.eye(3)

    # Populate R with rotation matrix values
    R[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

    # Populate R_prime with the derivative of the rotation matrix values
    R_prime[0:2, 0:2] = np.array([[-np.sin(theta), -np.cos(theta)],
                                  [np.cos(theta), -np.sin(theta)]])
    R_prime[2, 2] = 0  # Set the (3,3) element to 0

    # Populate R_double_prime with the second derivative of the rotation matrix values
    R_double_prime[0:2, 0:2] = np.array([[-np.cos(theta), np.sin(theta)],
                                         [-np.sin(theta), -np.cos(theta)]])
    R_double_prime[2, 2] = 0  # Set the (3,3) element to 0

    return R, R_prime, R_double_prime

def compute_p(T_matrices, R_matrices, r, n):
    p_result = np.dot(T_matrices[0], R_matrices[0][0])
    for i in range(1, n):
        p_result = np.dot(p_result, T_matrices[i])
        p_result = np.dot(p_result, R_matrices[i][0])
    return np.dot(p_result, r).T

# Compute p_prime for each joint
def compute_p_prime(T_matrices, R_matrices, r, index, n):
    product = T_matrices[0]
    for i in range(n):
        if i == index:
            product = np.dot(product, R_matrices[i][1])  # Use the derivative of R
        else:
            product = np.dot(product, R_matrices[i][0])
        product = np.dot(product, T_matrices[i + 1] if i < n - 1 else np.eye(3))
    return np.dot(product, r)

def create_P_prime(T_matrices, R_matrices, r, n):
    P_list = [compute_p_prime(T_matrices, R_matrices, r, i, n) for i in range(n)]
    P_prime_matrix = np.array([p[:2] for p in P_list]).T
    return P_prime_matrix

def compute_entry(T_matrices, R_matrices, r, n, i, j):
    product = T_matrices[0]
    R_list = [R_matrices[k][0] for k in range(n)]

    # Update the rotation matrix at the i-th and j-th positions
    if i == j:
        R_list[i] = R_matrices[i][2]
    else:
        R_list[i] = R_matrices[i][1]
        R_list[j] = R_matrices[j][1]

    # Compute the resultant matrix using the matrices in R_list
    for k in range(n):
        product = np.dot(product, R_list[k])
        if k < n - 1:
            product = np.dot(product, T_matrices[k + 1])

    result = np.dot(product, r)[:2].T
    return np.array([result[0], result[1]])

def create_P_double_prime(T_matrices, R_matrices, r, n):
    P_double_prime_matrix = np.zeros((n, n, 2))
    for i in range(n):
        for j in range(n):
            P_double_prime_matrix[i, j] = compute_entry(T_matrices, R_matrices, r, n, i, j)
    return P_double_prime_matrix

def f(x):
    # Extract theta values
    theta_vec = x[:n]

    # Update angles of links
    for i in range(len(links)):
        links[i].set_angle(theta_vec[i])

    # Update T and R matrices
    T_matrices = [translation_matrix(link) for link in links]
    R_matrices = [create_rotation_matrices(link) for link in links]

    # Compute p, P_prime, and P_double_prime
    p = np.dot(T_matrices[0], R_matrices[0][0])
    for i in range(1, n):
        p = np.dot(p, T_matrices[i])
        p = np.dot(p, R_matrices[i][0])
    p = np.dot(p, r).T
    delta_p = (p - ptar)[:2]

    # Calculate objective
    Wreg_matrix = np.diag(Wreg)  # Convert Wreg to a diagonal matrix
    return 0.5 * wtar * delta_p @ delta_p + 0.5 * (theta_vec @ Wreg_matrix @ theta_vec)

def g(x):
    # Extract theta values
    theta_vec = x[:n]

    # Update angles of links
    for i in range(len(links)):
        links[i].set_angle(theta_vec[i])

    # Update T and R matrices
    T_matrices = [translation_matrix(link) for link in links]
    R_matrices = [create_rotation_matrices(link) for link in links]

    # Compute p and P_prime
    p = np.dot(T_matrices[0], R_matrices[0][0])
    for i in range(1, n):
        p = np.dot(p, T_matrices[i])
        p = np.dot(p, R_matrices[i][0])
    p = np.dot(p, r).T
    P_prime = create_P_prime(T_matrices, R_matrices, r, n)
    delta_p = (p - ptar)[:2]

    # Calculate gradient
    Wreg_matrix = np.diag(Wreg)  # Convert Wreg to a diagonal matrix
    return wtar * delta_p @ P_prime + Wreg_matrix @ theta_vec

# Hessian
def h(x):
    # Extract theta values
    theta_vec = x[:n]

    # Update angles of links
    for i in range(len(links)):
        links[i].set_angle(theta_vec[i])

    # Update T and R matrices
    T_matrices = [translation_matrix(link) for link in links]
    R_matrices = [create_rotation_matrices(link) for link in links]

    # Compute p, P_prime, and P_double_prime
    p = compute_p(T_matrices, R_matrices, r, n)
    P_prime = create_P_prime(T_matrices, R_matrices, r, n)

    P_double_prime = create_P_double_prime(T_matrices, R_matrices, r, n)

    print("P'' =\n", P_double_prime)
    print("P' =\n", P_prime)
    delta_p = (p - ptar)[:2]
    print("delta_p =\n", delta_p)
    # Construct Hessian
    term1 = P_prime.T @ P_prime
    term2 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            term2[i, j] = delta_p @ P_double_prime[i, j]

    Wreg_matrix = np.diag(Wreg)  # Convert Wreg to a diagonal matrix
    print("term1 =\n", term1)
    print("term2 =\n", term2)
    print("Wreg_matrix =\n", Wreg_matrix)

    Hessian = wtar * (term1 + term2) + Wreg_matrix

    return Hessian


if __name__ == "__main__":
    # 1. Initialize parameters
    n = 10  # Number of links
    r = np.array([1, 0, 1])
    ptar = np.array([9, 1, 1])  # Target position
    wtar = 1e3  # Weight for target position
    epsilon = 1e-6
    alpha = 1.0
    gamma = 0.5

    # 2. Initialize the root link
    links = [Link((0, 0))]
    for _ in range(n - 1):
        link = Link((1, 0))
        links.append(link)

    # 3. Initialize Wreg
    Wreg = np.array([0] + [1e0] * (n - 1))

    # 4. Other Initializations
    theta_vec = np.array([link.get_angle() for link in links])
    T_matrices = [translation_matrix(link) for link in links]
    R_matrices = [create_rotation_matrices(link) for link in links]

    # 5. Compute P, P_prime, and P_double_prime
    p = compute_p(T_matrices, R_matrices, r, n)
    P_prime = create_P_prime(T_matrices, R_matrices, r, n)
    P_double_prime = create_P_double_prime(T_matrices, R_matrices, r, n)

    # 6. BFGS optimization & wrapping
    theta_bfgs = BFGS(theta_vec, alpha, gamma, epsilon, f, g, iter_max=2)
    result = newtons(theta_bfgs, alpha, gamma, epsilon, f, g, h, iter_max=30)
    while np.any(result > np.pi):
        result[result > np.pi] -= 2 * np.pi

    while np.any(result < -np.pi):
        result[result < -np.pi] += 2 * np.pi

    print("x = ", np.round(result, 6))

    # 7. Save the output to