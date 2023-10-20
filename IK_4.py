# 4-link Inverse Kinematics
import numpy as np
from Link import Link
from gradient_descent import grad
from BFGS import BFGS
from IK_1 import f, g

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


n = 4

# Initialize the root link
links = [Link((0, 0))]

# Create the rest of the links
for _ in range(n - 1):
    link = Link((1, 0))
    link.set_angle(0)
    links.append(link)

print("len of links = ", len(links))
print(links[0].get_position())
print(links[1].get_position())
# Gather all angles
theta_vec = np.array([link.get_angle() for link in links])

# Create translation and rotation matrices
T_matrices = [translation_matrix(link) for link in links]
R_matrices = [create_rotation_matrices(link) for link in links]

r = np.array([1, 0, 1])

# p = T0R0(theta)T1R1(theta)T2R2(theta)T3R3(theta)r
p = np.dot(T_matrices[0], R_matrices[0][0])
for i in range(1, 4):
    p = np.dot(p, T_matrices[i])
    p = np.dot(p, R_matrices[i][0])
p = np.dot(p, r)
print("p = ", p)

p_prime_matrix = np.zeros((4, 3))

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

n = len(R_matrices)  # Assuming the length of R_matrices corresponds to the number of links.
P_prime = create_P_prime(T_matrices, R_matrices, r, n)


print("P' =\n", P_prime)

