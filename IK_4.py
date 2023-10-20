# 4-link Inverse Kinematics
import numpy as np
from Link import Link
from gradient_descent import grad
from BFGS import BFGS
from IK_1 import f, g

# for root, t = (0, 0)
root = Link()
root.set_position(0, 0)
root.set_angle(0)
theta_0 = root.get_angle()
# for other links, t = (1, 0)

link1 = Link()
link1.set_position(1, 0)
theta_1 = link1.get_angle()

link2 = Link()
link2.set_position(1, 0)
theta_2 = link2.get_angle()

link3 = Link()
link3.set_position(1, 0)
theta_3 = link3.get_angle()

theta_vec = np.array([theta_0, theta_1, theta_2, theta_3])

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


matrices = [create_rotation_matrices(link) for link in [root, link1, link2, link3]]

R0, R0_prime, R0_double_prime = matrices[0]
R1, R1_prime, R1_double_prime = matrices[1]
R2, R2_prime, R2_double_prime = matrices[2]
R3, R3_prime, R3_double_prime = matrices[3]

# print these matrices to check that they are correct
print("R0 = ", R0)
print("R0_prime = ", R0_prime)
print("R0_double_prime = ", R0_double_prime)
print("R1 = ", R1)
print("R1_prime = ", R1_prime)
print("R1_double_prime = ", R1_double_prime)
print("R2 = ", R2)
print("R2_prime = ", R2_prime)
print("R2_double_prime = ", R2_double_prime)
print("R3 = ", R3)
print("R3_prime = ", R3_prime)
print("R3_double_prime = ", R3_double_prime)


r = np.array([1, 0, 1])
