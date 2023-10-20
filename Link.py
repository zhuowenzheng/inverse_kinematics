import numpy as np
from scipy.spatial.transform import Rotation as R

class Link:
    def __init__(self, position=(1, 0)):
        self.position = np.array([0, 0])
        self.set_position(position[0], position[1])
        self.parent = None
        self.child = None
        self.depth = 0
        self.angle = 0.0
        self.angle_rest = 0.0
        self.mesh_mat = np.eye(4)

    def add_child(self, child):
        child.parent = self
        child.depth = self.depth + 1
        self.child = child

    def get_child(self):
        return self.child

    def get_parent(self):
        return self.parent

    def get_depth(self):
        return self.depth

    def set_position(self, x, y):
        self.position = np.array([x, y])

    def get_position(self):
        return self.position

    def set_angle(self, a):
        self.angle = a

    def get_angle(self):
        return self.angle

    def set_angle_rest(self, a):
        self.angle_rest = a

    def get_angle_rest(self):
        return self.angle_rest

    def set_mesh_matrix(self, M):
        self.mesh_mat = M
