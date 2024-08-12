import numpy as np


class Quaternion:

    def __init__(self, q: np.array):

        # quaternion representation: [x, y, z, w]
        # w: skalar/real part
        self.q = q

    def __repr__(self):
        return f"Quaternion(q={self.q})"

    def __getitem__(self, key):
        return self.q[key]

    def __eq__(self, other):
        return np.all(self.q == other.q)

    def __add__(self, other):
        return Quaternion(self.q + other.q)

    def __sub__(self, other):
        return Quaternion(self.q - other.q)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            q1 = self.normalize()
            q2 = other.normalize()

            L = np.array(
                [
                    [q1[3], q1[2], -q1[1], q1[0]],
                    [-q1[2], q1[3], q1[0], q1[1]],
                    [q1[1], -q1[0], q1[3], q1[2]],
                    [-q1[0], -q1[1], -q1[2], q1[3]],
                ]
            )

            q = L @ q2.q

            return Quaternion(q).normalize()
        else:
            return Quaternion(self.q * other)

    def conjugate(self):
        return Quaternion(np.array([*-self.q[:3], self.q[3]]))

    def normalize(self):
        return Quaternion(self.q / np.linalg.norm(self.q))

    def inv(self):
        return Quaternion(self.conjugate().q / np.linalg.norm(self.q) ** 2)

    def vec(self):
        return self.q[:3]

    def rotate_vector(self, vec: np.array):
        """Implements the quaternion rotation action x' = q * x * q*"""

        # represent the vector as pure quaternion
        x = Quaternion(q=np.array([*vec, 0]))

        # rotate the vector
        x_rotated = self * x * self.conjugate()

        # extract the vector
        return x_rotated.vec()

    @staticmethod
    def skew(vec: np.array):
        """Creates a skew-symmetrix matrix from a 3-element vector"""
        x, y, z = vec
        return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    @classmethod
    def from_rotvec(cls, rotvec: np.array):

        rotation_angle = np.linalg.norm(rotvec)

        if np.isclose(rotation_angle, 0.0):
            # initialize a quaternion representing no rotation
            return cls(q=np.array(0.0, 0.0, 0.0, 1.0))

        rotation_axis = rotvec / rotation_angle

        rotation_angle_half = rotation_angle / 2.0

        return cls(
            q=np.hstack(
                (
                    rotation_axis * np.sin(rotation_angle_half),
                    np.cos(rotation_angle_half),
                )
            )
        )

    def to_rotation_matrix(self):
        """Converts the quaternion into the corresponding rotation matrix"""

        q = self.normalize()
        vec = q[:3]
        w = q[3]

        R = (
            (2 * w * w - 1) * np.identity(3)
            - 2 * w * Quaternion.skew(vec)
            + 2 * vec[:, None] * vec
        )

        return R
