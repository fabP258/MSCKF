import numpy as np
from msckf.quaternion import Quaternion


class ImuState:

    def __init__(
        self,
        q: Quaternion = None,
        b_g: np.array = None,
        v: np.array = None,
        b_a: np.array = None,
        p: np.array = None,
    ):

        # unit quaternion describing rotation from frame {G} (global ref. frame) to {I} (IMU fixed frame)
        self.q = q if q else Quaternion(np.array([0.0, 0.0, 0.0, 1.0]))

        # gyroscope bias (3 x 1 vector)
        self.b_g = b_g if b_g else np.zeros(3)

        # velocity of IMU fixed frame {I} w.r.t. {G}
        self.v = v if v else np.zeros(3)

        # accelerometer bias (3 x 1 vector)
        self.b_a = b_a if b_a else np.zeros(3)

        # position of IMU fixed frame {I} w.r.t. {G}
        self.p = p if p else np.zeros(3)
