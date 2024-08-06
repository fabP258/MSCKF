import pytest
import numpy as np
from msckf.quaternion import Quaternion


def test_addition():
    q1 = Quaternion(np.array([2, 3, 4, 1]))
    q2 = Quaternion(np.array([-1, 0.5, -2, 0.5]))
    q3 = q1 + q2
    assert q3 == Quaternion(np.array([1, 3.5, 2, 1.5]))


def test_subtract():
    q1 = Quaternion(np.array([2, 3, 4, 1]))
    q2 = Quaternion(np.array([-1, 0.5, -2, 0.5]))
    q3 = q1 - q2
    assert q3 == Quaternion(np.array([3, 2.5, 6, 0.5]))


def test_normalize():
    q = Quaternion(np.array([2, 3, 4, 1])).normalize()
    assert np.isclose(np.linalg.norm(q.q), 1.0)
