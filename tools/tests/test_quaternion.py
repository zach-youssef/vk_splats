import numpy as np
import pytest

from splatdata import quaternion


def random_unit_quaternions(n, seed=0):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_round_trip_through_matrix():
    q = random_unit_quaternions(1000)
    back = quaternion.from_matrix(quaternion.to_matrix(q))
    # q and −q are the same rotation; from_matrix picks w >= 0
    same_sign = np.where(q[:, :1] < 0, -q, q)
    np.testing.assert_allclose(back, same_sign, atol=1e-12)


def test_matrices_are_rotations():
    r = quaternion.to_matrix(random_unit_quaternions(100, seed=1))
    np.testing.assert_allclose(r @ np.swapaxes(r, 1, 2), np.broadcast_to(np.eye(3), r.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(r), 1, atol=1e-12)


def test_w_first():
    # 90 degrees about +z: (w, x, y, z) = (cos 45°, 0, 0, sin 45°) takes +x to +y
    s = np.sqrt(0.5)
    r = quaternion.to_matrix([s, 0, 0, s])
    np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(quaternion.to_matrix([1, 0, 0, 0]), np.eye(3))


@pytest.mark.parametrize("axis", range(3))
def test_half_turns(axis):
    # w = 0: the case where the trace-based formula alone divides by zero
    q = np.zeros(4)
    q[1 + axis] = 1
    np.testing.assert_allclose(quaternion.from_matrix(quaternion.to_matrix(q)), q, atol=1e-15)


def test_normalizes_and_rejects_zero():
    np.testing.assert_allclose(quaternion.to_matrix([2, 0, 0, 0]), np.eye(3))
    with pytest.raises(ValueError):
        quaternion.to_matrix([0, 0, 0, 0])
