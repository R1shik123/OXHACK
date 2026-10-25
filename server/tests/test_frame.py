import math
import random

from sightline.frame import (
    mat_mul,
    marker_basis,
    rigid_inverse,
    shared_frame,
    to_shared_from_marker,
    transform_dir,
    transform_point,
    yaw_rotation,
)
from sightline.geometry import segments_cross, wall_segment, walls_between


def close(a, b, tol=1e-6):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def wall_marker(yaw: float, pos):
    """A marker on a wall whose outward normal points along yaw (in shared terms: the marker frame itself).

    ARKit image anchor: image lies in its local x-z plane, y is the normal.
    Upright on a wall facing +Z: x = (1,0,0), y(normal) = (0,0,1), z = (0,-1,0).
    """
    upright = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    return mat_mul(yaw_rotation(yaw, pos), upright)


def test_marker_itself_maps_to_origin():
    m = wall_marker(0.7, (2.0, 1.3, -4.0))
    t = to_shared_from_marker(m)
    assert close(transform_point(t, (2.0, 1.3, -4.0)), (0, 0, 0))
    # A point 1 m out from the wall along its normal is +Z in the shared frame.
    n = (m[1][0], m[1][1], m[1][2])
    out = (2.0 + n[0], 1.3 + n[1], -4.0 + n[2])
    assert close(transform_point(t, out), (0, 0, 1))


def test_two_phones_agree():
    """Same physical point, two phones with different ARKit origins -> same shared coords."""
    rnd = random.Random(3)
    world_marker = wall_marker(0.0, (0, 1.4, 0))
    world_point = (3.0, 1.1, 5.0)
    results = []
    for _ in range(5):
        # phone's ARKit world = arbitrary yaw + offset of the physical world
        local_from_world = yaw_rotation(rnd.uniform(-math.pi, math.pi), (rnd.uniform(-9, 9), rnd.uniform(-1, 1), rnd.uniform(-9, 9)))
        m_local = mat_mul(local_from_world, world_marker)
        t = to_shared_from_marker(m_local)
        results.append(transform_point(t, transform_point(local_from_world, world_point)))
    for r in results:
        assert close(r, results[0], 1e-6)
    assert close(results[0], (3.0, -0.3, 5.0), 1e-6)


def test_tilted_marker_still_level():
    # marker taped slightly tilted back: normal has a y component, axes stay gravity-aligned
    tilt = [[1, 0, 0, 0], [0, 0.3, 0.95, 0], [0, -0.95, 0.3, 0], [0, 0, 0, 1]]
    t = to_shared_from_marker(tilt)
    assert close(transform_dir(t, (0, 1, 0)), (0, 1, 0))


def test_flat_marker_uses_column2():
    flat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0.5, 0, 0.5, 1]]  # lying on floor, normal up
    p, f = marker_basis(flat)
    assert close(f, (0, 0, 1))
    assert close(p, (0.5, 0, 0.5))


def test_inverse():
    s = shared_frame((1, 2, 3), (0.6, 0, 0.8))
    i = mat_mul(rigid_inverse(s), s)
    for c in range(4):
        assert close(i[c], [1 if r == c else 0 for r in range(4)])


def test_walls_between():
    wall = wall_segment([[0, 0, -2], [0, 0, 2], [0, 2.5, 2], [0, 2.5, -2]])
    assert wall is not None
    assert walls_between((-1, 0), (1, 0), 1.2, [wall]) == 1
    assert walls_between((-1, 3), (1, 3), 1.2, [wall]) == 0
    assert walls_between((-1, 0), (1, 0), 5.0, [wall]) == 0  # different floor
    assert not segments_cross((0, 0), (1, 0), (1, 0), (2, 0))
    # two phones' slightly offset copies of one wall count once; a second wall counts again
    copy = wall_segment([[0.05, 0, -2], [0.05, 0, 2], [0.05, 2.5, 2], [0.05, 2.5, -2]])
    other = wall_segment([[3, 0, -2], [3, 0, 2], [3, 2.5, 2], [3, 2.5, -2]])
    assert walls_between((-1, 0), (4, 0), 1.2, [wall, copy, other]) == 2
