"""Shared-frame math (Python mirror of ios/Sightline/SharedFrame.swift).

Matrices are 4x4, stored as lists of 4 *columns* (column-major, like simd),
so m[c][r] is column c, row r. Only used by the simulator and the tests;
the server itself only ever sees shared-frame coordinates.
"""

from __future__ import annotations

import math

Vec3 = tuple[float, float, float]
Mat4 = list[list[float]]  # 4 columns of 4 floats


def _norm(v: Vec3) -> Vec3:
    l = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if l < 1e-9:
        return (0.0, 0.0, 0.0)
    return (v[0] / l, v[1] / l, v[2] / l)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def marker_basis(m: Mat4) -> tuple[Vec3, Vec3]:
    """Origin p and horizontal forward f of the marker anchor transform M."""
    p = (m[3][0], m[3][1], m[3][2])
    n = (m[1][0], m[1][1], m[1][2])
    if math.hypot(n[0], n[2]) < 0.3:  # marker lying flat: use its in-plane z axis
        n = (m[2][0], m[2][1], m[2][2])
    f = _norm((n[0], 0.0, n[2]))
    return p, f


def shared_frame(p: Vec3, f: Vec3) -> Mat4:
    """S = [X Y Z p] with Y up, Z = f, X = Y x Z."""
    y = (0.0, 1.0, 0.0)
    z = _norm((f[0], 0.0, f[2]))
    x = _cross(y, z)
    return [[*x, 0.0], [*y, 0.0], [*z, 0.0], [p[0], p[1], p[2], 1.0]]


def rigid_inverse(s: Mat4) -> Mat4:
    """Inverse of a rotation+translation matrix."""
    r = [[s[c][r] for r in range(3)] for c in range(3)]  # r[c] = column c
    t = s[3][:3]
    # inverse rotation = transpose: column c of R^T is row c of R
    rt = [[r[0][c], r[1][c], r[2][c]] for c in range(3)]
    ti = [-(rt[0][i] * t[0] + rt[1][i] * t[1] + rt[2][i] * t[2]) for i in range(3)]
    return [[*rt[0], 0.0], [*rt[1], 0.0], [*rt[2], 0.0], [*ti, 1.0]]


def mat_mul(a: Mat4, b: Mat4) -> Mat4:
    return [[sum(a[k][r] * b[c][k] for k in range(4)) for r in range(4)] for c in range(4)]


def transform_point(m: Mat4, v: Vec3) -> Vec3:
    return tuple(m[0][r] * v[0] + m[1][r] * v[1] + m[2][r] * v[2] + m[3][r] for r in range(3))  # type: ignore[return-value]


def transform_dir(m: Mat4, v: Vec3) -> Vec3:
    return tuple(m[0][r] * v[0] + m[1][r] * v[1] + m[2][r] * v[2] for r in range(3))  # type: ignore[return-value]


def to_shared_from_marker(m: Mat4) -> Mat4:
    """T = inverse(S) for a marker anchor transform M."""
    p, f = marker_basis(m)
    return rigid_inverse(shared_frame(p, f))


def yaw_rotation(theta: float, t: Vec3 = (0.0, 0.0, 0.0)) -> Mat4:
    """Rotation about +Y by theta (right-handed), then translation t."""
    c, s = math.cos(theta), math.sin(theta)
    return [[c, 0.0, -s, 0.0], [0.0, 1.0, 0.0, 0.0], [s, 0.0, c, 0.0], [t[0], t[1], t[2], 1.0]]
