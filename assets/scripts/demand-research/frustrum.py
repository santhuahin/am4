# mypy: ignore-errors

"""
   011 >-------< 111
      /|      /|
     / |     /          far
001 >-------< 101
    |  |    |
       |    |  |
   010 >----|--< 110
      /     | /
    |/      |/          near
000 >-------< 100
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TransformData:
    p_source: NDArray
    N: NDArray
    d: float
    k: float
    p001: NDArray
    p101: NDArray
    p011: NDArray
    p111: NDArray


def create_transform_data(
    p001: NDArray, p101: NDArray, p011: NDArray, k: float, p111: NDArray | None = None
) -> TransformData:
    if abs(1.0 - k) < 1e-9:
        raise ValueError("k cannot be 1.0")

    if p111 is None:
        p111 = p101 + p011 - p001

    p_far = np.array([p001, p101, p011, p111])
    p_near = k * p_far
    p_source = np.array([p_near[0], p_near[1], p_near[2], p_near[3], p_far[0], p_far[1], p_far[2], p_far[3]])

    v1 = p101 - p001
    v2 = p011 - p001
    N = np.cross(v1, v2)

    norm_N = np.linalg.norm(N)
    if norm_N < 1e-9:
        raise ValueError("far plane corners are collinear.")

    N /= norm_N
    d = np.dot(N, p001)

    if abs(d) < 1e-9:
        raise ValueError("far plane passes through the origin.")

    return TransformData(p_source, N, d, k, p001, p101, p011, p111)


def forward_map(uvw: NDArray, data: TransformData) -> np.ndarray:
    u, v, w = uvw
    p = data.p_source
    P000, P100, P010, P110, P001, P101, P011, P111 = p

    c00 = P000 * (1 - u) + P100 * u
    c10 = P010 * (1 - u) + P110 * u
    c01 = P001 * (1 - u) + P101 * u
    c11 = P011 * (1 - u) + P111 * u

    c0 = c00 * (1 - v) + c10 * v
    c1 = c01 * (1 - v) + c11 * v

    return c0 * (1 - w) + c1 * w


def inverse_map(xyz: NDArray, data: TransformData) -> np.ndarray:
    s = np.dot(data.N, xyz) / data.d

    if abs(s) < 1e-9:
        w = -data.k / (1.0 - data.k)
        return np.array([0.5, 0.5, w])

    w = (s - data.k) / (1.0 - data.k)

    p_far_uv = xyz / s

    y001, y011 = data.p001[1], data.p011[1]
    v = (p_far_uv[1] - y001) / (y011 - y001)

    x_left = (1 - v) * data.p001[0] + v * data.p011[0]
    x_right = (1 - v) * data.p101[0] + v * data.p111[0]
    u = (p_far_uv[0] - x_left) / (x_right - x_left)

    return np.array([u, v, w])


if __name__ == "__main__":
    p001 = np.array([96.0, 180.0, 48.0])
    p101 = np.array([204.0, 180.0, 12.0])
    p011 = np.array([204.0, 45.0, 102.0])
    k = 0.8

    transform_data = create_transform_data(p001, p101, p011, k)

    print(f"computed p111: {np.round(transform_data.p111, 5)}")
    import matplotlib.pyplot as plt

    far_plane = np.vstack(
        [transform_data.p001, transform_data.p101, transform_data.p111, transform_data.p011, transform_data.p001]
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(far_plane[:, 0], far_plane[:, 1], far_plane[:, 2], "b-")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=120)
    plt.show()

    uvw_corners = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0.5, 0.5, 0.5]]
    )

    all_corners_ok = True
    for i, uvw_target in enumerate(uvw_corners):
        xyz_corner = forward_map(uvw_target, transform_data)
        uvw_calc = inverse_map(xyz_corner, transform_data)
        if not np.allclose(uvw_target, uvw_calc, atol=1e-9):
            all_corners_ok = False
            print(f"mismatch at corner {i}: target={uvw_target}, calc={np.round(uvw_calc, 5)}")

    if all_corners_ok:
        print("success")
