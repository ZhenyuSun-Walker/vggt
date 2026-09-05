# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from typing import Union


ArrayLike = Union[np.ndarray, torch.Tensor]


def _ensure_torch(value: ArrayLike) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)


def single_undistortion(params, tracks_normalized):
    """Apply the upstream one-step distortion mapping for [B, N, 2] tracks."""
    params = _ensure_torch(params)
    tracks_normalized = _ensure_torch(tracks_normalized)
    u, v = tracks_normalized[..., 0].clone(), tracks_normalized[..., 1].clone()
    u_undistorted, v_undistorted = apply_distortion(params, u, v)
    return torch.stack((u_undistorted, v_undistorted), dim=-1)


def iterative_undistortion(
    params, tracks_normalized, max_iterations=100, max_step_norm=1e-10, rel_step_size=1e-6
):
    """Numerically invert distortion, matching upstream VGGT's solver."""
    params = _ensure_torch(params)
    tracks_normalized = _ensure_torch(tracks_normalized)
    u, v = tracks_normalized[..., 0].clone(), tracks_normalized[..., 1].clone()
    original_u, original_v = u.clone(), v.clone()
    eps = torch.finfo(u.dtype).eps
    for _ in range(max_iterations):
        u_distorted, v_distorted = apply_distortion(params, u, v)
        dx, dy = original_u - u_distorted, original_v - v_distorted
        step_u = torch.clamp(torch.abs(u) * rel_step_size, min=eps)
        step_v = torch.clamp(torch.abs(v) * rel_step_size, min=eps)
        j00 = (apply_distortion(params, u + step_u, v)[0] - apply_distortion(params, u - step_u, v)[0]) / (2 * step_u)
        j01 = (apply_distortion(params, u, v + step_v)[0] - apply_distortion(params, u, v - step_v)[0]) / (2 * step_v)
        j10 = (apply_distortion(params, u + step_u, v)[1] - apply_distortion(params, u - step_u, v)[1]) / (2 * step_u)
        j11 = (apply_distortion(params, u, v + step_v)[1] - apply_distortion(params, u, v - step_v)[1]) / (2 * step_v)
        jacobian = torch.stack(
            (torch.stack((j00 + 1, j01), dim=-1), torch.stack((j10, j11 + 1), dim=-1)), dim=-2
        )
        delta = torch.linalg.solve(jacobian, torch.stack((dx, dy), dim=-1))
        u, v = u + delta[..., 0], v + delta[..., 1]
        if torch.max((delta ** 2).sum(dim=-1)) < max_step_norm:
            break
    return torch.stack((u, v), dim=-1)


def apply_distortion(extra_params, u, v):
    """Apply SIMPLE_RADIAL, RADIAL, or OpenCV camera distortion."""
    extra_params = _ensure_torch(extra_params)
    u, v = _ensure_torch(u), _ensure_torch(v)
    num_params = extra_params.shape[1]
    u2, v2, uv = u * u, v * v, u * v
    r2 = u2 + v2
    if num_params == 1:
        radial = extra_params[:, 0, None] * r2
        du, dv = u * radial, v * radial
    elif num_params == 2:
        k1, k2 = extra_params[:, 0, None], extra_params[:, 1, None]
        radial = k1 * r2 + k2 * r2 * r2
        du, dv = u * radial, v * radial
    elif num_params == 4:
        k1, k2, p1, p2 = (extra_params[:, index, None] for index in range(4))
        radial = k1 * r2 + k2 * r2 * r2
        du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2)
        dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2)
    else:
        raise ValueError("Unsupported number of distortion parameters")
    return u.clone() + du, v.clone() + dv
