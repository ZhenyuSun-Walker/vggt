"""Pi3-compatible TD-HOMES multi-view reconstruction helpers for VGGT."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree as KDTree
from torchvision.transforms.functional import pil_to_tensor

from vggt.utils.geometry import unproject_depth_map_to_point_map

from evaluation.td_homes import SequenceSample


METRIC_NAMES = (
    "Acc-mean",
    "Acc-med",
    "Comp-mean",
    "Comp-med",
    "NC-mean",
    "NC-med",
    "NC1-mean",
    "NC1-med",
    "NC2-mean",
    "NC2-med",
)


@dataclass(frozen=True)
class ReconstructionGroundTruth:
    images: np.ndarray
    points: np.ndarray
    valid_mask: np.ndarray
    intrinsics: np.ndarray
    extrinsics_w2c: np.ndarray


def _fp_intrinsics(pose_path: Path, width: int, height: int) -> np.ndarray:
    values = np.loadtxt(pose_path, dtype=np.float64).reshape(-1)
    xfov, yfov = values[9], values[10]
    return np.asarray(
        [
            [width / (2.0 * np.tan(xfov)), 0.0, width / 2.0],
            [0.0, height / (2.0 * np.tan(yfov)), height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _read_mansion_depth(path: Path) -> np.ndarray:
    import Imath
    import OpenEXR

    handle = OpenEXR.InputFile(str(path))
    header = handle.header()
    window = header["dataWindow"]
    width = window.max.x - window.min.x + 1
    height = window.max.y - window.min.y + 1
    channels = header["channels"].keys()
    channel = next((name for name in ("Y", "Z", "R") if name in channels), sorted(channels)[0])
    data = handle.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
    depth = np.frombuffer(data, dtype=np.float32).reshape(height, width)
    if hasattr(handle, "close"):
        handle.close()
    return depth


def _frame_calibration_and_depth(
    sample: SequenceSample, image_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    scene = image_path.parent.parent
    stem = image_path.stem
    with Image.open(image_path) as image:
        width, height = image.size

    if sample.dataset == "FP_Stru3D":
        pose_path = scene / "pose" / f"{stem}.txt"
        intrinsic = _fp_intrinsics(pose_path, width, height)
        with Image.open(scene / "depth" / f"{stem}.png") as depth_image:
            depth = np.asarray(depth_image, dtype=np.float32) / 1000.0
        depth[(depth > 10.0) | (depth < 1e-3)] = 0.0
    elif sample.dataset == "TD_HM3D":
        camera_path = scene / "camera" / f"{stem}_camera_params.json"
        import json

        with camera_path.open("r", encoding="utf-8") as handle:
            intrinsic = np.asarray(json.load(handle)["camera_intrinsics"], dtype=np.float64)
        depth = cv2.imread(str(scene / "depth" / f"{stem}_depth.exr"), cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Failed to read TD_HM3D depth: {scene / 'depth' / f'{stem}_depth.exr'}")
        depth = depth.astype(np.float32)
        depth[(depth > 10.0) | (depth < 1e-3)] = 0.0
    elif sample.dataset == "TD_Mansion":
        import json

        camera_path = scene / "cameras" / f"{stem}.json"
        with camera_path.open("r", encoding="utf-8") as handle:
            intrinsic = np.asarray(json.load(handle)["intrinsic_matrix"], dtype=np.float64)
        depth = _read_mansion_depth(scene / "depth" / f"{stem}.exr").astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[(depth < 1e-4) | (depth > 20.0)] = 0.0
    else:
        raise ValueError(f"Unsupported dataset: {sample.dataset}")

    return intrinsic, depth


def _resize_rgb_depth_intrinsic(
    image_path: Path,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    output_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_resolution = np.asarray(depth.shape[::-1], dtype=np.float64)
    output_height = round(input_resolution[1] * (output_width / input_resolution[0]) / 14) * 14
    output_resolution = np.asarray([output_width, output_height], dtype=np.int64)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        max_scale = max(output_resolution[0] / image.width, output_resolution[1] / image.height)
        resample = Image.Resampling.LANCZOS if max_scale < 1 else Image.Resampling.BICUBIC
        image = image.resize(tuple(output_resolution), resample=resample)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    depth = cv2.resize(depth, tuple(output_resolution), interpolation=cv2.INTER_NEAREST)
    intrinsic = intrinsic.copy()
    intrinsic[0, 2] += 0.5
    intrinsic[1, 2] += 0.5
    intrinsic[:2] *= np.max(output_resolution / input_resolution)
    intrinsic[0, 2] -= 0.5
    intrinsic[1, 2] -= 0.5
    return rgb, depth, intrinsic


def load_reconstruction_ground_truth(
    sample: SequenceSample, output_width: int = 512
) -> ReconstructionGroundTruth:
    images, depths, intrinsics = [], [], []
    for image_path in sample.image_paths:
        intrinsic, depth = _frame_calibration_and_depth(sample, image_path)
        image, depth, intrinsic = _resize_rgb_depth_intrinsic(
            image_path, depth, intrinsic, output_width
        )
        images.append(image)
        depths.append(depth)
        intrinsics.append(intrinsic)
    images_array = np.stack(images).astype(np.float32)
    depth_array = np.stack(depths).astype(np.float32)
    intrinsics_array = np.stack(intrinsics).astype(np.float64)
    extrinsics = sample.gt_extrinsics.astype(np.float64)
    points = unproject_depth_map_to_point_map(
        depth_array[..., None], extrinsics[:, :3], intrinsics_array
    )
    return ReconstructionGroundTruth(
        images=images_array,
        points=points,
        valid_mask=depth_array > 1e-4,
        intrinsics=intrinsics_array,
        extrinsics_w2c=extrinsics,
    )


def _pil_tensor(path: Path, size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        return pil_to_tensor(image).float() / 255.0


def load_pi3_style_vggt_input(
    sample: SequenceSample, load_width: int = 512, include_td: bool = True
) -> torch.Tensor:
    """Match Pi3's resize14 perspective path, optionally prepending TD."""
    with Image.open(sample.image_paths[0]) as first:
        target_height = round(first.height * (load_width / first.width) / 14) * 14
    perspective = torch.stack(
        [_pil_tensor(path, (load_width, target_height)) for path in sample.image_paths]
    )
    patch_size = (target_height // 14 * 14, load_width // 14 * 14)
    perspective = F.interpolate(
        perspective, patch_size, mode="bilinear", align_corners=False, antialias=True
    )

    if not include_td:
        return perspective

    with Image.open(sample.td_path) as td_image:
        width, height = td_image.size
        scale = math.sqrt(255000.0 / (width * height))
        approx_width, approx_height = width * scale, height * scale
        k, m = round(approx_width / 14), round(approx_height / 14)
        while (k * 14) * (m * 14) > 255000:
            if k / m > approx_width / approx_height:
                k -= 1
            else:
                m -= 1
        td_pre_size = (max(1, k) * 14, max(1, m) * 14)
    td = _pil_tensor(sample.td_path, td_pre_size).unsqueeze(0)
    td = F.interpolate(td, patch_size, mode="bilinear", align_corners=False, antialias=True)
    return torch.cat([td, perspective], dim=0)


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    mu_source = source.mean(axis=1, keepdims=True)
    mu_target = target.mean(axis=1, keepdims=True)
    variance = np.square(source - mu_source).sum(axis=0).mean()
    covariance = ((target - mu_target) @ (source - mu_source).T) / source.shape[1]
    u, singular, vh = np.linalg.svd(covariance)
    sign = np.eye(source.shape[0])
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        sign[-1, -1] = -1
    scale = np.trace(np.diag(singular) @ sign) / variance
    rotation = u @ sign @ vh
    translation = mu_target - scale * rotation @ mu_source
    return float(scale), rotation, translation


def _accuracy(
    ground_truth: np.ndarray,
    reconstruction: np.ndarray,
    gt_normals: np.ndarray,
    rec_normals: np.ndarray,
    workers: int,
) -> tuple[float, float, float, float]:
    distances, indices = KDTree(ground_truth).query(reconstruction, workers=workers)
    normal_dot = np.abs(np.sum(gt_normals[indices] * rec_normals, axis=-1))
    return (
        float(np.mean(distances)),
        float(np.median(distances)),
        float(np.mean(normal_dot)),
        float(np.median(normal_dot)),
    )


def _completion(
    ground_truth: np.ndarray,
    reconstruction: np.ndarray,
    gt_normals: np.ndarray,
    rec_normals: np.ndarray,
    workers: int,
) -> tuple[float, float, float, float]:
    distances, indices = KDTree(reconstruction).query(ground_truth, workers=workers)
    normal_dot = np.abs(np.sum(gt_normals * rec_normals[indices], axis=-1))
    return (
        float(np.mean(distances)),
        float(np.median(distances)),
        float(np.mean(normal_dot)),
        float(np.median(normal_dot)),
    )


def align_and_score_reconstruction(
    predicted_points: np.ndarray,
    ground_truth: ReconstructionGroundTruth,
    pred_ply: Path | None = None,
    gt_ply: Path | None = None,
    workers: int = -1,
) -> dict[str, float]:
    if predicted_points.shape != ground_truth.points.shape:
        raise ValueError(
            f"Point-map shapes differ: predicted={predicted_points.shape}, "
            f"ground_truth={ground_truth.points.shape}"
        )
    mask = ground_truth.valid_mask
    if np.count_nonzero(mask) < 3:
        raise ValueError("Fewer than three valid GT points")
    scale, rotation, translation = umeyama(
        predicted_points[mask].T, ground_truth.points[mask].T
    )
    predicted_points = (
        scale * np.einsum("...j,ij->...i", predicted_points, rotation)
        + translation.reshape(1, 1, 1, 3)
    )
    predicted = predicted_points[mask].reshape(-1, 3)
    target = ground_truth.points[mask].reshape(-1, 3)
    colors = ground_truth.images[mask].reshape(-1, 3)

    predicted_cloud = o3d.geometry.PointCloud()
    predicted_cloud.points = o3d.utility.Vector3dVector(predicted)
    predicted_cloud.colors = o3d.utility.Vector3dVector(colors)
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target)
    target_cloud.colors = o3d.utility.Vector3dVector(colors)

    registration = o3d.pipelines.registration.registration_icp(
        predicted_cloud,
        target_cloud,
        0.1,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    predicted_cloud.transform(registration.transformation)
    if pred_ply is not None:
        pred_ply.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(pred_ply), predicted_cloud)
    if gt_ply is not None:
        gt_ply.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(gt_ply), target_cloud)

    predicted_cloud.estimate_normals()
    target_cloud.estimate_normals()
    predicted = np.asarray(predicted_cloud.points)
    target = np.asarray(target_cloud.points)
    predicted_normals = np.asarray(predicted_cloud.normals)
    target_normals = np.asarray(target_cloud.normals)
    acc, acc_med, nc1, nc1_med = _accuracy(
        target, predicted, target_normals, predicted_normals, workers
    )
    comp, comp_med, nc2, nc2_med = _completion(
        target, predicted, target_normals, predicted_normals, workers
    )
    return {
        "Acc-mean": acc,
        "Acc-med": acc_med,
        "Comp-mean": comp,
        "Comp-med": comp_med,
        "NC-mean": (nc1 + nc2) / 2.0,
        "NC-med": (nc1_med + nc2_med) / 2.0,
        "NC1-mean": nc1,
        "NC1-med": nc1_med,
        "NC2-mean": nc2,
        "NC2-med": nc2_med,
        "Sim3-scale": scale,
        "ICP-fitness": float(registration.fitness),
        "ICP-inlier-rmse": float(registration.inlier_rmse),
        "num-valid-points": int(len(target)),
    }


def save_image_grid(images: np.ndarray, path: Path) -> None:
    rows = math.floor(math.sqrt(len(images)))
    columns = math.ceil(len(images) / rows)
    height, width = images.shape[1:3]
    grid = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    uint8_images = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    for index, image in enumerate(uint8_images):
        row, column = divmod(index, columns)
        grid[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)
