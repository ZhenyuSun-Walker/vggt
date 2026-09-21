"""TD-HOMES dataset adapters and relative-pose metrics for VGGT.

The sequence maps and conventions intentionally follow Pi3_early_fusion.  Each
VGGT input is ordered as ``[top-down reference, perspective view 0, ...]``.
The reference prediction is excluded from scoring because these datasets do not
provide a calibrated camera pose for the rendered top-down image.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


DATASET_NAMES = ("FP_Stru3D", "TD_HM3D", "TD_Mansion")
OVERLAP_NAMES = ("LO", "normal", "NO")

MAP_FILENAMES = {
    "FP_Stru3D": {
        "LO": "FPStru3d_relpose_seq-id-map_eval_large_overlap.json",
        "normal": "FPStru3d_relpose_seq-id-map_eval_normal.json",
        "NO": "FPStru3d_relpose_seq-id-map_eval_no_overlap.json",
    },
    "TD_HM3D": {
        "LO": "TDHM3D_relpose_seq-id-map_eval_large_overlap.json",
        "normal": "TDHM3D_relpose_seq-id-map_eval_normal.json",
        "NO": "TDHM3D_relpose_seq-id-map_eval_no_overlap.json",
    },
    "TD_Mansion": {
        "LO": "TDMansion_relpose_seq-id-map_eval_large_overlap.json",
        "normal": "TDMansion_relpose_seq-id-map_eval_normal.json",
        "NO": "TDMansion_relpose_seq-id-map_eval_no_overlap.json",
    },
}

EXPECTED_ALL_OVERLAP_COUNTS = {
    "FP_Stru3D": 903,
    "TD_HM3D": 486,
    "TD_Mansion": 240,
}

# Match the active Pi3_early_fusion configs exactly. In particular,
# FP_Stru3D is evaluated with the mirrored top-down render.
TD_REFERENCE_FILENAMES = {
    "FP_Stru3D": "rgb_td_mirror.png",
    "TD_HM3D": "rgb_td.png",
    "TD_Mansion": "rgb_td.png",
}


@dataclass(frozen=True)
class SequenceSample:
    dataset: str
    overlap: str
    sequence_name: str
    frame_ids: tuple[int, ...]
    td_path: Path
    image_paths: tuple[Path, ...]
    gt_extrinsics: np.ndarray

    @property
    def model_image_paths(self) -> tuple[Path, ...]:
        """VGGT input order, with the TD view fixed as reference index zero."""
        return (self.td_path, *self.image_paths)


def _physical_sequence_name(name: str) -> str:
    return name.split("::", 1)[-1]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_image(stem: Path, extensions: Sequence[str]) -> Path:
    if stem.suffix and stem.is_file():
        return stem
    for extension in extensions:
        candidate = stem.with_suffix(extension)
        if candidate.is_file():
            return candidate
    choices = ", ".join(str(stem.with_suffix(ext)) for ext in extensions)
    raise FileNotFoundError(f"Could not resolve image; tried: {choices}")


def _as_w2c_4x4(matrix: np.ndarray, source: Path) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (3, 4):
        result = np.eye(4, dtype=np.float64)
        result[:3] = matrix
        return result
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 3x4 or 4x4 pose in {source}, got {matrix.shape}")
    return matrix


def _fp_stru3d_pose(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if values.size != 12:
        raise ValueError(f"Expected 12 pose values in {path}, got {values.size}")
    vx, vy, vz, tx, ty, tz, ux, uy, uz, _, _, _ = values
    forward = np.asarray([tx, ty, tz], dtype=np.float64)
    forward /= np.linalg.norm(forward) + 1e-12
    up = np.asarray([ux, uy, uz], dtype=np.float64)
    up /= np.linalg.norm(up) + 1e-12
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-12

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.stack([right, -up, forward], axis=1)
    c2w[:3, 3] = np.asarray([vx, vy, vz]) / 1000.0
    return np.linalg.inv(c2w)


def _fp_stru3d_sample(
    root: Path, overlap: str, sequence_name: str, frame_ids: Sequence[int]
) -> SequenceSample:
    sequence_name = _physical_sequence_name(sequence_name)
    scene = root / sequence_name
    td_dir = scene / "rgb_td"
    td_path = td_dir / TD_REFERENCE_FILENAMES["FP_Stru3D"]
    image_paths = []
    extrinsics = []
    for frame_id in frame_ids:
        image_paths.append(_resolve_image(scene / "rgb" / str(frame_id), (".png", ".jpg", ".jpeg")))
        pose_path = scene / "pose" / f"{frame_id}.txt"
        if not pose_path.is_file():
            raise FileNotFoundError(pose_path)
        extrinsics.append(_fp_stru3d_pose(pose_path))
    if not td_path.is_file():
        raise FileNotFoundError(f"No TD reference image found under {td_dir}")
    return SequenceSample(
        "FP_Stru3D",
        overlap,
        sequence_name,
        tuple(int(value) for value in frame_ids),
        td_path,
        tuple(image_paths),
        np.stack(extrinsics),
    )


def _td_hm3d_sample(
    root: Path, overlap: str, sequence_name: str, frame_ids: Sequence[int]
) -> SequenceSample:
    sequence_name = _physical_sequence_name(sequence_name)
    scene = root / sequence_name
    td_path = scene / "rgb_td" / TD_REFERENCE_FILENAMES["TD_HM3D"]
    image_paths = []
    extrinsics = []
    for encoded_id in frame_ids:
        encoded_id = int(encoded_id)
        frame_id, view_id = divmod(encoded_id, 10)
        stem = f"{frame_id:08d}_{view_id}"
        image_paths.append(_resolve_image(scene / "rgb" / stem, (".jpeg", ".jpg", ".png")))
        camera_path = scene / "camera" / f"{stem}_camera_params.json"
        camera = _read_json(camera_path)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = np.asarray(camera["R_cam2world"], dtype=np.float64)
        c2w[:3, 3] = np.asarray(camera["t_cam2world"], dtype=np.float64)
        extrinsics.append(np.linalg.inv(c2w))
    if not td_path.is_file():
        raise FileNotFoundError(f"No TD reference image found under {scene / 'rgb_td'}")
    return SequenceSample(
        "TD_HM3D",
        overlap,
        sequence_name,
        tuple(int(value) for value in frame_ids),
        td_path,
        tuple(image_paths),
        np.stack(extrinsics),
    )


def _td_mansion_sample(
    root: Path, overlap: str, sequence_name: str, frame_ids: Sequence[int]
) -> SequenceSample:
    sequence_name = _physical_sequence_name(sequence_name)
    scene = root / sequence_name
    td_path = scene / "rgb_td" / TD_REFERENCE_FILENAMES["TD_Mansion"]
    image_paths = []
    extrinsics = []
    for frame_id in frame_ids:
        image_paths.append(_resolve_image(scene / "rgb" / str(frame_id), (".png", ".jpg", ".jpeg")))
        camera_path = scene / "cameras" / f"{frame_id}.json"
        camera = _read_json(camera_path)
        extrinsics.append(_as_w2c_4x4(camera["world_to_camera_cv"], camera_path))
    if not td_path.is_file():
        raise FileNotFoundError(f"No TD reference image found under {scene / 'rgb_td'}")
    return SequenceSample(
        "TD_Mansion",
        overlap,
        sequence_name,
        tuple(int(value) for value in frame_ids),
        td_path,
        tuple(image_paths),
        np.stack(extrinsics),
    )


def load_sample(
    data_root: Path,
    dataset: str,
    overlap: str,
    sequence_name: str,
    frame_ids: Sequence[int],
) -> SequenceSample:
    roots = {
        "FP_Stru3D": data_root / "FP_Stru3d",
        "TD_HM3D": data_root / "TD_HM3D",
        "TD_Mansion": data_root / "TD_mansion",
    }
    loaders = {
        "FP_Stru3D": _fp_stru3d_sample,
        "TD_HM3D": _td_hm3d_sample,
        "TD_Mansion": _td_mansion_sample,
    }
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {dataset}")
    return loaders[dataset](roots[dataset], overlap, sequence_name, frame_ids)


def load_sequence_map(map_dir: Path, dataset: str, overlap: str) -> dict[str, list[int]]:
    try:
        filename = MAP_FILENAMES[dataset][overlap]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset/overlap: {dataset}/{overlap}") from exc
    path = map_dir / filename
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Sequence map must contain a JSON object: {path}")
    return value


def iter_map_entries(
    map_dir: Path, datasets: Iterable[str], overlaps: Iterable[str]
) -> Iterable[tuple[str, str, str, list[int]]]:
    for dataset in datasets:
        for overlap in overlaps:
            for sequence_name, frame_ids in load_sequence_map(map_dir, dataset, overlap).items():
                yield dataset, overlap, sequence_name, frame_ids


def inverse_se3(poses: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(poses)
    rotation_t = poses[..., :3, :3].transpose(-1, -2)
    result[..., :3, :3] = rotation_t
    result[..., :3, 3] = -(rotation_t @ poses[..., :3, 3:4]).squeeze(-1)
    result[..., 3, 3] = 1
    return result


def relative_pose_errors(
    predicted_w2c: torch.Tensor, ground_truth_w2c: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Return Pi3-compatible rotation/translation errors for all view pairs."""
    if predicted_w2c.shape != ground_truth_w2c.shape:
        raise ValueError(
            f"Pose shapes differ: predicted={predicted_w2c.shape}, "
            f"ground_truth={ground_truth_w2c.shape}"
        )
    count = predicted_w2c.shape[0]
    if count < 2:
        # Pi3's allow_partial_no_overlap_eval maps retain a few one-view
        # samples. They are inferred and counted, but have no calibrated
        # perspective-perspective pair to contribute to pose metrics.
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty.copy()
    pair_indices = torch.combinations(torch.arange(count, device=predicted_w2c.device), r=2)
    first, second = pair_indices.unbind(dim=1)
    gt_relative = ground_truth_w2c[second] @ inverse_se3(ground_truth_w2c[first])
    pred_relative = predicted_w2c[second] @ inverse_se3(predicted_w2c[first])

    delta_rotation = gt_relative[:, :3, :3] @ pred_relative[:, :3, :3].transpose(-1, -2)
    cosine = ((torch.diagonal(delta_rotation, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    rotation_error = torch.rad2deg(torch.acos(cosine))

    gt_translation = gt_relative[:, :3, 3]
    pred_translation = pred_relative[:, :3, 3]
    denominator = torch.linalg.vector_norm(gt_translation, dim=1) * torch.linalg.vector_norm(
        pred_translation, dim=1
    )
    translation_cosine = (
        (gt_translation * pred_translation).sum(dim=1) / denominator.clamp_min(1e-15)
    ).abs().clamp(0.0, 1.0)
    translation_error = torch.rad2deg(torch.acos(translation_cosine))
    translation_error = torch.where(
        denominator > 1e-15,
        translation_error,
        torch.full_like(translation_error, 90.0),
    )
    return (
        rotation_error.detach().cpu().double().numpy(),
        translation_error.detach().cpu().double().numpy(),
    )


def calculate_auc_np(
    rotation_error: np.ndarray, translation_error: np.ndarray, max_threshold: int
) -> float:
    """Match the histogram AUC used by Pi3_early_fusion and VGGT's CO3D eval."""
    max_errors = np.maximum(rotation_error, translation_error)
    histogram, _ = np.histogram(max_errors, bins=np.arange(max_threshold + 1))
    normalized = histogram.astype(np.float64) / float(len(max_errors))
    return float(np.mean(np.cumsum(normalized)))


def compute_pose_metrics(
    rotation_error: np.ndarray,
    translation_error: np.ndarray,
    thresholds: Sequence[int] = (5, 15, 30),
) -> dict[str, float]:
    rotation_error = np.asarray(rotation_error, dtype=np.float64)
    translation_error = np.asarray(translation_error, dtype=np.float64)
    if rotation_error.size == 0 or translation_error.size == 0:
        raise ValueError("Cannot compute metrics from an empty error population")
    metrics = {
        "MeanRE": float(np.mean(rotation_error)),
        "MeanTE": float(np.mean(translation_error)),
        "MRE": float(np.median(rotation_error)),
        "MTE": float(np.median(translation_error)),
    }
    for threshold in thresholds:
        metrics[f"Racc_{threshold}"] = float(np.mean(rotation_error < threshold) * 100.0)
        metrics[f"Tacc_{threshold}"] = float(np.mean(translation_error < threshold) * 100.0)
        metrics[f"Auc_{threshold}"] = calculate_auc_np(
            rotation_error, translation_error, threshold
        ) * 100.0
    return metrics


def write_one_row_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def safe_sample_stem(overlap: str, sequence_name: str) -> str:
    sequence_name = _physical_sequence_name(sequence_name)
    safe_name = sequence_name.replace("/", "_").replace("\\", "_")
    return f"{overlap}__{safe_name}"


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")
