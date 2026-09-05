"""TD_HM3D and FP_Stru3D loader for MV + unregistered top-down training."""

import json
from pathlib import Path

import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import read_depth


class TopDownFusionDataset(BaseDataset):
    """Return views in the fixed order [MV reference, remaining MV, TD]."""

    def __init__(self, common_conf, data_root, dataset_kind, split="train",
                 train_fraction=0.9, use_td_mirror=True, min_mv_views=2):
        super().__init__(common_conf)
        if dataset_kind not in {"td_hm3d", "fp_stru3d"}:
            raise ValueError("dataset_kind must be 'td_hm3d' or 'fp_stru3d'.")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test.")
        self.root = Path(data_root)
        self.dataset_kind, self.split = dataset_kind, split
        self.training = bool(common_conf.training)
        self.use_td_mirror = use_td_mirror
        self.allow_duplicate_img = bool(getattr(common_conf, "allow_duplicate_img", True))
        self.load_depth = bool(getattr(common_conf, "load_depth", True))
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        scenes = sorted(path for path in self.root.iterdir() if path.is_dir())
        if dataset_kind == "fp_stru3d":
            scenes = [path for path in scenes if path.name.startswith("stru3d_")]
        else:
            scenes = [path for path in scenes if len(path.name) > 6 and path.name[:5].isdigit()]
        self.scenes = [
            path for path in scenes if (path / "rgb").is_dir()
            and any((path / "rgb_td" / name).is_file() for name in self._td_names())
            and len(self._frame_paths(path)) >= min_mv_views
        ]
        split_index = int(len(self.scenes) * train_fraction)
        self.scenes = self.scenes[:split_index] if split == "train" else self.scenes[split_index:]
        if not self.scenes:
            raise RuntimeError(f"No valid {dataset_kind} scenes found for split={split} under {self.root}")
        self.len_train = len(self.scenes)

    def _td_names(self):
        return ("rgb_td_mirror.png", "rgb_td.png") if self.use_td_mirror else ("rgb_td.png", "rgb_td_mirror.png")

    @staticmethod
    def _frame_key(path):
        return tuple(int(part) if part.isdigit() else part for part in path.stem.replace("-", "_").split("_"))

    def _frame_paths(self, scene):
        return sorted((scene / "rgb").glob("*.jpeg" if self.dataset_kind == "td_hm3d" else "*.png"), key=self._frame_key)

    def _td_path(self, scene):
        for name in self._td_names():
            candidate = scene / "rgb_td" / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No top-down image found for scene {scene}")

    @staticmethod
    def _c2w_to_w2c(c2w):
        return np.linalg.inv(c2w).astype(np.float32)[:3]

    def _load_hm3d_pose(self, scene, stem):
        with (scene / "camera" / f"{stem}_camera_params.json").open() as handle:
            camera = json.load(handle)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.asarray(camera["R_cam2world"], dtype=np.float32)
        c2w[:3, 3] = np.asarray(camera["t_cam2world"], dtype=np.float32)
        return self._c2w_to_w2c(c2w), np.asarray(camera["camera_intrinsics"], dtype=np.float32)

    def _load_stru3d_pose(self, scene, stem):
        values = np.loadtxt(scene / "pose" / f"{stem}.txt", dtype=np.float32).reshape(-1)
        if values.size != 12:
            raise ValueError(f"Expected 12 pose values for {scene.name}/{stem}, got {values.size}")
        vx, vy, vz, tx, ty, tz, ux, uy, uz, xfov, yfov, _ = values
        forward = np.asarray((tx, ty, tz), dtype=np.float32); forward /= np.linalg.norm(forward) + 1e-6
        up = np.asarray((ux, uy, uz), dtype=np.float32); up /= np.linalg.norm(up) + 1e-6
        right = np.cross(forward, up); right /= np.linalg.norm(right) + 1e-6
        up = np.cross(right, forward); up /= np.linalg.norm(up) + 1e-6
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.stack((right, -up, forward), axis=1)
        c2w[:3, 3] = np.asarray((vx, vy, vz), dtype=np.float32) / 1000.0
        height, width = 720.0, 1280.0
        intrinsics = np.asarray(((width / 2 / np.tan(xfov), 0, width / 2),
                                 (0, height / 2 / np.tan(yfov), height / 2), (0, 0, 1)), dtype=np.float32)
        return self._c2w_to_w2c(c2w), intrinsics

    @staticmethod
    def _read_rgb(path):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _load_mv(self, scene, image_path, target_shape):
        image = self._read_rgb(image_path)
        if self.dataset_kind == "td_hm3d":
            depth_path = scene / "depth" / f"{image_path.stem}_depth.exr"
            extrinsic, intrinsic = self._load_hm3d_pose(scene, image_path.stem)
            depth = read_depth(str(depth_path)) if self.load_depth else np.zeros(image.shape[:2], dtype=np.float32)
        else:
            depth_path = scene / "depth" / f"{image_path.stem}.png"
            extrinsic, intrinsic = self._load_stru3d_pose(scene, image_path.stem)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise RuntimeError(f"Failed to read depth {depth_path}")
            depth = np.asarray(depth, dtype=np.float32) / 1000.0
            if depth.ndim == 3:
                depth = depth[..., 0]
        return self.process_one_image(image, depth, extrinsic, intrinsic, np.asarray(image.shape[:2]), target_shape, filepath=str(image_path))

    def _load_td(self, scene, target_shape):
        path = self._td_path(scene)
        image = self._read_rgb(path)
        height, width = image.shape[:2]
        depth = np.zeros((height, width), dtype=np.float32)
        intrinsic = np.asarray(((max(height, width), 0, width / 2), (0, max(height, width), height / 2), (0, 0, 1)), dtype=np.float32)
        extrinsic = np.concatenate((np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)), axis=1)
        # Zero depth makes this an intentionally unregistered conditioning view:
        # no TD pose, depth, or point loss is ever constructed.
        return self.process_one_image(image, depth, extrinsic, intrinsic, np.asarray(image.shape[:2]), target_shape, filepath=str(path))

    def get_data(self, seq_index=None, img_per_seq=None, seq_name=None, ids=None, aspect_ratio=1.0):
        scene = self.scenes[seq_index] if seq_name is None else self.root / seq_name
        frames = self._frame_paths(scene)
        if img_per_seq is None:
            raise ValueError("img_per_seq is required")
        if ids is None:
            # Preserve the actual first perspective image as the reference.
            selected = np.random.choice(len(frames) - 1, img_per_seq - 1, replace=self.allow_duplicate_img) + 1
            frame_indices = np.concatenate((np.asarray((0,)), np.sort(selected)))
        else:
            frame_indices = np.asarray(ids, dtype=np.int64)
            if frame_indices[0] != 0:
                frame_indices = np.concatenate((np.asarray((0,)), frame_indices[:-1]))
        target_shape = self.get_target_shape(aspect_ratio)
        processed = [self._load_mv(scene, frames[index], target_shape) for index in frame_indices]
        processed.append(self._load_td(scene, target_shape))
        images, depths, extrinsics, intrinsics, world_points, cam_points, masks, _ = zip(*processed)
        return {
            "seq_name": f"{self.dataset_kind}_{scene.name}", "ids": frame_indices,
            "images": list(images), "depths": list(depths), "extrinsics": list(extrinsics),
            "intrinsics": list(intrinsics), "world_points": list(world_points), "cam_points": list(cam_points),
            "point_masks": list(masks), "is_top_down": np.asarray([False] * len(frame_indices) + [True]),
            "td_num_views": np.int64(1), "reference_frame_idx": np.int64(0),
        }
