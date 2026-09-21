import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluation.td_homes import (
    TD_REFERENCE_FILENAMES,
    compute_pose_metrics,
    load_sample,
    relative_pose_errors,
)


class TDHomesTest(unittest.TestCase):
    def test_td_reference_is_first_for_fp_stru3d(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "FP_Stru3d" / "stru3d_00001"
            for child in ("rgb", "pose", "rgb_td"):
                (scene / child).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16)).save(scene / "rgb_td" / "rgb_td_mirror.png")
            pose = "0 0 0 0 0 1 0 1 0 0.5 0.5 1\n"
            for frame_id in (0, 1):
                Image.new("RGB", (16, 16)).save(scene / "rgb" / f"{frame_id}.png")
                (scene / "pose" / f"{frame_id}.txt").write_text(pose, encoding="utf-8")
            sample = load_sample(root, "FP_Stru3D", "LO", "stru3d_00001", [0, 1])
            self.assertEqual(sample.model_image_paths[0], sample.td_path)
            self.assertEqual(sample.td_path.name, TD_REFERENCE_FILENAMES["FP_Stru3D"])
            self.assertEqual(len(sample.model_image_paths), 3)
            self.assertEqual(sample.gt_extrinsics.shape, (2, 4, 4))

    def test_identical_poses_have_zero_pair_errors(self):
        poses = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
        poses[1, 0, 3] = 1
        poses[2, 1, 3] = 2
        rotation, translation = relative_pose_errors(poses, poses)
        np.testing.assert_allclose(rotation, 0.0, atol=1e-8)
        np.testing.assert_allclose(translation, 0.0, atol=1e-5)
        metrics = compute_pose_metrics(rotation, translation)
        self.assertEqual(metrics["Racc_5"], 100.0)
        self.assertEqual(metrics["Tacc_5"], 100.0)

    def test_single_perspective_view_has_no_scored_pairs(self):
        poses = torch.eye(4, dtype=torch.float64).repeat(1, 1, 1)
        rotation, translation = relative_pose_errors(poses, poses)
        self.assertEqual(rotation.shape, (0,))
        self.assertEqual(translation.shape, (0,))


if __name__ == "__main__":
    unittest.main()
