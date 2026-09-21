import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.mv_recon_td_homes import (
    ReconstructionGroundTruth,
    align_and_score_reconstruction,
    load_pi3_style_vggt_input,
    umeyama,
)
from evaluation.td_homes import SequenceSample


class MVReconTDHomesTest(unittest.TestCase):
    def _sample(self, root: Path) -> SequenceSample:
        td = root / "td.png"
        first = root / "first.png"
        second = root / "second.png"
        Image.new("RGB", (800, 400), (255, 0, 0)).save(td)
        Image.new("RGB", (640, 320), (0, 255, 0)).save(first)
        Image.new("RGB", (640, 320), (0, 0, 255)).save(second)
        return SequenceSample(
            dataset="FP_Stru3D",
            overlap="LO",
            sequence_name="synthetic",
            frame_ids=(0, 1),
            td_path=td,
            image_paths=(first, second),
            gt_extrinsics=np.eye(4)[None].repeat(2, axis=0),
        )

    def test_pi3_style_input_is_td_first_and_14_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            tensor = load_pi3_style_vggt_input(self._sample(Path(directory)))
        self.assertEqual(tuple(tensor.shape), (3, 3, 252, 504))
        self.assertGreater(float(tensor[0, 0].mean()), 0.99)
        self.assertGreater(float(tensor[1, 1].mean()), 0.99)
        self.assertGreater(float(tensor[2, 2].mean()), 0.99)

    def test_pi3_style_input_without_td_contains_only_perspectives(self):
        with tempfile.TemporaryDirectory() as directory:
            tensor = load_pi3_style_vggt_input(
                self._sample(Path(directory)), include_td=False
            )
        self.assertEqual(tuple(tensor.shape), (2, 3, 252, 504))
        self.assertGreater(float(tensor[0, 1].mean()), 0.99)
        self.assertGreater(float(tensor[1, 2].mean()), 0.99)
        self.assertLess(float(tensor[:, 0].mean()), 0.01)

    def test_umeyama_recovers_known_similarity(self):
        source = np.asarray(
            [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 1.0, 2.0]]
        )
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        target = 2.5 * rotation @ source + np.asarray([[3.0], [-4.0], [1.5]])
        scale, estimated_rotation, translation = umeyama(source, target)
        self.assertAlmostEqual(scale, 2.5, places=10)
        np.testing.assert_allclose(estimated_rotation, rotation, atol=1e-10)
        np.testing.assert_allclose(translation, [[3.0], [-4.0], [1.5]], atol=1e-10)

    def test_identity_reconstruction_scores_zero_distance(self):
        y, x = np.mgrid[:4, :5]
        points = np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float64)[None]
        images = np.ones((1, 4, 5, 3), dtype=np.float32) * 0.5
        ground_truth = ReconstructionGroundTruth(
            images=images,
            points=points,
            valid_mask=np.ones((1, 4, 5), dtype=bool),
            intrinsics=np.eye(3)[None],
            extrinsics_w2c=np.eye(4)[None],
        )
        metrics = align_and_score_reconstruction(points.copy(), ground_truth)
        self.assertLess(metrics["Acc-mean"], 1e-10)
        self.assertLess(metrics["Comp-mean"], 1e-10)
        self.assertEqual(metrics["num-valid-points"], 20)


if __name__ == "__main__":
    unittest.main()
