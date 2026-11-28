import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from seamless import generate_seamless_pano, make_boundary_repair_mask, make_overlap_mask, soften_wrap


class SeamlessTests(unittest.TestCase):
    def setUp(self):
        self.height, self.width = 128, 256
        self.source = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.source[20:108, 64:192] = (20, 80, 140)
        self.mask = np.full((self.height, self.width), 255, dtype=np.uint8)
        self.mask[20:108, 64:192] = 0

    def test_overlap_exposes_photo_boundary_to_model_but_preserves_core(self):
        new_mask, alpha = make_overlap_mask(Image.fromarray(self.mask))
        new_mask = np.asarray(new_mask)
        self.assertEqual(new_mask[64, 128], 0)
        self.assertEqual(alpha[64, 128], 1)
        self.assertEqual(new_mask[20, 128], 255)
        self.assertEqual(alpha[20, 128], 0)
        self.assertTrue(np.any((alpha > 0) & (alpha < 1)))

    def test_wrap_blend_only_changes_edges_and_matches_join(self):
        rgb = np.zeros((4, 50, 3), dtype=np.uint8)
        rgb[:, :5] = 10
        rgb[:, -5:] = 100
        fixed = np.asarray(soften_wrap(Image.fromarray(rgb), band=5))
        np.testing.assert_array_equal(fixed[:, 0], fixed[:, -1])
        np.testing.assert_array_equal(fixed[:, 5:-5], rgb[:, 5:-5])

    def test_boundary_repair_straddles_edge_and_protects_source_core(self):
        repair_mask, alpha = make_boundary_repair_mask(Image.fromarray(self.mask))
        repair_mask = np.asarray(repair_mask)
        self.assertEqual(repair_mask[64, 128], 0)
        self.assertEqual(alpha[64, 128], 0)
        self.assertEqual(repair_mask[64, 64], 255)
        self.assertEqual(repair_mask[64, 62], 255)
        self.assertEqual(repair_mask[64, 0], 0)
        self.assertEqual(alpha[64, 0], 0)
        self.assertTrue(np.any((alpha > 0) & (alpha < 1)))

    def test_generation_preserves_core_and_optional_pass_repairs_only_wrap(self):
        calls = []

        def fill(_model, image, mask, **kwargs):
            calls.append(np.asarray(mask).copy())
            color = (200, 40, 30) if len(calls) == 1 else (10, 220, 20)
            return Image.new("RGB", image.size, color)

        def mapper(_predictions, device):
            return Image.fromarray(self.source), Image.fromarray(self.mask)

        worldgen = SimpleNamespace(device="cpu", pano_gen_model=None, resolution=self.width)
        image = Image.new("RGB", (40, 20))
        for repair_boundary, repair_wrap, expected_calls in ((False, False, 1), (True, False, 2), (True, True, 3)):
            calls.clear()
            result = np.asarray(generate_seamless_pano(
                worldgen, "street", image, repair_boundary=repair_boundary, repair_wrap=repair_wrap, fill_fn=fill,
                resize_fn=lambda im: im, mapper=mapper))
            self.assertEqual(len(calls), expected_calls)
            np.testing.assert_array_equal(result[64, 128], self.source[64, 128])
            self.assertEqual(calls[0][20, 128], 255)
            np.testing.assert_array_equal(result[:, 0], result[:, -1])
            if repair_boundary:
                self.assertEqual(calls[1][64, 64], 255)
                self.assertEqual(calls[1][64, 128], 0)
            if repair_wrap:
                self.assertTrue(np.any(calls[-1][:, 128] == 255))
                self.assertTrue(np.all(calls[-1][:, 0] == 0))


if __name__ == "__main__":
    unittest.main()
