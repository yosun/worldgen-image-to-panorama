import math
import unittest

import numpy as np
import torch
from PIL import Image

from projection import perspective_predictions


class PerspectiveProjectionTests(unittest.TestCase):
    def test_photo_produces_forward_facing_rays_and_preserves_pixels(self):
        pixels = np.arange(60 * 120 * 3, dtype=np.uint8).reshape(60, 120, 3)
        image = Image.fromarray(pixels, "RGB")

        result = perspective_predictions(None, image, device="cpu")
        rays = result["rays"]

        self.assertTrue(np.array_equal(result["rgb"].numpy(), pixels))
        self.assertEqual(tuple(rays.shape), (60, 120, 3))
        self.assertTrue(torch.all(rays[..., 2] > 0).item())
        torch.testing.assert_close(torch.linalg.vector_norm(rays, dim=-1), torch.ones(60, 120))
        self.assertLess(rays[0, 0, 0].item(), 0)
        self.assertLess(rays[0, 0, 1].item(), 0)
        self.assertGreater(rays[-1, -1, 0].item(), 0)
        self.assertGreater(rays[-1, -1, 1].item(), 0)

        left = math.atan2(rays[30, 0, 0].item(), rays[30, 0, 2].item())
        right = math.atan2(rays[30, -1, 0].item(), rays[30, -1, 2].item())
        self.assertAlmostEqual(math.degrees(right - left), 90.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
