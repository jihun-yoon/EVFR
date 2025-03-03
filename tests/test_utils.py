import unittest
import numpy as np
from evfr.utils.distance_metrics import (
    euclidean_distance, cosine_distance, manhattan_distance, minkowski_distance
)

class TestUtils(unittest.TestCase):
    def test_euclidean_distance(self):
        v1 = np.array([1, 2, 3])
        v2 = np.array([1, 2, 3])
        self.assertAlmostEqual(euclidean_distance(v1, v2), 0.0)

    def test_cosine_distance(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        self.assertAlmostEqual(cosine_distance(v1, v2), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
