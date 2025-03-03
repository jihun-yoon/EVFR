import unittest
import numpy as np
from evfr.sampling.fps import FarthestPointSampler
from evfr.utils.distance_metrics import euclidean_distance
from evfr.utils.exceptions import EVFRException

class TestSampling(unittest.TestCase):
    def test_fps_basic(self):
        sampler = FarthestPointSampler(distance_fn=euclidean_distance)
        embeddings = np.random.rand(5, 128)
        indices = sampler.sample(embeddings, k=2)
        self.assertEqual(len(indices), 2)

    def test_fps_exceptions(self):
        sampler = FarthestPointSampler(distance_fn=euclidean_distance)
        embeddings = np.random.rand(3, 128)
        with self.assertRaises(EVFRException):
            sampler.sample(embeddings, k=4)

if __name__ == '__main__':
    unittest.main()
