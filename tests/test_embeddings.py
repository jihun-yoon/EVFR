import unittest
import numpy as np
from evfr.embeddings.example_embedding import ExampleEmbeddingModel

class TestEmbeddings(unittest.TestCase):
    def test_example_embedding_output_shape(self):
        model = ExampleEmbeddingModel()
        frame = np.random.rand(224, 224, 3)
        embedding = model.extract(frame)
        self.assertEqual(len(embedding), 512)

if __name__ == '__main__':
    unittest.main()
