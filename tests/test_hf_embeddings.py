import unittest
from PIL import Image
import numpy as np

from evfr.embeddings.hf_base import HuggingFaceDinoV2Embedding


class TestHuggingFaceDinoV2Embedding(unittest.TestCase):

    def setUp(self):
        # Initialize the model once for all tests
        self.model = HuggingFaceDinoV2Embedding()

    def test_extract_single_image(self):
        # Create a dummy PIL Image
        dummy_image = Image.new("RGB", (224, 224), color="red")  # Example size
        embedding = self.model.extract(dummy_image)

        # Check if the output is a numpy array and has the correct shape
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (1, 768))  # Example dimension

    def test_extract_batch_images(self):
        # Create a list of dummy PIL Images
        dummy_images = [
            Image.new("RGB", (224, 224), color=f"rgb({i},0,0)") for i in range(5)
        ]
        embeddings = self.model.extract_batch(dummy_images, batch_size=2)

        # Check embeddings shape
        self.assertEqual(embeddings.shape, (5, 768))

    def test_invalid_input_type(self):
        # Test with an invalid input type
        with self.assertRaises(TypeError):
            self.model.extract("not an image")

    def test_default_model_name(self):
        self.assertEqual(self.model._default_model_name(), "facebook/dinov2-base")


if __name__ == "__main__":
    unittest.main()
