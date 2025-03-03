"""
huggingface_base.py
-------------------
A base class for Hugging Face-based embedding models, now supporting batch inference.
"""

import torch
import numpy as np
from abc import abstractmethod
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from torch.utils.data import DataLoader

from evfr.embeddings.base import BaseEmbeddingModel


class HuggingFaceBaseEmbedding(BaseEmbeddingModel):
    """
    Abstract base class for Hugging Face-based embedding models.
    Provides generic logic for:
      - Model loading
      - Image preprocessing (single or batch)
      - Feature extraction (pooler or custom)
      - Optional usage of AutoImageProcessor
    """

    def __init__(
        self,
        model_name=None,
        use_pooler=True,
        device=None,
        use_processor=True,
    ):
        """
        :param model_name: Hugging Face model checkpoint name.
        :param use_pooler: If True, use outputs.pooler_output if available.
        :param device: "cuda" or "cpu" or None. Defaults to cuda if available.
        :param use_processor: If True, apply AutoImageProcessor transforms
                              (resize, normalize, etc.). If False, assume
                              images/tensors are already in correct format.
        """
        super().__init__()
        self.model_name = model_name if model_name else self._default_model_name()
        self.use_pooler = use_pooler
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_processor = use_processor

        # Load model
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # If we want to handle transforms inside the model, load processor
        if self.use_processor:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        else:
            self.processor = None

    @abstractmethod
    def _default_model_name(self):
        pass

    def extract(self, frame):
        """
        Extract embeddings from a single frame (PIL Image or tensor).
        Returns a 1D NumPy array of shape (hidden_dim,).
        """
        # If using processor, we expect a PIL Image
        # If not using processor, we expect a Torch tensor [C, H, W] or a PIL that you manually handle
        if self.use_processor:
            if not isinstance(frame, Image.Image):
                raise TypeError("Frame must be a PIL Image when use_processor=True.")
            inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
        else:
            # We assume 'frame' is already a tensor
            if isinstance(frame, Image.Image):
                # If your dataset returned a PIL image but you're skipping HF transforms,
                # you must manually convert to tensor if needed. Example:
                frame = self._pil_to_tensor(frame)  # user-defined
            # Now frame should be a torch.Tensor of shape [C, H, W]
            inputs = {
                "pixel_values": frame.unsqueeze(0).to(self.device)
            }  # Add batch dimension

        with torch.no_grad():
            outputs = self.model(**inputs)

        return self._postprocess_single(outputs)

    def extract_batch(self, frames, batch_size=8):
        """
        Extract embeddings for a list of frames in batches.
        frames: List of frames (PIL or tensor).
        """
        all_embeddings = []
        for start_idx in range(0, len(frames), batch_size):
            batch_frames = frames[start_idx : start_idx + batch_size]

            if self.use_processor:
                # Expect a list of PIL images
                if any(not isinstance(f, Image.Image) for f in batch_frames):
                    raise TypeError(
                        "All frames must be PIL Images when use_processor=True."
                    )
                inputs = self.processor(images=batch_frames, return_tensors="pt").to(
                    self.device
                )
            else:
                # Expect a list of torch tensors
                tensor_list = []
                for f in batch_frames:
                    if isinstance(f, Image.Image):
                        f = self._pil_to_tensor(f)
                    tensor_list.append(f)
                # Stack them into shape [B, C, H, W]
                inputs = {
                    "pixel_values": torch.stack(tensor_list, dim=0).to(self.device)
                }

            with torch.no_grad():
                outputs = self.model(**inputs)

            batch_embeddings = self._postprocess_batch(outputs)
            all_embeddings.append(batch_embeddings)

        return np.concatenate(all_embeddings, axis=0)

    def extract_dataset(self, dataset, batch_size=8, num_workers=0):
        """
        Extract embeddings from an entire dataset.
        Creates a DataLoader internally.

        :param dataset: A PyTorch Dataset returning either PIL Images or Tensors
                        (depending on self.use_processor).
        :param batch_size: DataLoader batch size
        :param num_workers: Number of workers for DataLoader
        :return: NumPy array of shape (N, hidden_dim), where N=len(dataset)
        """
        # Build DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        all_embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                # batch could be a list of PIL Images or a single tensor of shape [B, C, H, W]
                if self.use_processor:
                    # Expect a list of PIL images in batch
                    inputs = self.processor(images=batch, return_tensors="pt").to(
                        self.device
                    )
                else:
                    # Expect a tensor of shape [B, C, H, W]
                    # If your dataset returns a list of PIL, you'll need to convert inside the Dataset or here
                    if isinstance(batch, list):
                        # Possibly the dataset returns a list of PIL even though use_processor=False.
                        # Convert each to tensor
                        batch_tensors = []
                        for f in batch:
                            batch_tensors.append(self._pil_to_tensor(f))
                        batch = torch.stack(batch_tensors, dim=0)
                    inputs = {"pixel_values": batch.to(self.device)}

                outputs = self.model(**inputs)
                batch_embeddings = self._postprocess_batch(outputs)
                all_embeddings.append(batch_embeddings)

        return np.concatenate(all_embeddings, axis=0)

    def _postprocess_single(self, outputs):
        """
        Postprocess for a single image (batch_size=1).
        Returns a 1D NumPy array (hidden_dim,).
        """
        if (
            self.use_pooler
            and hasattr(outputs, "pooler_output")
            and outputs.pooler_output is not None
        ):
            embedding = outputs.pooler_output.squeeze(0)
        else:
            hidden_states = outputs.last_hidden_state.squeeze(0)
            embedding = torch.mean(hidden_states, dim=0)

        return embedding.cpu().numpy()

    def _postprocess_batch(self, outputs):
        """
        Postprocess for a batch of images.
        Returns a 2D NumPy array (batch_size, hidden_dim).
        """
        if (
            self.use_pooler
            and hasattr(outputs, "pooler_output")
            and outputs.pooler_output is not None
        ):
            # shape: (batch_size, hidden_dim)
            batch_embeddings = outputs.pooler_output
        else:
            # shape: (batch_size, seq_len, hidden_dim)
            hidden_states = outputs.last_hidden_state
            # average-pool along seq_len
            batch_embeddings = torch.mean(hidden_states, dim=1)

        return batch_embeddings.cpu().numpy()

    def _pil_to_tensor(self, pil_img):
        """
        Convert a PIL Image to a torch.Tensor [C, H, W].
        Simple example. For real usage, consider using
        torchvision.transforms.functional.to_tensor.
        """
        return torch.from_numpy(np.array(pil_img).transpose(2, 0, 1)).float() / 255.0
