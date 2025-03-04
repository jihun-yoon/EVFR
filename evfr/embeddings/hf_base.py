"""
huggingface_base.py
-------------------
A base class for Hugging Face-based embedding models supporting batch inference.
"""

from typing import List, Optional, Union
import torch
import numpy as np
from abc import abstractmethod
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from evfr.embeddings.base import BaseEmbeddingModel


class HuggingFaceBaseEmbedding(BaseEmbeddingModel):
    """Base class for Hugging Face embedding models with batched inference support.

    Provides functionality for:
    - Model loading and initialization
    - Single and batch image processing
    - Feature extraction with pooling options
    - Optional AutoImageProcessor integration
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_pooler: bool = True,
        device: Optional[str] = None,
        use_processor: bool = True,
    ) -> None:
        """Initialize HuggingFace embedding model.

        Args:
            model_name: HuggingFace model identifier. Uses default if None.
            use_pooler: Whether to use pooler output when available.
            device: Computation device ('cuda'/'cpu'). Auto-detects if None.
            use_processor: Whether to use AutoImageProcessor for preprocessing.
        """
        super().__init__()
        self.model_name = model_name if model_name else self._default_model_name()
        self.use_pooler = use_pooler
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_processor = use_processor

        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.processor = AutoImageProcessor.from_pretrained(self.model_name) if use_processor else None

    @abstractmethod
    def _default_model_name(self) -> str:
        """Return default model identifier if none provided."""
        pass

    def extract(self, frame: Union[Image.Image, torch.Tensor]) -> np.ndarray:
        """Extract embeddings from single frame.

        Args:
            frame: Input image as PIL Image or tensor [C,H,W].

        Returns:
            1D numpy array of shape (hidden_dim,).

        Raises:
            TypeError: If frame format doesn't match processor settings.
        """
        if self.use_processor:
            if not isinstance(frame, Image.Image):
                raise TypeError("Frame must be PIL Image when use_processor=True")
            inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
        else:
            if isinstance(frame, Image.Image):
                frame = self._pil_to_tensor(frame)
            inputs = {"pixel_values": frame.unsqueeze(0).to(self.device)}

        with torch.no_grad():
            outputs = self.model(**inputs)

        return self._postprocess_single(outputs)

    def extract_batch(self, frames: List[Union[Image.Image, torch.Tensor]], batch_size: int = 8) -> np.ndarray:
        """Extract embeddings from multiple frames.

        Args:
            frames: List of frames (PIL Images or tensors).
            batch_size: Number of frames to process at once.

        Returns:
            2D numpy array of shape (n_frames, hidden_dim).
        """
        all_embeddings = []
        for start_idx in range(0, len(frames), batch_size):
            batch_frames = frames[start_idx : start_idx + batch_size]

            if self.use_processor:
                if any(not isinstance(f, Image.Image) for f in batch_frames):
                    raise TypeError("All frames must be PIL Images when use_processor=True")
                inputs = self.processor(images=batch_frames, return_tensors="pt").to(self.device)
            else:
                tensor_list = [
                    self._pil_to_tensor(f) if isinstance(f, Image.Image) else f
                    for f in batch_frames
                ]
                inputs = {"pixel_values": torch.stack(tensor_list, dim=0).to(self.device)}

            with torch.no_grad():
                outputs = self.model(**inputs)

            batch_embeddings = self._postprocess_batch(outputs)
            all_embeddings.append(batch_embeddings)

        return np.concatenate(all_embeddings, axis=0)

    def extract_dataset(self, dataset: Dataset, batch_size: int = 8, num_workers: int = 0) -> np.ndarray:
        """Extract embeddings from a PyTorch dataset.

        Args:
            dataset: PyTorch Dataset returning PIL Images or tensors.
            batch_size: Samples per batch.
            num_workers: DataLoader worker processes.

        Returns:
            2D numpy array of shape (len(dataset), hidden_dim).
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        all_embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                if self.use_processor:
                    inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                else:
                    if isinstance(batch, list):
                        batch = torch.stack([self._pil_to_tensor(f) for f in batch], dim=0)
                    inputs = {"pixel_values": batch.to(self.device)}

                outputs = self.model(**inputs)
                batch_embeddings = self._postprocess_batch(outputs)
                all_embeddings.append(batch_embeddings)

        return np.concatenate(all_embeddings, axis=0)

    def _postprocess_single(self, outputs) -> np.ndarray:
        """Process model outputs for single image.

        Returns:
            1D numpy array of shape (hidden_dim,).
        """
        if self.use_pooler and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embedding = outputs.pooler_output.squeeze(0)
        else:
            hidden_states = outputs.last_hidden_state.squeeze(0)
            embedding = torch.mean(hidden_states, dim=0)

        return embedding.cpu().numpy()

    def _postprocess_batch(self, outputs) -> np.ndarray:
        """Process model outputs for image batch.

        Returns:
            2D numpy array of shape (batch_size, hidden_dim).
        """
        if self.use_pooler and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            batch_embeddings = outputs.pooler_output
        else:
            hidden_states = outputs.last_hidden_state
            batch_embeddings = torch.mean(hidden_states, dim=1)

        return batch_embeddings.cpu().numpy()

    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to normalized tensor.

        Args:
            pil_img: Input PIL Image.

        Returns:
            Tensor of shape [C,H,W] with values in [0,1].
        """
        return torch.from_numpy(np.array(pil_img).transpose(2, 0, 1)).float() / 255.0
