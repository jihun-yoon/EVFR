"""
huggingface_swin.py
-------------------
A Hugging Face embedding class for DinoV2 checkpoints.
"""

from evfr.embeddings.hf_base import HuggingFaceBaseEmbedding


class HuggingFaceDinoV2Embedding(HuggingFaceBaseEmbedding):
    """
    Concrete implementation for a DinoV2 model on Hugging Face.
    """

    def _default_model_name(self):
        # Return a default checkpoint if user doesn't specify one
        return "facebook/dinov2-base"
