"""
Base Embedding Class
--------------------
Defines a generic interface for embedding extraction.
"""

from abc import ABC, abstractmethod

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def extract(self, frame):
        """
        Extract embedding from a frame (or batch of frames).
        Returns a vector or matrix representing the embedding.
        """
        pass
