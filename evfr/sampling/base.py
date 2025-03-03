"""
Base Sampling Class
-------------------
Defines a generic interface for sampling algorithms.
"""

from abc import ABC, abstractmethod

class BaseSampler(ABC):
    @abstractmethod
    def sample(self, embeddings, k):
        """
        Select k samples out of the total embeddings.
        
        :param embeddings: A list/array of embeddings.
        :param k: Number of samples to select.
        :return: Indices or embeddings for the selected samples.
        """
        pass
