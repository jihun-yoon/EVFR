"""
Submodular Optimization (Placeholder)
-------------------------------------
Template for submodular-based sampling.
"""

from .base import BaseSampler
from evfr.utils.exceptions import EVFRException

class SubmodularSampler(BaseSampler):
    def sample(self, embeddings, k):
        raise EVFRException("SubmodularSampler not implemented yet.")
