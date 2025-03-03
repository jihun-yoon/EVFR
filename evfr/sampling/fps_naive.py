"""
Farthest Point Sampling
-----------------------
Selects k points by iteratively choosing the point furthest
from the already chosen set in embedding space.
"""

import numpy as np
from .base import BaseSampler
from evfr.utils.exceptions import EVFRException

class FarthestPointSampler(BaseSampler):
    def __init__(self, distance_fn):
        self.distance_fn = distance_fn

    def sample(self, embeddings, k):
        if k <= 0:
            raise EVFRException("Number of samples k must be positive.")
        if len(embeddings) < k:
            raise EVFRException("Not enough embeddings to sample from.")

        selected_indices = []
        # Start with a random index
        selected_indices.append(np.random.randint(len(embeddings)))

        for _ in range(1, k):
            dist_to_selected = []
            for i in range(len(embeddings)):
                # Distance to the closest selected point
                distances = [self.distance_fn(embeddings[i], embeddings[idx]) 
                             for idx in selected_indices]
                dist_to_selected.append(min(distances))
            next_index = np.argmax(dist_to_selected)
            selected_indices.append(next_index)

        return selected_indices
