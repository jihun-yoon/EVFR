"""
custom_ddp_sampler.py
--------------------
A custom sampler for DistributedDataParallel (DDP) inference.
"""

import math
import torch
from torch.utils.data import Sampler

class CustomDistributedSampler(Sampler):
    """
    A custom sampler that splits the dataset among 'num_replicas' processes,
    each with a different 'rank'. By default, no shuffling is applied.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        """
        :param dataset: The dataset to sample from.
        :param num_replicas: Total number of distributed processes (world_size).
        :param rank: The current process index.
        :param shuffle: If True, randomize the order (not typical for inference).
        """
        if num_replicas is None:
            if not torch.distributed.is_initialized():
                raise RuntimeError("Requires either 'num_replicas' or a default group.")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_initialized():
                raise RuntimeError("Requires either 'rank' or a default group.")
            rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle

        self.num_samples = len(self.dataset)
        self.num_per_replica = math.ceil(self.num_samples / self.num_replicas)

        # If total samples not divisible, last rank might get fewer.

    def __iter__(self):
        indices = list(range(self.num_samples))

        if self.shuffle:
            # For inference, we often don't shuffle, but you can add your logic
            g = torch.Generator()
            g.manual_seed(0)  # or any seed
            indices = torch.randperm(self.num_samples, generator=g).tolist()

        # Partition the data for this rank
        start = self.rank * self.num_per_replica
        end = min(start + self.num_per_replica, self.num_samples)

        indices = indices[start:end]
        return iter(indices)

    def __len__(self):
        # Return how many samples THIS rank sees
        start = self.rank * self.num_per_replica
        end = min(start + self.num_per_replica, self.num_samples)
        return end - start
