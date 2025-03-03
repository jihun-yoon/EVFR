"""
distributed_sampler.py
----------------------
A custom distributed sampler for inference in DDP contexts.

Usage Example:
    from torch.utils.data import DataLoader
    from evfr.utils.distributed_sampler import CustomDistributedInferenceSampler

    dataset = MyCustomDataset(...)
    sampler = CustomDistributedInferenceSampler(
        dataset_size=len(dataset),
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False  # or True if you want random order
    )
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=4)

Then each DDP process can do:
    for batch in dataloader:
        ...
"""

import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class CustomDistributedInferenceSampler(Sampler):
    """
    A simple distributed sampler for inference, splitting the dataset among
    multiple processes (GPUs) without overlap.
    """

    def __init__(
        self,
        dataset_size: int,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        """
        :param dataset_size: Total number of samples in the dataset
        :param num_replicas: Number of processes participating in DDP (world_size)
        :param rank: Current process index
        :param shuffle: Whether to shuffle indices before splitting
        :param seed: Random seed for shuffling
        """
        super().__init__(None)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed

        # number of samples per replica
        self.samples_per_replica = math.ceil(dataset_size / self.num_replicas)
        # total size (padded) so that all replicas have the same number of indices
        self.total_size = self.samples_per_replica * self.num_replicas

    def __iter__(self):
        # 1. Create a list of all indices
        indices = list(range(self.dataset_size))

        # 4. Slice for this rank
        start = self.rank * self.samples_per_replica
        end = start + self.samples_per_replica
        subset = indices[start:end]

        return iter(subset)

    def __len__(self):
        return len(self._cached_subset) if self._cached_subset is not None else 0
