"""
custom_ddp_sampler.py
--------------------
Implements a custom sampler for distributed inference in PyTorch DDP setups.
Distributes dataset samples across multiple processes efficiently.
"""

from typing import Iterator, List, Optional
import math
import torch
from torch.utils.data import Sampler, Dataset


class CustomDistributedSampler(Sampler[int]):
    """
    Distributed sampler optimized for inference that partitions data across processes.
    
    Divides a dataset into non-overlapping chunks across multiple processes for
    parallel inference. Supports optional shuffling, though typically not used
    during inference.
    """

    def __init__(
        self, 
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False
    ) -> None:
        """
        Initialize the distributed sampler.

        Args:
            dataset: Dataset to sample from
            num_replicas: Number of distributed processes. If None, obtained from distributed group
            rank: Process ID within distributed group. If None, obtained from distributed group
            shuffle: Whether to shuffle indices. Defaults to False for deterministic inference

        Raises:
            RuntimeError: If distributed environment not initialized when num_replicas/rank not provided
        """
        if num_replicas is None:
            if not torch.distributed.is_initialized():
                raise RuntimeError("Distributed environment must be initialized when num_replicas not provided")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_initialized():
                raise RuntimeError("Distributed environment must be initialized when rank not provided")
            rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle

        self.num_samples = len(self.dataset)
        self.num_per_replica = math.ceil(self.num_samples / self.num_replicas)

    def __iter__(self) -> Iterator[int]:
        """
        Generate iteration order for current process's data partition.

        Returns:
            Iterator over sample indices for this process's partition
        """
        indices: List[int] = list(range(self.num_samples))

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(0)
            indices = torch.randperm(self.num_samples, generator=g).tolist()

        start = self.rank * self.num_per_replica
        end = min(start + self.num_per_replica, self.num_samples)

        return iter(indices[start:end])

    def __len__(self) -> int:
        """
        Get number of samples for current process.

        Returns:
            Number of samples in this process's partition
        """
        start = self.rank * self.num_per_replica
        end = min(start + self.num_per_replica, self.num_samples)
        return end - start
