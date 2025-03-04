"""
fps_faiss.py
--------------------
Farthest Point Sampling (FPS) using Faiss for fast nearest-neighbor searches.
"""

import numpy as np
import faiss
import random
from typing import Optional, List, Union


class FarthestPointSampler:
    """
    A class that performs Farthest Point Sampling (FPS) using Faiss for efficient
    nearest-neighbor searches. Supports both L2 and cosine distance metrics,
    and can run on either CPU or GPU.
    """

    def __init__(
        self,
        metric: str = "L2",
        device: str = "cpu",
        seed: int = 0,
        npz_key: str = "embedding"
    ):
        """
        Initialize the FPS sampler.

        Parameters
        ----------
        metric : {'L2', 'cosine'}
            Distance metric to use:
            - 'L2': squared L2 distance
            - 'cosine': 1 - cosine_similarity
        device : {'cpu', 'gpu'}
            Device to run Faiss index on
        seed : int
            Random seed for initial point selection
        npz_key : str
            Key to load data from .npz files if using file paths
        """
        self.metric = metric
        self.device = device
        self.seed = seed
        self.npz_key = npz_key
        
        if metric not in ["L2", "cosine"]:
            raise ValueError("metric must be 'L2' or 'cosine'")
        if device not in ["cpu", "gpu"]:
            raise ValueError("device must be 'cpu' or 'gpu'")

    def sample(
        self,
        points: Optional[np.ndarray] = None,
        paths: Optional[List[str]] = None,
        k: int = 1,
    ) -> np.ndarray:
        """
        Perform FPS sampling to select k points.

        Parameters
        ----------
        points : np.ndarray, optional
            Data points of shape (N, d). Float32 recommended.
        paths : list of str, optional
            List of .npz file paths containing embeddings
        k : int
            Number of points to sample

        Returns
        -------
        np.ndarray
            Indices of selected points, shape (k,)
        """
        # Load data if needed
        if points is None and paths is not None:
            points = self._load_from_paths(paths)
        elif points is None and paths is None:
            raise ValueError("Must provide either points array or paths list")

        # Convert to float32 if needed
        if points.dtype != np.float32:
            points = points.astype(np.float32)

        # Initialize Faiss index
        index = self._build_index(points)
        
        N = points.shape[0]
        data_for_index = (
            self._normalize(points) if self.metric == "cosine" else points
        )

        # Initialize with random point
        random.seed(self.seed)
        init_idx = random.randrange(N)
        
        dist2S = self._compute_distance_to_query(
            index, data_for_index, init_idx, N
        )
        selected = [init_idx]

        # Main FPS loop
        for _ in range(1, k):
            next_idx = int(np.argmax(dist2S))
            selected.append(next_idx)
            new_dists = self._compute_distance_to_query(
                index, data_for_index, next_idx, N
            )
            np.minimum(dist2S, new_dists, out=dist2S)

        return np.array(selected, dtype=np.int64)

    def _load_from_paths(self, paths: List[str]) -> np.ndarray:
        """Load and stack embeddings from npz files."""
        all_embeddings = []
        for p in paths:
            emb = np.load(p)[self.npz_key]
            if emb.ndim == 1:
                emb = emb[np.newaxis, :]
            all_embeddings.append(emb)
        return np.vstack(all_embeddings)

    def _build_index(self, points: np.ndarray) -> faiss.Index:
        """Build and return appropriate Faiss index."""
        d = points.shape[1]
        if self.metric == "L2":
            index = faiss.IndexFlatL2(d)
        else:  # cosine
            index = faiss.IndexFlatIP(d)
            points = self._normalize(points)
        
        index.add(points)
        
        if self.device == "gpu":
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            
        return index

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """L2 normalize points for cosine similarity."""
        y = np.copy(x)
        faiss.normalize_L2(y)
        return y

    def _compute_distance_to_query(
        self,
        index: faiss.Index,
        data: np.ndarray,
        query_idx: int,
        N: int
    ) -> np.ndarray:
        """Compute distances from all points to a query point."""
        query_vec = data[query_idx : query_idx + 1]
        D, I = index.search(query_vec, k=N)

        dist_arr = np.zeros(N, dtype=np.float32)
        for col in range(N):
            pt_idx = I[0, col]
            if self.metric == "L2":
                dist_arr[pt_idx] = D[0, col]
            else:  # cosine
                dist_arr[pt_idx] = 1.0 + D[0, col]

        return dist_arr