"""
file_io.py
--------------------
Utility functions for saving and loading embeddings.
"""

import torch
import numpy as np


def save_embeddings_torch(tensor: torch.Tensor, path: str) -> None:
    """
    Save embeddings as a PyTorch file (.pt).
    """
    torch.save(tensor, path)


def load_embeddings_torch(path: str, map_location="cpu") -> torch.Tensor:
    """
    Load embeddings from a PyTorch file (.pt).
    """
    return torch.load(path, map_location=map_location)


def save_embeddings_np(tensor: torch.Tensor, path: str, compressed=True) -> None:
    """
    Save embeddings as .npz (compressed) or .npy (uncompressed) using NumPy.
    """
    arr = tensor.cpu().numpy()
    if compressed:
        np.savez_compressed(path, arr=arr)
    else:
        # remove .npz or .npy from path if you want to handle extension automatically
        np.save(path, arr)


def load_embeddings_np(path: str) -> torch.Tensor:
    """
    Load embeddings from a NumPy file (.npz or .npy) and return as torch.Tensor.
    """
    if path.endswith(".npz"):
        data = np.load(path)
        arr = data[list(data.keys())[0]]  # default key: "arr"
    else:
        arr = np.load(path)
    return torch.from_numpy(arr)
