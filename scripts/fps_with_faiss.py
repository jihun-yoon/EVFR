"""
fps_with_faiss.py
------------------
Example usage of FarthestPointSampler class
"""

import time
import numpy as np
from evfr.sampling.fps_faiss import FarthestPointSampler

if __name__ == "__main__":

    # Example 1: Direct array
    data = np.random.rand(10, 4).astype(np.float32)
    
    # CPU L2
    sampler = FarthestPointSampler(metric="L2", device="cpu")
    t0 = time.time()
    selected_arr = sampler.sample(points=data, k=3)
    print(f"[CPU-L2] Selected = {selected_arr}, time={time.time()-t0:.2f}s")
    print("Selected from array:", selected_arr)

    # GPU L2
    sampler = FarthestPointSampler(metric="L2", device="gpu")
    t0 = time.time()
    selected_arr = sampler.sample(points=data, k=3)
    print(f"[GPU-L2] Selected = {selected_arr}, time={time.time()-t0:.2f}s")
    print("Selected from array:", selected_arr)

    # Example 2: .npz list
    # Suppose we have a list of .npz files, each containing one 4D embedding
    npz_files = ["img0.npz", "img1.npz", "img2.npz", ...]  # Example
    
    # CPU cosine
    sampler = FarthestPointSampler(metric="cosine", device="cpu")
    t0 = time.time()
    selected_paths = sampler.sample(paths=npz_files, k=3)
    print(f"[CPU-cosine] Selected = {selected_paths}, time={time.time()-t0:.2f}s")
    print("Selected from paths:", selected_paths)

    # GPU cosine
    sampler = FarthestPointSampler(metric="cosine", device="gpu")
    t0 = time.time()
    selected_paths = sampler.sample(paths=npz_files, k=3)
    print(f"[GPU-cosine] Selected = {selected_paths}, time={time.time()-t0:.2f}s")
    print("Selected from paths:", selected_paths)