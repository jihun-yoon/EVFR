"""
distance_metrics.py
----------------
A collection of common distance metrics for comparing vectors.
"""

import numpy as np

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def cosine_distance(vec1, vec2):
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-10
    cosine_sim = np.dot(vec1, vec2) / denom
    return 1.0 - cosine_sim

def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))

def minkowski_distance(vec1, vec2, p=3):
    return np.power(np.sum(np.power(np.abs(vec1 - vec2), p)), 1/p)
