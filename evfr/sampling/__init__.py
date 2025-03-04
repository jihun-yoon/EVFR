"""
# evfr/sampling/__init__.py
--------------------
Sampling algorithms for EVFR.
"""

from .fps_faiss import farthest_point_sampling_faiss

__all__ = [
    "farthest_point_sampling_faiss",
    # ...
]
