import numpy as np
import sys

def patch_chromadb_numpy():
    """
    Monkey patch to fix ChromaDB compatibility with NumPy 2.0
    """
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
    if not hasattr(np, 'int_'):
        np.int_ = np.int64
    if not hasattr(np, 'uint'):
        np.uint = np.uint64

    # Add to sys.modules to ensure it's available
    sys.modules['numpy'].float_ = np.float64
    sys.modules['numpy'].int_ = np.int64
    sys.modules['numpy'].uint = np.uint64 