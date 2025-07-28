import pandas as pd
import random
import numpy as np
import torch

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  


def add_nan(y, min_nan=1, max_nan=None):
    """
    Randomly sets several elements per row in y to NaN.
    
    Parameters:
        y (np.ndarray): A NumPy array of shape (n, k).
        min_nan (int): Minimum number of NaNs per row.
        max_nan (int): Maximum number of NaNs per row. If None, defaults to k.
        
    Returns:
        np.ndarray: A copy of y with NaNs inserted.
    """
    y_nan = y.copy()
    n, k = y_nan.shape
    if max_nan is None:
        max_nan = k-1

    for i in range(n):
        num_nan = np.random.randint(min_nan, max_nan + 1)  # number of NaNs for this row
        nan_indices = np.random.choice(k, size=num_nan, replace=False)
        y_nan[i, nan_indices] = np.nan
        
    return y_nan