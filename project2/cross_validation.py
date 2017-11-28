import numpy as np

def build_k_indices(data_and_targets, k_fold, seed=1):
    """Builds k indices for k-fold."""
    num_row = len(data_and_targets)
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    return np.array(k_indices)
