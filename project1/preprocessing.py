"""Preprocesses and cleans the data set."""
import numpy as np
import random

def standardize(tx):
    """Standardizes the data."""

    mean = np.nanmean(tx, axis=0)
    std = np.nanstd(tx, axis=0)

    centered_data = tx - mean
    std_data = centered_data / std

    return mean, std, std_data

def standardize_predef(tx, mean, std):
    """Standardizes the data."""

    centered_data = tx - mean
    std_data = centered_data / std

    return std_data

def replace_nan_by_median(data):
    """Replaces the NaN values with the median of the corresponding feature."""
    return np.where(np.isnan(data), np.nanmedian(data, axis=0), data)

def get_jet_masks(x):
    """
    Returns 4 masks corresponding to the rows of x with jet num 0, 1, 2 and 3.
    """
    return {
        0: x[:, 22] == 0,
        1: x[:, 22] == 1,
        2: x[:, 22] == 2,
        3: x[:, 22] == 3
    }
