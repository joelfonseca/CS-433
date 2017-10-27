"""Preprocesse and clean the data set."""
import numpy as np
import random

def min_max(tx):
    """Applies the min-max scaling."""
    return (tx - np.min(tx, axis=1)[:,np.newaxis]) / (np.max(tx, axis=1)[:,np.newaxis] - np.min(tx, axis=1)[:,np.newaxis])

def standardize(tx):
    """Standardizes the data."""

    mean = np.nanmean(tx, axis=1)[:,np.newaxis]
    std = np.nanstd(tx, axis=1)[:,np.newaxis]

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
    return np.where(np.isnan(data), np.nanmedian(data, axis=1)[:,np.newaxis], data)

def replace_nan_by_mean(data):
    """Replaces the NaN values with the mean of the corresponding feature."""
    return np.where(np.isnan(data), np.nanmean(data, axis=1)[:,np.newaxis], data)

def categorical_rep_data(cat_col):
    """
    Replaces the NaN values of a categorical feature with the most frequent
    occurence.
    """
    cat_col_wo_nan = cat_col.dropna()
    v = cat_col_wo_nan.value_counts().idxmax()

    return cat_col.fillna(v)

def balance(x, y, lower_bound, upper_bound):
    """Balances data with equal number of occurencies s and b."""

    idx_first = np.nonzero(y == upper_bound)[0]
    idx_second = np.nonzero(y == lower_bound)[0]

    size_first = idx_first.shape[0]
    size_second = idx_second.shape[0]

    min_ = np.min([size_first, size_second])

    random.shuffle(idx_first)
    random.shuffle(idx_second)

    idx_list = np.concatenate((idx_first[:min_], idx_second[:min_]), axis=0)

    random.shuffle(idx_list)

    y = y[idx_list,:]
    x = x[idx_list,:]

    return x.T, y

def delete_features(tx, threshold):
    """Deletes the idx from tx which pourcentage of nan is higher than the threshold."""
    idx_to_del = []
    for idx_feature in range(tx.shape[0]):
        if np.isnan(tx[idx_feature]).sum()/tx.shape[1] > threshold:
            idx_to_del.append(idx_feature)

    return np.delete(tx,idx_to_del, axis=0), idx_to_del

def delete_features_from_idx(tx, idx_to_del):
    """Deletes the idx from tx."""
    return np.delete(tx,idx_to_del, axis=0)

def get_jet_masks(x):
    """
    Returns 4 masks corresponding to the rows of x with a jet value
    of 0, 1, 2 and 3 respectively.
    """
    return {
        0: x[:, 22] == 0,
        1: x[:, 22] == 1,
        2: x[:, 22] == 2,
        3: x[:, 22] == 3
        #2: (x[:, 22] == 2) | (x[:, 22] == 3)
    }
