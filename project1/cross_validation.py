"""Cross-Validation"""
from implementations import *
import numpy as np

def cross_validation(y, x, initial_w, max_iter, k_indices, k, gamma, lambda_, lower_bound, upper_bound, model="least_squares", batch_size=1):


    """Returns the weight vector with the corresponding accuracy for the specific model and parameters."""
    y_test = y[k_indices[k]]
    x_test = x[k_indices[k]]

    k_indices = np.delete(k_indices, k, 0)
    k_indices = k_indices.flatten()

    y_train = y[k_indices]
    x_train = x[k_indices]

    if model == "logistic_regression":
        (w_tr, loss_tr) = logistic_regression(y_train, x_train.T, initial_w, max_iter, gamma)
    elif model == "least_squares":
        (w_tr, loss_tr) = least_squares(y_train, x_train.T)
    elif model == "ridge_regression":
        (w_tr, loss_tr) = ridge_regression(y_train, x_train.T, lambda_)
    elif model == "least_squares_GD":
        (w_tr, loss_tr) = least_squares_GD(y_train, x_train.T, initial_w, max_iter, gamma)
    elif model == "least_squares_SGD":
        (w_tr, loss_tr) = least_squares_SGD(y_train, x_train.T, initial_w, max_iter, gamma, batch_size)
    elif model == "reg_logistic_regression":
        (w_tr, loss_tr) = reg_logistic_regression(y_train, x_train.T, initial_w, max_iter, gamma, lambda_)
    else:
        raise ValueError("Unknown model: %s" % model)

    acc = accuracy(y_test, x_test, w_tr, lower_bound, upper_bound)

    return w_tr, acc

def build_k_indices(y, k_fold, seed):
    """Builds k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
