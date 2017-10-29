"""Cross-Validation"""
from implementations import *
from features_eng import *
import numpy as np

def cross_validation(y, x, initial_w, max_iter, k_indices, k, gamma, lambda_, lower_bound, upper_bound, model="least_squares"):
    """Returns the weight vector with the corresponding accuracy for the specific model and parameters."""
    y_test = y[k_indices[k]]
    x_test = x[k_indices[k]]

    k_indices = np.delete(k_indices, k, 0)
    k_indices = k_indices.flatten()

    y_train = y[k_indices]
    x_train = x[k_indices]

    if model == "logistic_regression":
        (w_tr, loss_tr) = logistic_regression(y_train, x_train, initial_w, max_iter, gamma)
    elif model == "least_squares":
        (w_tr, loss_tr) = least_squares(y_train, x_train)
    elif model == "ridge_regression":
        (w_tr, loss_tr) = ridge_regression(y_train, x_train, lambda_)
    elif model == "least_squares_GD":
        (w_tr, loss_tr) = least_squares_GD(y_train, x_train, initial_w, max_iter, gamma)
    elif model == "least_squares_SGD":
        (w_tr, loss_tr) = least_squares_SGD(y_train, x_train, initial_w, max_iter, gamma)
    elif model == "reg_logistic_regression":
        (w_tr, loss_tr) = reg_logistic_regression(y_train, x_train, initial_w, max_iter, gamma, lambda_)
    else:
        raise ValueError("Unknown model: %s" % model)

    acc = accuracy(y_test, x_test, w_tr, lower_bound, upper_bound)

    return w_tr, acc

def cross_validation_demo(y, tx, model="least_squares", degrees=[1], lambdas=[0], gammas=[0], max_iters=[50], k_fold=10, lower_bound=-1, upper_bound=1):
    """Do a cross-validation with the given parameters on the given model"""

    k_indices = build_k_indices(y, k_fold)
    results = []
    for degree in degrees:
        tx_poly = build_poly_tx(tx, degree)
        initial_w = init_w(tx_poly)
        for max_iter in max_iters:
            for gamma in gammas:
                for lambda_ in lambdas:
                    accs = []
                    ws = []
                    for k in range(k_fold):
                        w_tr, acc = cross_validation(y, tx_poly, initial_w,
                                                     int(max_iter), k_indices, k, gamma, lambda_, lower_bound, upper_bound, model)
                        ws.append(w_tr)
                        accs.append(acc)

                    w_mean = np.mean(ws, axis=0)
                    acc_mean = np.mean(accs)
                    results.append((degree, max_iter, gamma, lambda_, acc_mean, w_mean))

                    print("Finished: " + str((degree, max_iter, gamma, lambda_, acc_mean)))
    
    return results

def build_k_indices(y, k_fold, seed=1):
    """Builds k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    return np.array(k_indices)
