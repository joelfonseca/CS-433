import numpy as np
import random



################################################################################
# Asked implementations ########################################################
################################################################################
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Linear regression using gradient descent

    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # update w by gradient
        w = w - gamma * gradient

    return (w, loss)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1):
    # Linear regression using stochastic gradient descent

    w = initial_w
    gradient = 0
    for n_iter in range(max_iters):
        # compute gradient from minibatch
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
        # compute loss
        loss = compute_loss(y, tx, w)
        # update w by gradient
        w = w - gamma * gradient

    return (w, loss)

def least_squares(y, tx):
    # Least squares regression using normal equations

    w = np.linalg.solve(tx.dot(tx.T), tx.dot(y))
    loss = compute_loss(y, tx, w)

    return (w, loss)

def ridge_regression(y, tx, lambda_):
    # Ridge regression using normal equations

    (D,N) = tx.shape
    print('D: ', D, 'N: ', N)
    tikhonov_matrix = lambda_*2*N * np.identity(D)
    w = np.linalg.solve((tx.dot(tx.T) + tikhonov_matrix), tx.dot(y))
    loss = compute_loss(y, tx, w)

    return (w, loss)

def logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_=0):
    # Logistic regression using gradient descent or SGD

    w = initial_w
    for n_iter in range(max_iters):
        (w, loss) = learning_by_gradient_descent(y, tx, w, gamma/(n_iter+1), lambda_)
        """if n_iter%100 == 0:
            print(accuracy(y, tx, w))"""

    return (w, loss)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # Regularized logistic regression using gradient descent or SGD
    return 0

################################################################################
################################################################################
################################################################################

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_gradient(y, tx, w):
    """Compute the gradient and loss using MSE"""

    N = len(y)
    e = y - tx.dot(w)
    gradient = -1/N * tx.T.dot(e)

    return gradient

def compute_loss(y, tx, w):
     """Compute the gradient and loss using MSE."""
     N = len(y)
     e = y - tx.T.dot(w)
     loss = 1/(2*N) * np.sum(e**2, axis=0)

     return loss

def sigmoid(t):
    """Apply sigmoid function on t."""
    return 1/(1+np.exp(-t))

def calculate_loss(y, tx, w, lambda_=0):
    """Compute the cost by negative log likelihood."""

    #print(y.shape, tx.shape, w.shape)

    exp_ = np.exp(tx.dot(w))
    #print("exp_: ", exp_)
    log_ = np.log(1 + exp_)
    y_ = y * tx.dot(w)
    #sum_test1 = np.sum(log_, axis=0)
    #sum_test2 = np.sum(y_, axis=0)
    sum_ = np.sum(log_ - y_ , axis=0)
    reg_term = (lambda_/2)*np.linalg.norm(w)**2

    #print("\nexp_: ", exp_.max(), "\nlog_: ", log_, "\ny_: ", y_, "\nsum_: ", sum_)
    #print("exp_: ", exp_)

    return sum_ + reg_term

def calculate_gradient(y, tx, w, lambda_=0):
    """Compute the gradient of loss."""

    epsilon = 10e-6

    true = tx.T.dot(sigmoid(tx.dot(w)) - y) + lambda_*np.linalg.norm(w)
    #test = (calculate_loss(y, tx, w + epsilon, 0) - calculate_loss(y, tx, w - epsilon, 0)) / (2*epsilon)

    """print("true: ", true)
    print("test: ", test)
    print("true-test: ", np.linalg.norm(true-test))"""

    return true

def learning_by_gradient_descent(y, tx, w, gamma, lambda_=0):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """

    loss = calculate_loss(y, tx, w, lambda_)
    gradient = calculate_gradient(y, tx, w, lambda_)
    w = w - gamma * gradient
    #print("loss: ", loss, " gradient: ", np.linalg.norm(gradient), "w: ", w)

    return (w, loss)

def build_poly_feature(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""

    if x.ndim == 1:
        x = x[:,np.newaxis]

    polynomial_basis = x**1

    for j in range(2, degree+1):
        polynomial_basis = np.vstack((polynomial_basis, x**j))
    return polynomial_basis

def build_poly_tx(tx, degree):

    x = tx.T

    (D,N) = x.shape
    tx_polynomial = build_poly_feature(x[0], degree)

    for c in x[1:]:
        tx_polynomial = np.hstack((tx_polynomial, build_poly_feature(c, degree)))
    ones = np.ones((1,D))
    tx_polynomial = np.vstack((ones, tx_polynomial))

    return tx_polynomial

def init_w(tx):
    """Initializes w with random values in [0,1) based on shape of tx."""
    return np.random.rand(tx.shape[0])[:,np.newaxis]

#def rename_y(higgs_value, background_value, y):
#    """Remaps values of higgs_value and background_value of y vector."""
#    return pd.Series(list(map(lambda x : higgs_value if x=='s' else background_value, named_y)))

def accuracy(y, tx, w, lower_bound, upper_bound):
    """Computes the accuracy of the predictions."""
    return np.mean(y == predict_labels(w, tx, lower_bound, upper_bound))

def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, initial_w, max_iters, k_indices, k, gamma, lambda_, lower_bound, upper_bound, degree=0):
    """return the loss of logistic regression."""
    y_test = y[k_indices[k]]
    x_test = x[k_indices[k]]

    k_indices = np.delete(k_indices, k, 0)
    k_indices = k_indices.flatten()

    y_train = y[k_indices]
    x_train = x[k_indices]

    #basis_train = build_poly(x_train, degree)
    #basis_test = build_poly(x_test, degree)

    (w_tr, loss_tr) = logistic_regression(y_train, x_train, initial_w, max_iters, gamma, lambda_)
    #(w_tr, loss_tr) = least_squares(y_train, x_train.T)
    #(w_tr, loss_tr) = ridge_regression(y_train, x_train.T, lambda_)

    acc = accuracy(y_test, x_test, w_tr, lower_bound, upper_bound)

    return w_tr, acc

def predict_labels(weights, data, lower_bound, upper_bound):
    """Generates class predictions given weights, and a test data matrix"""
    threshold = (upper_bound - lower_bound)/2
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= threshold)] = lower_bound
    y_pred[np.where(y_pred > threshold)] = upper_bound
    return y_pred

import numpy as np
import matplotlib.pyplot as plt

################################################################################
# About plotting results
################################################################################

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    plt.plot(
        degrees,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        label='train',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        label='test',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    plt.ylim(0, 0.4)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.title("Bias-Variance Decomposition")
    plt.savefig("bias_variance")

################################################################################
# About submission
################################################################################

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle.
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

################################################################################
# About cleaning or preparing the data
################################################################################

def min_max(tx):
    """Applies the min-max scaling."""
    return (tx - np.min(tx, axis=1)[:,np.newaxis]) / (np.max(tx, axis=1)[:,np.newaxis] - np.min(tx, axis=1)[:,np.newaxis])

def standardize(tx):
    """Standardizes the data."""

    mean = np.mean(tx, axis=1)[:,np.newaxis]
    std = np.std(tx, axis=1)[:,np.newaxis]

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
    """Balances data with equal number of occurencies s and b"""

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

#create function that transfrom nan into -9999 inverse

def delete_features(tx, threshold):
    idx_to_del = []
    for idx_feature in range(tx.shape[0]):
        if np.isnan(tx[idx_feature]).sum()/tx.shape[1] > threshold:
            idx_to_del.append(idx_feature)

    return np.delete(tx,idx_to_del, axis=0)

# tentative of doing something
#def select_corr(y, data, threshold):
    #data.apply(lambda x: x.corr(y)).sort_values().where(lambda x : abs(x) > 0).dropna().index
