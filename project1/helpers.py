"""Some helper functions."""
from costs import *
from proj1_helpers import *
import numpy as np

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generates a minibatch iterator for a dataset.
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
    """Computes the gradient and loss using MSE."""

    N = len(y)
    e = y - tx.dot(w)
    gradient = -(1/N) * tx.T.dot(e)

    return gradient

def sigmoid(t):
    """Applies sigmoid function on t."""
    return 1/(1+np.exp(-t))

def compute_gradient_sigmoid(y, tx, w, lambda_=0):
    """Computes the gradient of loss."""

    gradient = tx.T.dot(sigmoid(tx.dot(w)) - y) + lambda_ * np.linalg.norm(w)

    return gradient

def learning_by_gradient_descent(y, tx, w, gamma, lambda_=0):
    """
    Does one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """

    loss = compute_neg_log_likelihood(y, tx, w, lambda_)
    gradient = compute_gradient_sigmoid(y, tx, w, lambda_)
    w = w - gamma * gradient

    return (w, loss)

def init_w(tx, seed=1):
    """Initializes w with random values in [0,1) based on shape of tx."""
    np.random.seed(seed)
    return np.random.rand(tx.shape[1])

def accuracy(y, x, w, lower_bound, upper_bound):
    """Computes the accuracy of the predictions."""
    return np.mean(y == predict_labels(w, x, lower_bound, upper_bound))
