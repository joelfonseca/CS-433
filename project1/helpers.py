"""Some helper functions."""
from costs import *
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
    """Compute the gradient and loss using MSE."""

    N = len(y)
    e = y - tx.T.dot(w)
    if np.isnan(e).any():
        print("e contains NaN")
        print("e.shape", e.shape)
        print("tx.T.shape", tx.T.shape)
        print("w.shape", w.shape)
        print("y.shape", y.shape)
        print(e)
    gradient = -(1/N) * tx.dot(e)

    return gradient

def sigmoid(t):
    """Apply sigmoid function on t."""
    return 1/(1+np.exp(-t))

def calculate_gradient(y, tx, w, lambda_=0):
    """Compute the gradient of loss."""

    epsilon = 10e-6

    true = tx.dot(sigmoid(tx.T.dot(w)) - y) + lambda_*np.linalg.norm(w)
    #test = (calculate_loss(y, tx, w + epsilon, 0) - calculate_loss(y, tx, w - epsilon, 0)) / (2*epsilon)

    """print("true: ", true)
    print("test: ", test)
    print("true-test: ", np.linalg.norm(true-test))"""

    return true

def learning_by_gradient_descent(y, tx, w, gamma, lambda_=0):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """

    loss = compute_log_likelihood(y, tx, w, lambda_)
    gradient = calculate_gradient(y, tx, w, lambda_)
    w = w - gamma * gradient

    return (w, loss)


def init_w(tx):
    """Initializes w with random values in [0,1) based on shape of tx."""
    return np.random.rand(tx.shape[0])[:,np.newaxis]


def accuracy(y, x, w, lower_bound, upper_bound):
    """Computes the accuracy of the predictions."""
    return np.mean(y == predict_labels(w, x, lower_bound, upper_bound))


def predict_labels(weights, data, lower_bound, upper_bound):
    """Generates class predictions given weights, and a test data matrix"""
    threshold = (upper_bound + lower_bound)/2
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= threshold)] = lower_bound
    y_pred[np.where(y_pred > threshold)] = upper_bound
    return y_pred
