"""Models implementation."""
from helpers import *
from costs import *
import numpy as np

def least_squares_GD(y, tx, initial_w, max_iter, gamma):
    """Linear regression using gradient descent."""

    w = initial_w
    for n_iter in range(max_iter):

        gradient = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * gradient

    return (w, loss)

def least_squares_SGD(y, tx, initial_w, max_iter, gamma, batch_size=1):
    """Linear regression using stochastic gradient descent."""

    w = initial_w
    gradient = 0
    for n_iter in range(max_iter):

        for minibatch_y, minibatch_tx in batch_iter(y, tx.T, batch_size):
            gradient = compute_gradient(minibatch_y, minibatch_tx.T, w)
        
        loss = compute_mse(y, tx, w)
        w = w - gamma * gradient

    return (w, loss)

def least_squares(y, tx):
    """Least squares regression using normal equations."""

    w = np.linalg.solve(tx.dot(tx.T), tx.dot(y))
    loss = compute_mse(y, tx, w)

    return (w, loss)

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""

    (D,N) = tx.shape
    tikhonov_matrix = lambda_*2*N * np.identity(D)
    w = np.linalg.solve((tx.dot(tx.T) + tikhonov_matrix), tx.dot(y))
    loss = compute_mse(y, tx, w)

    return (w, loss)

def logistic_regression(y, tx, initial_w, max_iter, gamma, lambda_=0):
    """Logistic regression using gradient descent."""

    w = initial_w
    for n_iter in range(max_iter):
        (w, loss) = learning_by_gradient_descent(y, tx, w, gamma, lambda_)

    return (w, loss)

def reg_logistic_regression(y, tx, initial_w, max_iter, gamma, lambda_):
    """Regularized logistic regression using gradient descent."""
    return logistic_regression(y, tx, initial_w, max_iter, gamma, lambda_)