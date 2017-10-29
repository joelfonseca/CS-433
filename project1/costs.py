"""Compute the loss for different models."""
import numpy as np

def compute_mse_loss(y, tx, w):
    """Compute the gradient and loss using MSE."""
    N = len(y)
    e = y - tx.dot(w)
    loss = 1/(2*N) * np.sum(e**2, axis=0)
    
    return loss

def compute_log_likelihood(y, tx, w, lambda_=0):
    """Compute the cost by negative log likelihood."""

    exp_ = np.exp(tx.dot(w))
    log_ = np.log(1 + exp_)
    y_ = y * tx.dot(w)
    sum_ = np.sum(log_ - y_ , axis=0)
    reg_term = (lambda_/2)*np.linalg.norm(w)**2

    return sum_ + reg_term