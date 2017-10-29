"""Compute the loss for different models."""
import numpy as np

def compute_loss(y, tx, w):
    """Compute the gradient and loss using MSE."""
    N = len(y)
    e = y - tx.dot(w)
    loss = 1/(2*N) * np.sum(e**2, axis=0)
    
    return loss

def sigmoid(t):
    """Apply sigmoid function on t."""
    return 1/(1+np.exp(-t))

def calculate_loss(y, tx, w, lambda_=0):
    """Compute the cost by negative log likelihood."""

    #print(y.shape, tx.shape, w.shape)

    #  exp_ = np.exp(tx.dot(w))
    #print("exp_: ", exp_)
    #  log_ = np.log(1 + exp_)
    #  y_ = y * tx.dot(w)
    #sum_test1 = np.sum(log_, axis=0)
    #sum_test2 = np.sum(y_, axis=0)
    # sum_ = np.sum(log_ - y_ , axis=0)
    #  reg_term = (lambda_/2)*np.linalg.norm(w)**2

    #print("\nexp_: ", exp_.max(), "\nlog_: ", log_, "\ny_: ", y_, "\nsum_: ", sum_)
    #print("exp_: ", exp_)

    #return sum_ + reg_term

    return 0