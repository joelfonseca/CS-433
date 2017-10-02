# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE / MAE
    # ***************************************************
        
    N = len(y)
    e = y-tx.dot(w)
    
    # MSE
    # ***************************************************
    loss = 1/(2*N) * np.sum(e**2, axis=0)
    
    # MAE
    # ***************************************************
    #loss = 1/N * np.sum(np.abs(e), axis=0)
    
    return loss