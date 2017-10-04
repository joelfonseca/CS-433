# Implementation of methods seen in class and in the labs

def compute_gradient(y, tx, w):
    # Compute the gradient and loss using MSE

    N = len(y)
    e = y - tx.dot(w)
    gradient = -1/N * tx.T.dot(e)

    return gradient

def compute_loss(y, tx, w):
    # Compute the gradient and loss using MSE

    N = len(y)
    e = y - tx.dot(w)
    loss = 1/(2*N) * np.sum(e**2, axis=0)

    return loss

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


    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w)

    return (w, loss)

def ridge_regression(y, tx, lambda_):
    # Ridge regression using normal equations

    (N,D) = tx.shape
    tikhonov_matrix = lambda_ * np.identity(D)
    w = np.linealg.solve((tx.T.dot(tx) + tikhonov_matrix.T.dot(tikhonov_matrix)), tx.T.dot(y))
    loss = compute_loss(y, tx, w)

    return (w, loss)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # Logistic regression using gradient descent or SGD
    return 0

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # Regularized logistic regression using gradient descent or SGD
    return 0
