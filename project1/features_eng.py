"""Some features engineering."""
import numpy as np

def build_poly_feature(x, degree):
    """Polynomial basis for input feature x, for j=0 up to j=degree."""

    if x.ndim == 1:
        x = x[:,np.newaxis]

    polynomial_basis = x**1

    for j in range(2, degree+1):
        polynomial_basis = np.hstack((polynomial_basis, x**j))
    return polynomial_basis

def build_poly_tx(tx, degree):
    """Polynomial basis for input features tx, for j=0 up to j=degree."""
    (N,D) = tx.shape
    tx_polynomial = build_poly_feature(tx[:,0], degree)

    for c_idx in range(1, D):
        tx_polynomial = np.hstack((tx_polynomial, build_poly_feature(tx[:,c_idx], degree)))
    ones = np.ones((N,1))
    tx_polynomial = np.hstack((ones, tx_polynomial))

    return tx_polynomial
