"""Some features engineering."""
import numpy as np

def build_poly_feature(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""

    if x.ndim == 1:
        x = x[:,np.newaxis]

    polynomial_basis = x**1

    for j in range(2, degree+1):
        polynomial_basis = np.hstack((polynomial_basis, x**j))
    return polynomial_basis

def build_poly_tx(tx, degree):
    
    x = tx.T

    (N,D) = x.shape
    x_polynomial = build_poly_feature(x[:,0], degree)

    for c_idx in range(1, D):
        x_polynomial = np.hstack((x_polynomial, build_poly_feature(x[:,c_idx], degree)))
    ones = np.ones((N,1))
    x_polynomial = np.hstack((ones, x_polynomial))

    return x_polynomial.T
