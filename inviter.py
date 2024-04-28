import numpy as np
from numpy.linalg import norm 
from random import random, randrange
from gauss import gauss_solve
from triangular import (
    is_lower_triangular,
    is_upper_triangular,
    lower_solve,
    upper_solve
)


def randsign():
    return [-1,1][randrange(2)]


def normalize(x : np.ndarray) -> np.ndarray:
    return x / norm(x)


def get_eigenvector_by_inviter(
    A : np.ndarray,
    eigenvalue : float,
    eps : float = 1e-5,
    noise_ratio : float = 1e-2,
    max_iters : int = None
) -> np.ndarray:
    """
    Finds an eigenvector of the matrix A for the given eigenvalue
    by inverse iteration method. 
    
    Parameters:
    1) A : np.array (N x N)  - square matrix with the given eigenvalue.
    2) eigenvalue : np.float - one of the eigenvalues of the matrix A.
    3) eps : float           - precision of the eigenvector to be found. 1e-5 by default. 
       If the eigenvalue is complex with non-zero imaginary part, use max_iters instead.
    4) noise_ratio : float   - the ratio of perturbation (noising) of the eigenvalue. 0.01 by default.
       Increasing leads to increasing the stability of the method, but decreasing its convergence rate.
    5) max_iters : int       - upper limit on the iteration number. If None, there is no limit.
    
    Returns:
    1) x : np.array (N) - found approximation of eigenvector.
    """
    assert len(A.shape) == 2, "Only square np.arrays with the shape (N, N) are allowed"
    assert A.shape[0] == A.shape[1], f"Non-square matrix are not allowed! Got shape {A.shape}"
    
    # Select the best method to solve (A-shift)x_i = x_(i-1)
    if is_lower_triangular(A):
        solve_method = lower_solve
    elif is_upper_triangular(A):
        solve_method = upper_solve
    else:
        solve_method = gauss_solve
    
    # Initialize starting x
    x = np.random.random(A.shape[0]).astype(A.dtype)
    if norm(x) == 0:
        x = np.ones_like(x, dtype=A.dtype)
    x = normalize(x)
    
    # Noise the eigenvalue to get initial shift
    noise = randsign() * max(random(), 0.1) * noise_ratio
    shift = eigenvalue * (1 + noise) * np.identity(A.shape[0], dtype=A.dtype)
    
    # Descent
    last_x = np.zeros_like(x, dtype=x.dtype)
    
    iter = 0
    while iter != max_iters and min(norm(x - last_x), norm(x + last_x)) >= eps:
        last_x = x
        x = solve_method(A - shift, last_x)
        x = normalize(x)
        
        iter += 1
        # update shift by Rayleigh
        #shift = (x @ (A @ x)) / (x @ x) * np.identity(A.shape[0])
    
    return x