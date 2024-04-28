import numpy as np


def lower_solve(L : np.ndarray, b : np.ndarray):
    """
    Solves the equation Lx = b where L is a lower-triangular matrix.
    
    Parameters:
    1) L : np.array - a lower-triangular matrix (N x N) with non-zero diagonal values.
    2) b : np.array - a float vector of size N
    
    Returns:
    1) x : np.array - a float vector of size N
    """
    assert len(L.shape) == 2, "Only square np.arrays with the shape (N, N) are allowed"
    assert L.shape[0] == L.shape[1], f"Non-square matrix are not allowed! Got shape {L.shape}"
    N = L.shape[0]
    
    x = np.zeros(N, dtype=L.dtype)
    
    for i in range(N):
        s = np.dot(x[:i], L[i,:i])
        x[i] = (b[i] - s) / L[i, i]
    return x


def upper_solve(U : np.ndarray, b : np.ndarray):
    """
    Solves the equation Ux = b where L is an upper-triangular matrix.
    
    Parameters:
    1) U : np.array - an upper-triangular matrix (N x N)
    2) b : np.array - a float vector of size N
    
    Returns:
    1) x : np.array - a float vector of size N
    """
    assert len(U.shape) == 2, "Only square np.arrays with the shape (N, N) are allowed"
    assert U.shape[0] == U.shape[1], f"Non-square matrix are not allowed! Got shape {U.shape}"
    N = U.shape[0]
    
    x = np.zeros(N, dtype=U.dtype)
    
    for i in reversed(range(N)):
        s = np.dot(x[i+1:], U[i,i+1:])
        x[i] = (b[i] - s) / U[i, i]
    return x


def is_lower_triangular(matrix : np.ndarray):
    return np.allclose(matrix, np.tril(matrix))


def is_upper_triangular(matrix : np.ndarray):
    return np.allclose(matrix, np.triu(matrix))