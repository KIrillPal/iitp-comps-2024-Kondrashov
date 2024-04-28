import numpy as np
from triangular import lower_solve, upper_solve


def gauss_solve(A : np.ndarray, b : np.ndarray, eps=1e-9) -> np.ndarray:
    """
    Solves Ax=b by the Gauss method with major element selection by columns.

    Parameters:
    1) A : np.array - a non-singular matrix (N x N)
    1) b : np.array - a vector of size N
    2) eps : float - a constant to consider values in (-eps, eps) as zeros

    Returns:
    1) x : np.array - a vector of size N
    """
    assert len(A.shape) == 2, "Only 2d np.arrays with the shape (N, N) are allowed"
    assert  A.shape[0] == A.shape[1], f"Non-square matrix can not be inversed! Got shape {A.shape}"
    assert  A.shape[0] == b.shape[0], f"Vector b must have the same size as A.shape[0]"
    
    A = A.T # To solve 'vertically'
     
    N = A.shape[0]
    not_used = np.ones(N)
    pos = np.zeros(N, dtype=np.int64)
    
    concat = np.concatenate((A, b[np.newaxis, :]), axis=0, dtype=A.dtype)

    for i in range(N):
        main_idx = np.abs(concat[i] * not_used).argmax()
        main_elem = concat[i, main_idx]
        
        assert not (-eps < main_elem < eps), "Matrix is singular! Found linear dependence"

        # Perform gauss's step
        coeffs = concat[i] / main_elem
        coeffs[main_idx] = 0

        concat -= concat[:, main_idx, np.newaxis] * coeffs
        concat[:, main_idx] /= main_elem
        not_used[main_idx] = 0    
        pos[i] = main_idx # To perform final transpose
    
    concat = concat[:, pos]
    U = concat[:N]
    c = concat[N]
    return upper_solve(U, c)