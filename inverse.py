import numpy as np


def inverse(matrix : np.ndarray, eps=1e-9) -> np.ndarray:
    """
    Inverses the given matrix by the Gauss method with major element selection by columns.

    Parameters:
    1) matrix : np.array - a non-singular square matrix (N x N)
    2) eps : float - a constant to consider values in (-eps, eps) as zeros

    Returns:
    1) inversed : np.array - inversed matrix (N x N)
    """
    assert len(matrix.shape) == 2, "Only 2d np.arrays with the shape (N, N) are allowed"
    assert  matrix.shape[0] == matrix.shape[1], f"Non-square matrix can not be inversed! Got shape {matrix.shape}"

    matrix = matrix.T # To solve 'vertically'

    N = matrix.shape[0]
    not_used = np.ones(N)
    pos = np.zeros(N, dtype=np.int64)
    
    identity = np.identity(N, dtype=matrix.dtype)
    concat = np.concatenate((matrix, identity), axis=0, dtype=matrix.dtype)

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
    
    inversed = concat[N:, pos]
    return inversed.T