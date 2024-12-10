import numpy as np
from scipy.linalg import svd

def estimate_pose(x, X):
    """
    computes the pose matrix (camera matrix) P given 2D and 3D
    points.
    
    Args:
        x: 2D points with shape [2, N]
        X: 3D points with shape [3, N]
    """
    # Ensure inputs are numpy arrays
    x = x.T
    X = X.T
    N = x.shape[0]
    A = []

    for i in range(N):
        xi = x[i, 0]
        yi = x[i, 1]
        Xi = X[i, 0]
        Yi = X[i, 1]
        Zi = X[i, 2]

        row1 = [Xi, Yi, Zi, 1, 0, 0, 0, 0, -xi * Xi, -xi * Yi, -xi * Zi, -xi]
        row2 = [0, 0, 0, 0, Xi, Yi, Zi, 1, -yi * Xi, -yi * Yi, -yi * Zi, -yi]

        A.append(row1)
        A.append(row2)

    A = np.array(A)
    # Compute SVD of A
    _, _, V = np.linalg.svd(A)
    # Solution is last row of Vt
    P_vector = V[-1]
    # Reshape to 3x4 matrix
    P = P_vector.reshape(3, 4)
    
    return P
