import numpy as np
from numpy.linalg import svd
from refineF import refineF

def eightpoint(pts1, pts2, M):
    """
    eightpoint:
        pts1 - Nx2 matrix of (x,y) coordinates
        pts2 - Nx2 matrix of (x,y) coordinates
        M    - max(imwidth, imheight)
    """
    
    # Implement the eightpoint algorithm
    # Generate a matrix F from correspondence '../data/some_corresp.npy'
    F = None

    # Normalize the points by the factor M
    pts1_norm = pts1 / M
    pts2_norm = pts2 / M

    N = pts1_norm.shape[0]
    if N < 8:
        raise ValueError("At least 8 point correspondences are required.")

    # Extract coordinates
    x1 = pts1_norm[:, 0]
    y1 = pts1_norm[:, 1]
    x2 = pts2_norm[:, 0]
    y2 = pts2_norm[:, 1]

    # Construct matrix A based on the correspondences
    A = np.column_stack((
        x2 * x1,      # x2 * x1
        x2 * y1,      # x2 * y1
        x2,           # x2
        y2 * x1,      # y2 * x1
        y2 * y1,      # y2 * y1
        y2,           # y2
        x1,           # x1
        y1,           # y1
        np.ones(N)    # 1
    ))

    # Perform SVD on matrix A
    _, _, V = svd(A)
    F = V[-1].reshape(3, 3)  # Extract the last row of Vt and reshape it to 3x3

    # Enforce the Rank-2 Constraint on F
    U_f, S_f, V_f = svd(F)
    S_f[2] = 0  # Set the smallest singular value to zero
    F_rank2 = U_f @ np.diag(S_f) @ V_f  # Reconstruct F with rank 2

    # Refine F using non-linear minimization (optional but recommended)
    F_refined = refineF(F_rank2, pts1_norm, pts2_norm)

    # Unnormalize the Fundamental matrix
    # Scaling matrix
    scale = np.array([
        [1/M,   0,  0],
        [  0, 1/M,  0],
        [  0,   0,  1]
    ])
    
    F = scale.T @ F_refined @ scale

    return F
