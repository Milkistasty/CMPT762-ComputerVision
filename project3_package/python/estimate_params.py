import numpy as np

def estimate_params(P):
    """
    computes the intrinsic K, rotation R, and translation t from
    given camera matrix P.
    
    Args:
        P: Camera matrix
    """
    # Compute SVD of P
    _, _, V = np.linalg.svd(P)

    # Compute camera center c
    c = V[0:3, -1] / V[-1, -1] 

    # Perform RQ decomposition on P[:, :3]
    K, R = QR_Decomposition(P[:3, :3])

    # Ensure R is a proper rotation matrix
    if np.linalg.det(R) < 0:
        R = -R

    # Compute translation vector t
    t = -R @ c

    return K, R, t

def QR_Decomposition(A):
    eps = 1e-10

    A[2, 2] += eps
    c = -A[2, 2] / np.sqrt(A[2, 2]**2 + A[2, 1]**2)
    s = A[2, 1] / np.sqrt(A[2, 2]**2 + A[2, 1]**2)
    Qx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    R = A @ Qx

    R[2, 2] += eps
    c = R[2, 2] / np.sqrt(R[2, 2]**2 + R[2, 0]**2)
    s = R[2, 0] / np.sqrt(R[2, 2]**2 + R[2, 0]**2)
    Qy = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    R = R @ Qy

    R[1, 1] += eps
    c = -R[1, 1] / np.sqrt(R[1, 1]**2 + R[1, 0]**2)
    s = R[1, 0] / np.sqrt(R[1, 1]**2 + R[1, 0]**2)
    Qz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = R @ Qz

    Q = Qz.T @ Qy.T @ Qx.T

    for n in range(3):
        if R[n, n] < 0:
            R[:, n] = -R[:, n]
            Q[n, :] = -Q[n, :]

    return R, Q